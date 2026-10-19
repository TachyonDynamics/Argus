from argus.diagnostics.engine import decompose, explain
from argus.diagnostics.models import CONFIRMED, LIKELY, POSSIBLE

NS = 1_000_000_000


def base_row(**over):
    row = {
        "outcome": "ok", "kind": "generate", "error_code": None, "status_code": 200,
        "t_sent_ns": 1_000_000, "t_first_out_ns": 1_000_000, "t_first_byte_ns": 1_000_000,
        "t_end_ns": 2_000_000, "conn_ns": 500_000, "client_wait_ns": 0,
        "o_load_ns": 0, "o_prompt_ns": 100_000, "o_prompt_count": 5,
        "o_eval_ns": 900_000, "o_eval_count": 10,
        "inflight_at_start": 0, "inflight_same_model": 0, "max_gap_ns": 0, "max_gap_at_ns": 0,
    }
    row.update(over)
    return row


# ---- decompose() --------------------------------------------------------

def test_decompose_returns_none_without_enough_timestamps():
    assert decompose({"t_sent_ns": None, "t_end_ns": 1}) is None
    assert decompose(base_row(t_end_ns=None)) is None


def test_decompose_returns_none_for_an_impossible_timeline():
    assert decompose(base_row(t_first_out_ns=500, t_sent_ns=1000)) is None  # first before sent


def test_decompose_splits_load_prompt_generate_when_final_chunk_present():
    row = base_row(t_sent_ns=0, t_first_out_ns=5 * NS, t_end_ns=6 * NS,
                   o_load_ns=4 * NS, o_prompt_ns=int(0.9 * NS), o_eval_ns=1 * NS)
    b = decompose(row)
    assert b.total_ns == 6 * NS
    assert b.ns("load") == 4 * NS and b.ns("prompt") == int(0.9 * NS)
    assert b.ns("generate") == 1 * NS
    assert b.ns("queue") == 5 * NS - 4 * NS - int(0.9 * NS)  # tiny leftover, not huge
    assert all(s.measured for s in b.segments if s.kind in ("load", "prompt", "generate", "connect"))
    assert not any(s.measured for s in b.segments if s.kind == "queue")


def test_decompose_labels_pre_first_token_time_as_other_without_a_final_chunk():
    row = base_row(t_sent_ns=0, t_first_out_ns=3 * NS, t_end_ns=3 * NS + 1,
                   o_load_ns=None, o_prompt_ns=None, o_eval_ns=None)
    b = decompose(row)
    assert b.ns("queue") == 0
    assert b.ns("other") >= 3 * NS - 1  # unattributed wait, not mislabelled as "queue"
    assert not any(s.measured for s in b.segments if s.kind == "other")


def test_decompose_estimates_generate_when_stream_was_truncated():
    row = base_row(t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=4 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=None, o_eval_count=None)
    b = decompose(row)
    gen = next(s for s in b.segments if s.kind == "generate")
    assert gen.ns == 3 * NS and gen.measured is False


def test_decompose_puts_client_wait_and_leftover_after_generation():
    row = base_row(t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=5 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=3 * NS, client_wait_ns=int(0.5 * NS))
    b = decompose(row)
    assert b.ns("generate") == 3 * NS
    assert b.ns("client_wait") == int(0.5 * NS)
    assert b.ns("other") == 5 * NS - 1 * NS - 3 * NS - int(0.5 * NS)


def test_decompose_handles_non_token_kinds_like_embed():
    row = base_row(kind="embed", t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=int(1.2 * NS),
                   o_load_ns=0, o_prompt_ns=int(0.9 * NS), o_eval_ns=None, o_eval_count=None)
    b = decompose(row)
    assert not any(s.kind == "generate" for s in b.segments)
    assert b.ns("other") == int(0.2 * NS)


def test_breakdown_largest_and_fraction():
    row = base_row(t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=5 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=4 * NS)
    b = decompose(row)
    assert b.largest.kind == "generate"
    assert 0.79 < b.fraction("generate") < 0.81


# ---- explain(): errors and cancellation suppress timing rules -----------

def test_explain_reports_client_cancelled_without_severity_of_an_error():
    d = explain(base_row(outcome="client_cancelled", t_end_ns=9 * NS))
    assert d.primary.code == "CLIENT_CANCELLED" and d.primary.severity == "info"
    assert d.also == []


def test_explain_maps_known_error_codes_to_specific_guidance():
    d = explain(base_row(outcome="stream_error", error_code="out_of_memory"))
    assert d.primary.code == "OUT_OF_MEMORY" and d.primary.confidence == CONFIRMED
    assert "memory" in d.primary.explanation.lower() and d.primary.fix


def test_explain_falls_back_gracefully_for_an_unrecognised_error_code():
    d = explain(base_row(outcome="http_error", error_code="something_new"))
    assert d.primary.confidence == POSSIBLE and d.primary.title


def test_explain_labels_http_status_errors_even_without_a_classified_code():
    d = explain(base_row(outcome="http_error", error_code="http_500", status_code=500))
    assert "500" in d.primary.title


def test_explain_embeds_the_upstream_url_for_unreachable():
    d = explain(base_row(outcome="upstream_unreachable", error_code="connection_refused"),
               upstream_url="http://127.0.0.1:11435")
    assert "127.0.0.1:11435" in d.primary.explanation


def test_explain_returns_no_finding_for_a_still_running_request():
    d = explain(base_row(outcome="unfinished"))
    assert d.primary is None


# ---- explain(): timing rules on successful requests ----------------------

def test_cold_load_is_reported_when_it_dominates():
    row = base_row(t_sent_ns=0, t_first_out_ns=10 * NS, t_end_ns=11 * NS,
                   o_load_ns=9 * NS, o_prompt_ns=int(0.9 * NS), o_eval_ns=int(0.1 * NS))
    d = explain(row)
    assert d.primary.code == "COLD_MODEL_LOAD" and d.primary.confidence == CONFIRMED
    assert "9.0 s" in d.primary.title or "9 s" in d.primary.title


def test_cold_load_does_not_fire_for_a_small_load_time():
    row = base_row(t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=2 * NS,
                   o_load_ns=int(0.05 * NS), o_prompt_ns=int(0.05 * NS), o_eval_ns=int(0.9 * NS))
    d = explain(row)
    assert d.primary is None or d.primary.code != "COLD_MODEL_LOAD"


def test_queueing_cites_concurrency_when_present():
    row = base_row(t_sent_ns=0, t_first_out_ns=6 * NS, t_end_ns=7 * NS,
                   o_load_ns=0, o_prompt_ns=int(0.1 * NS), o_eval_ns=int(0.9 * NS),
                   inflight_at_start=1, inflight_same_model=1)
    d = explain(row)
    assert d.primary.code == "QUEUED" and d.primary.confidence == LIKELY
    assert "same model" in d.primary.explanation


def test_unexplained_wait_is_downgraded_without_concurrency_evidence():
    row = base_row(t_sent_ns=0, t_first_out_ns=6 * NS, t_end_ns=7 * NS,
                   o_load_ns=0, o_prompt_ns=int(0.1 * NS), o_eval_ns=int(0.9 * NS),
                   inflight_at_start=0)
    d = explain(row)
    assert d.primary.code == "UNEXPLAINED_WAIT" and d.primary.confidence == POSSIBLE
    assert d.primary.note is not None


def test_long_prompt_needs_both_a_dominant_fraction_and_a_token_count():
    row = base_row(t_sent_ns=0, t_first_out_ns=5 * NS, t_end_ns=6 * NS,
                   o_load_ns=0, o_prompt_ns=int(4.5 * NS), o_prompt_count=12_000,
                   o_eval_ns=int(0.5 * NS))
    d = explain(row)
    assert d.primary.code == "LONG_PROMPT"
    assert "12,000" in d.primary.evidence[0].text


def test_slow_generation_needs_enough_baseline_samples():
    row = base_row(t_sent_ns=0, t_first_out_ns=0, t_end_ns=5 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=5 * NS, o_eval_count=10)  # 2 tok/s
    assert explain(row, baseline_tps=40, baseline_n=2).primary is None       # too few samples
    d = explain(row, baseline_tps=40, baseline_n=10)
    assert d.primary.code == "SLOW_GENERATION" and d.primary.confidence == POSSIBLE
    assert d.primary.note is not None


def test_slow_generation_does_not_fire_close_to_baseline():
    row = base_row(t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=int(1.25 * NS),
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=int(0.25 * NS), o_eval_count=10)  # 40 tok/s
    d = explain(row, baseline_tps=42, baseline_n=10)
    assert d.primary is None or d.primary.code != "SLOW_GENERATION"


def test_mid_stream_stall_fires_on_a_large_unexplained_gap():
    row = base_row(t_sent_ns=0, t_first_out_ns=0, t_end_ns=10 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=10 * NS, o_eval_count=5,
                   max_gap_ns=6 * NS, max_gap_at_ns=1 * NS, client_wait_ns=0)
    d = explain(row)
    assert d.primary.code == "MID_STREAM_STALL" and d.primary.confidence == LIKELY


def test_mid_stream_stall_does_not_fire_when_the_gap_is_really_client_backpressure():
    row = base_row(t_sent_ns=0, t_first_out_ns=0, t_end_ns=10 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=10 * NS, o_eval_count=5,
                   max_gap_ns=6 * NS, client_wait_ns=6 * NS)
    d = explain(row)
    assert d.primary is None or d.primary.code != "MID_STREAM_STALL"


def test_slow_client_is_confirmed_not_a_guess():
    row = base_row(t_sent_ns=0, t_first_out_ns=1 * NS, t_end_ns=6 * NS,
                   o_load_ns=0, o_prompt_ns=0, o_eval_ns=1 * NS, o_eval_count=10,
                   client_wait_ns=4 * NS)
    d = explain(row)
    assert d.primary.code == "SLOW_CLIENT" and d.primary.confidence == CONFIRMED


def test_no_finding_when_everything_is_unremarkable():
    row = base_row(t_sent_ns=0, t_first_out_ns=int(0.05 * NS), t_end_ns=int(0.3 * NS),
                   o_load_ns=0, o_prompt_ns=int(0.05 * NS), o_eval_ns=int(0.25 * NS),
                   o_eval_count=10)
    d = explain(row, baseline_tps=40, baseline_n=10)
    assert d.primary is None
    assert d.breakdown is not None            # still enough to draw a waterfall


def test_candidates_are_ranked_by_impact_and_extras_go_to_also():
    # a big cold load AND a smaller (but still qualifying) long-prompt condition,
    # both true at once; queue is kept at 0 so it doesn't add a third candidate.
    row = base_row(t_sent_ns=0, t_first_out_ns=int(9.7 * NS), t_end_ns=10 * NS,
                   o_load_ns=int(5.5 * NS), o_prompt_ns=int(4.2 * NS), o_prompt_count=9000,
                   o_eval_ns=int(0.3 * NS))
    d = explain(row)
    assert d.primary.code == "COLD_MODEL_LOAD"
    assert any(f.code == "LONG_PROMPT" for f in d.also)


def test_diagnosis_with_no_breakdown_data_has_neither_primary_nor_crash():
    d = explain({"outcome": "ok"})
    assert d.primary is None and d.breakdown is None and d.also == []
