"""Turns one stored request row into a timing Breakdown and a plain-English
explanation of what happened.

This works only from what the proxy already measures and what Ollama's own
final chunk reports - there is no process or GPU monitoring yet, so no rule
here claims a cause that data would be needed for. Where that limits how far
a finding can go, its `note` says so rather than guessing.
"""
from __future__ import annotations

from argus.diagnostics.models import (CONFIRMED, LIKELY, POSSIBLE, Breakdown, Diagnosis,
                                      Evidence, Finding, Segment)
from argus.format import fmt_duration

_TOKEN_KINDS = frozenset({"generate", "chat"})

# How much of the total a segment must occupy before a rule built on it fires.
COLD_LOAD_FRACTION = 0.30
QUEUE_FRACTION = 0.15
PROMPT_FRACTION = 0.40
STALL_FRACTION = 0.20
CLIENT_WAIT_FRACTION = 0.20
SLOWDOWN_RATIO = 0.6          # fires when speed drops below this fraction of baseline
MIN_BASELINE_SAMPLES = 5


def decompose(row: dict) -> Breakdown | None:
    """Split a row's total wall time into named segments.

    Returns None when there isn't enough data to say anything (the request
    never got as far as a first token, or basic timestamps are missing).
    """
    total = row.get("t_end_ns")
    sent = row.get("t_sent_ns")
    first = row.get("t_first_out_ns")
    if first is None:
        first = row.get("t_first_byte_ns")
    if total is None or sent is None or first is None or first < sent or total < first:
        return None

    connect = max(0, row.get("conn_ns") or 0)
    client_wait = max(0, row.get("client_wait_ns") or 0)
    has_final = row.get("o_load_ns") is not None or row.get("o_prompt_ns") is not None
    load = row.get("o_load_ns") or 0
    prompt = row.get("o_prompt_ns") or 0
    wait_to_first = max(0, first - sent)

    segments: list[Segment] = [Segment("connect", connect, True)]
    if has_final:
        segments.append(Segment("load", load, True))
        segments.append(Segment("prompt", prompt, True))
        segments.append(Segment("queue", max(0, wait_to_first - load - prompt), False))
    else:
        # No final chunk at all: Argus has no idea how this time split, so it
        # isn't labelled "queue" (that implies a specific, checkable cause).
        segments.append(Segment("other", wait_to_first, False))

    eval_ns = row.get("o_eval_ns")
    if (row.get("kind") in _TOKEN_KINDS) and (total > first or eval_ns):
        if eval_ns is None:
            segments.append(Segment("generate", max(0, total - first), False))
        else:
            segments.append(Segment("generate", eval_ns, True))
            tail = max(0, (total - first) - eval_ns - client_wait)
            if client_wait:
                segments.append(Segment("client_wait", client_wait, True))
            if tail:
                segments.append(Segment("other", tail, False))
    else:
        tail = max(0, total - first)
        if client_wait:
            segments.append(Segment("client_wait", min(client_wait, tail), True))
            tail = max(0, tail - client_wait)
        if tail:
            segments.append(Segment("other", tail, False))

    segments = [s for s in segments if s.ns > 0]
    if not segments:
        return None
    return Breakdown(total_ns=total, segments=segments)


def _cold_load(b: Breakdown) -> Finding | None:
    load = b.ns("load")
    if load <= 0 or b.fraction("load") < COLD_LOAD_FRACTION:
        return None
    return Finding(
        code="COLD_MODEL_LOAD", severity="warn", confidence=CONFIRMED,
        title=f"The model had to be loaded first: {fmt_duration(load)} of your "
              f"{fmt_duration(b.total_ns)} wait",
        explanation="The model wasn't already in memory when this request arrived, so "
                    "Ollama had to load it before it could start. Ollama unloads models "
                    "that have been idle for a while (5 minutes by default).",
        evidence=[Evidence(f"model load took {fmt_duration(load)}")],
        fix="Set keep_alive on your requests (or OLLAMA_KEEP_ALIVE) to a longer duration "
            "if you use this model often, so it stays loaded between requests.",
        impact_ns=load,
    )


def _queued(b: Breakdown, row: dict) -> Finding | None:
    queue = b.ns("queue")
    if queue <= 0 or b.fraction("queue") < QUEUE_FRACTION:
        return None
    at_start = row.get("inflight_at_start") or 0
    same_model = row.get("inflight_same_model") or 0
    if at_start > 0:
        others = (f"{same_model} other request{'s' if same_model != 1 else ''} for the same "
                 f"model were" if same_model else f"{at_start} other request"
                 f"{'s' if at_start != 1 else ''} were") + " already running"
        return Finding(
            code="QUEUED", severity="warn", confidence=LIKELY if same_model else POSSIBLE,
            title=f"Waited {fmt_duration(queue)} before Ollama started on this request",
            explanation=f"{others} when this one arrived. Ollama can only work on a "
                        "limited number of requests per model at once, so this one had to wait.",
            evidence=[Evidence(f"{at_start} request(s) already in flight, {same_model} for "
                               "the same model")],
            fix="Avoid sending multiple requests to the same model at once, or raise "
                "OLLAMA_NUM_PARALLEL if you have memory to spare.",
            impact_ns=queue,
        )
    return Finding(
        code="UNEXPLAINED_WAIT", severity="warn", confidence=POSSIBLE,
        title=f"Waited {fmt_duration(queue)} before Ollama started responding",
        explanation="There was a delay before generation began that Ollama's own timings "
                    "don't account for. No other request was queued at the same time, so "
                    "the cause isn't clear from what Argus can currently see.",
        evidence=[Evidence(f"{fmt_duration(queue)} unaccounted for before the first token",
                           measured=False)],
        note="Process and system monitoring (not yet built into Argus) would help narrow "
             "this down.",
        impact_ns=queue,
    )


def _long_prompt(b: Breakdown, row: dict) -> Finding | None:
    prompt = b.ns("prompt")
    count = row.get("o_prompt_count")
    if prompt <= 0 or b.fraction("prompt") < PROMPT_FRACTION or not count:
        return None
    return Finding(
        code="LONG_PROMPT", severity="info", confidence=CONFIRMED,
        title=f"Reading the prompt took {fmt_duration(prompt)}",
        explanation="The prompt was long enough that Ollama spent most of the time "
                    "processing it rather than generating a reply.",
        evidence=[Evidence(f"{count:,} prompt tokens, {fmt_duration(prompt)} to process")],
        fix="If this is more context than you need, trimming the prompt will speed this up.",
        impact_ns=prompt,
    )


def _slow_generation(b: Breakdown, row: dict, baseline_tps: float | None,
                     baseline_n: int) -> Finding | None:
    generate = b.ns("generate")
    count, ns = row.get("o_eval_count"), row.get("o_eval_ns")
    if (generate <= 0 or not count or not ns or ns <= 0
            or baseline_n < MIN_BASELINE_SAMPLES or not baseline_tps):
        return None
    tps = count / (ns / 1e9)
    if tps >= baseline_tps * SLOWDOWN_RATIO:
        return None
    expected_ns = int(count / baseline_tps * 1e9)
    return Finding(
        code="SLOW_GENERATION", severity="warn", confidence=POSSIBLE,
        title=f"Slower than usual: {tps:.0f} tokens/s (normally {baseline_tps:.0f})",
        explanation="Generation was noticeably slower than this model's recent average. "
                    "Argus can't yet say why - that needs the process and GPU monitoring "
                    "planned for a later release.",
        evidence=[Evidence(f"{tps:.0f} tok/s over the last {baseline_n} clean run"
                           f"{'s' if baseline_n != 1 else ''}, usually {baseline_tps:.0f}")],
        note="CPU, memory and GPU monitoring aren't built into this version of Argus yet.",
        impact_ns=max(0, generate - expected_ns),
    )


def _mid_stream_stall(b: Breakdown, row: dict) -> Finding | None:
    gap = row.get("max_gap_ns") or 0
    if gap <= 0 or b.total_ns <= 0 or gap / b.total_ns < STALL_FRACTION:
        return None
    client_wait = row.get("client_wait_ns") or 0
    if client_wait >= gap * 0.8:
        return None  # that silence is accounted for by the client, not a stall
    at_s = fmt_duration(row.get("max_gap_at_ns"))
    return Finding(
        code="MID_STREAM_STALL", severity="warn", confidence=LIKELY,
        title=f"Paused for {fmt_duration(gap)} in the middle of the answer",
        explanation=f"Generation produced nothing for {fmt_duration(gap)}, starting about "
                    f"{at_s} in. Argus can't yet say why - that needs the process and "
                    "system monitoring planned for a later release.",
        evidence=[Evidence(f"longest gap between chunks: {fmt_duration(gap)}")],
        note="Process and system monitoring aren't built into this version of Argus yet.",
        impact_ns=gap,
    )


def _slow_client(b: Breakdown, row: dict) -> Finding | None:
    wait = b.ns("client_wait")
    if wait <= 0 or b.fraction("client_wait") < CLIENT_WAIT_FRACTION:
        return None
    return Finding(
        code="SLOW_CLIENT", severity="info", confidence=CONFIRMED,
        title=f"Ollama was ready, but the calling app took {fmt_duration(wait)} to read the reply",
        explanation="Argus had data ready to hand over but had to wait for the calling app "
                    "to accept it. The delay is in that app, not in Ollama or the model.",
        evidence=[Evidence(f"{fmt_duration(wait)} spent waiting for the client to read data")],
        impact_ns=wait,
    )


_OUTCOME_INFO: dict[str, tuple[str, str, str, str | None]] = {
    # code -> (title, explanation, confidence, fix)
    "out_of_memory": ("Ran out of memory",
        "Ollama reported it didn't have enough memory (system or GPU) to run this request.",
        CONFIRMED, "Try a smaller model, a smaller context size, or close other GPU-heavy apps."),
    "runner_crashed": ("The model runner crashed",
        "Ollama's own model runner process exited unexpectedly while handling this request.",
        CONFIRMED, "Updating your GPU driver often fixes this; otherwise try a smaller "
        "model or context and check Ollama's log."),
    "context_length": ("The prompt didn't fit in the model's context window",
        "Ollama rejected this request because the prompt (plus history) was too long for "
        "the model's context size.", CONFIRMED, "Shorten the prompt, or raise num_ctx if "
        "you have memory to spare."),
    "model_not_found": ("Model not found",
        "Ollama doesn't have this model. It may need to be pulled first, or the name may "
        "be misspelled.", CONFIRMED, "Run `ollama pull <model>`, or check the model name."),
    "bad_request": ("Ollama rejected the request",
        "Ollama reported the request itself was malformed or missing a required field.",
        CONFIRMED, None),
    "unclassified": ("Ollama reported an error",
        "Ollama returned an error Argus doesn't recognize a specific pattern for.",
        POSSIBLE, None),
    "connection_refused": ("Couldn't reach Ollama",
        "Nothing was listening on Ollama's address. It may not be running.", CONFIRMED,
        "Start Ollama; Argus will pick it back up automatically."),
    "dns_failure": ("Couldn't reach Ollama",
        "The upstream address Argus is configured for couldn't be resolved.", CONFIRMED,
        "Check the upstream address in Argus's configuration."),
    "connect_failed": ("Couldn't reach Ollama",
        "Argus couldn't establish a connection to Ollama.", CONFIRMED,
        "Check that Ollama is running and reachable at its configured address."),
    "timeout": ("Ollama timed out",
        "Ollama didn't respond in time.", CONFIRMED,
        "This can happen under heavy load; try again, or check Ollama isn't stuck."),
    "connection_reset": ("Connection to Ollama dropped",
        "The connection to Ollama closed unexpectedly.", CONFIRMED,
        "Check Ollama is still running; retry the request."),
    "stream_cut": ("Ollama's connection closed before finishing",
        "The connection to Ollama closed partway through the answer.", CONFIRMED,
        "Check Ollama is still running; retry the request."),
    "no_final_chunk": ("The answer stopped without a completion marker",
        "Ollama's stream ended without the usual final message, so this reply may be "
        "incomplete.", CONFIRMED, "Retry the request; check Ollama's log if this repeats."),
}


def _outcome_finding(row: dict, upstream_url: str | None) -> Finding | None:
    outcome = row.get("outcome")
    total = row.get("t_end_ns")
    if outcome in (None, "ok", "unfinished"):
        return None
    if outcome == "client_cancelled":
        return Finding(
            code="CLIENT_CANCELLED", severity="info", confidence=CONFIRMED,
            title="Stopped before finishing",
            explanation="This request was cancelled before Ollama finished responding - "
                        "most often because the app or the person using it stopped waiting.",
            impact_ns=total,
        )
    code = row.get("error_code") or ""
    if code.startswith("http_") and code[5:].isdigit():
        return Finding(
            code="HTTP_ERROR", severity="critical", confidence=CONFIRMED,
            title=f"Ollama returned an error (HTTP {code[5:]})",
            explanation="Ollama rejected this request.", impact_ns=total,
        )
    title, explanation, confidence, fix = _OUTCOME_INFO.get(code, (
        "The request failed", "Argus doesn't have a specific explanation for this failure.",
        POSSIBLE, None))
    if outcome == "upstream_unreachable" and upstream_url and "reach" in title.lower():
        explanation = f"Nothing was listening at {upstream_url}. " + explanation.split(". ", 1)[-1]
    return Finding(
        code=(code or outcome).upper(), severity="critical", confidence=confidence,
        title=title, explanation=explanation, fix=fix, impact_ns=total,
    )


def explain(row: dict, *, baseline_tps: float | None = None, baseline_n: int = 0,
           upstream_url: str | None = None) -> Diagnosis:
    """The one entry point the UI calls: a Breakdown (or None) plus the best
    explanation Argus can currently give, ranked by how much time it explains."""
    breakdown = decompose(row)
    outcome_finding = _outcome_finding(row, upstream_url)
    if outcome_finding is not None:
        return Diagnosis(breakdown=breakdown, primary=outcome_finding, also=[])
    if breakdown is None:
        return Diagnosis(breakdown=None, primary=None, also=[])

    candidates = [f for f in (
        _cold_load(breakdown),
        _queued(breakdown, row),
        _long_prompt(breakdown, row),
        _slow_generation(breakdown, row, baseline_tps, baseline_n),
        _mid_stream_stall(breakdown, row),
        _slow_client(breakdown, row),
    ) if f is not None]
    if not candidates:
        return Diagnosis(breakdown=breakdown, primary=None, also=[])
    candidates.sort(key=lambda f: f.impact_ns or 0, reverse=True)
    return Diagnosis(breakdown=breakdown, primary=candidates[0], also=candidates[1:])