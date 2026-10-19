from argus.core.models import GAP_BUCKETS, ROW_COLUMNS, Outcome, Trace


def test_row_keys_match_schema_exactly():
    t = Trace("POST", "/api/generate")
    t.finish(Outcome.OK)
    assert set(t.to_row()) == set(ROW_COLUMNS)


def test_gap_recording_and_first_write_wins():
    t = Trace("POST", "/api/generate")
    t.stamp("hdr", 100)
    t.stamp("hdr", 999)
    assert t.stamps.hdr == 100
    t._record_gap(5_000_000, t.stamps.recv + 6_000_000)
    assert t.max_gap_ns == 5_000_000
    assert t.max_gap_at_ns == 1_000_000
    assert sum(t.gap_hist) == 1 and len(t.gap_hist) == GAP_BUCKETS


def test_finish_is_idempotent():
    t = Trace("GET", "/x")
    t.finish(Outcome.OK)
    t.finish(Outcome.PROXY_ERROR, "late")
    assert t.outcome is Outcome.OK and t.error_code is None
