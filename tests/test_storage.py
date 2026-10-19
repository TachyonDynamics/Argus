import asyncio

import pytest

from argus.core.events import EventBus
from argus.core.models import Outcome, RequestInfo, Trace
from argus.storage.db import Database
from argus.storage.writer import Writer, should_store


def make_trace(model="m", outcome=Outcome.OK, method="POST", path="/api/generate", inference=True):
    t = Trace(method, path)
    if inference:
        t.provider = "ollama"
        t.request = RequestInfo(kind="generate", model=model, stream=True)
    t.finish(outcome)
    return t


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.init()
    return d


def test_init_is_idempotent_and_uses_wal(db):
    db.init()
    with db.connect() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA user_version").fetchone()[0] == 2


def test_refuses_a_database_from_a_newer_argus(db):
    with db.connect() as c:
        c.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="newer Argus"):
        db.init()


def test_roundtrip_and_keyset_paging(db):
    rows = []
    for i in range(5):
        t = make_trace(model=f"m{i}")
        t.wall_start_ns = 1_000 + i
        rows.append(t.to_row())
    db.insert_requests(rows)
    first = db.list_requests(limit=2)
    assert [r["model"] for r in first] == ["m4", "m3"]
    cursor = (first[-1]["started_wall_ns"], first[-1]["id"])
    second = db.list_requests(limit=2, cursor=cursor)
    assert [r["model"] for r in second] == ["m2", "m1"]
    assert db.get_request(rows[0]["id"])["model"] == "m0"
    assert db.count_requests() == 5


def test_problems_only_excludes_ok_and_cancelled(db):
    ts = [make_trace(outcome=o) for o in (Outcome.OK, Outcome.CLIENT_CANCELLED,
                                          Outcome.UPSTREAM_UNREACHABLE, Outcome.STREAM_ERROR)]
    db.insert_requests([t.to_row() for t in ts])
    problems = db.list_requests(problems_only=True)
    assert {r["outcome"] for r in problems} == {"upstream_unreachable", "stream_error"}


def test_should_store_skips_polling_but_keeps_mutations():
    assert should_store(make_trace())
    assert not should_store(make_trace(method="GET", path="/api/tags", inference=False))
    assert should_store(make_trace(method="POST", path="/api/pull", inference=False))


async def test_writer_batches_flushes_on_stop_and_survives_db_errors(db, tmp_path):
    bus = EventBus()
    w = Writer(bus, db, flush_interval_s=0.05)
    await w.start()
    for _ in range(3):
        bus.publish("trace.finished", make_trace())
    bus.publish("trace.finished", make_trace(method="GET", path="/api/tags", inference=False))
    await asyncio.sleep(0.3)
    assert w.written == 3 and db.count_requests() == 3

    bus.publish("trace.finished", make_trace())      # still queued when we stop
    await w.stop()
    assert db.count_requests() == 4

    broken = Writer(bus, Database(tmp_path / "missing_dir" / "x.db"), flush_interval_s=0.05)
    await broken.start()
    bus.publish("trace.finished", make_trace())
    await asyncio.sleep(0.3)
    assert broken.dropped == 1                        # dropped, not raised
    await broken.stop()