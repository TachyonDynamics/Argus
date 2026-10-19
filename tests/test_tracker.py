from argus.core.events import EventBus
from argus.core.models import RequestInfo, Trace
from argus.proxy.tracker import Tracker


def make(model=None, inference=True):
    t = Trace("POST", "/api/generate" if inference else "/api/tags")
    if inference:
        t.provider = "ollama"
        t.request = RequestInfo(kind="generate", model=model, stream=True)
    return t


async def test_publishes_started_and_finished():
    bus = EventBus()
    tr = Tracker(bus)
    async with bus.subscribe() as q:
        t = make()
        tr.start(t)
        assert await q.get() == ("trace.started", t)
        tr.finish(t)
        assert await q.get() == ("trace.finished", t)


def test_snapshot_reflects_current_inflight_set():
    tr = Tracker(EventBus())
    a, b = make(), make()
    tr.start(a)
    tr.start(b)
    assert {t.id for t in tr.snapshot()} == {a.id, b.id}   # Trace is unhashable (a plain dataclass)
    tr.finish(a)
    assert [t.id for t in tr.snapshot()] == [b.id]


def test_first_request_sees_no_concurrency():
    tr = Tracker(EventBus())
    t = make(model="llama3")
    tr.start(t)
    assert (t.inflight_at_start, t.inflight_same_model) == (0, 0)


def test_second_request_for_same_model_is_counted():
    tr = Tracker(EventBus())
    tr.start(make(model="llama3"))
    t2 = make(model="llama3")
    tr.start(t2)
    assert (t2.inflight_at_start, t2.inflight_same_model) == (1, 1)


def test_concurrent_different_model_counts_overall_but_not_same_model():
    tr = Tracker(EventBus())
    tr.start(make(model="llama3"))
    t2 = make(model="qwen")
    tr.start(t2)
    assert (t2.inflight_at_start, t2.inflight_same_model) == (1, 0)


def test_finished_requests_no_longer_count_toward_concurrency():
    tr = Tracker(EventBus())
    a = make(model="llama3")
    tr.start(a)
    tr.finish(a)
    t2 = make(model="llama3")
    tr.start(t2)
    assert (t2.inflight_at_start, t2.inflight_same_model) == (0, 0)


def test_non_inference_traffic_does_not_affect_or_gain_concurrency_fields():
    tr = Tracker(EventBus())
    tr.start(make(inference=False))       # a GET /api/tags in flight
    t2 = make(model="llama3")
    tr.start(t2)
    assert (t2.inflight_at_start, t2.inflight_same_model) == (0, 0)


def test_finishing_an_unknown_trace_does_not_raise():
    Tracker(EventBus()).finish(make())     # never started; pop() must tolerate this