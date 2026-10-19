import asyncio

from argus.core.events import EventBus


async def test_publish_with_no_subscribers_does_not_raise():
    EventBus().publish("topic", {"a": 1})  # just must not raise


async def test_subscriber_receives_published_events_in_order():
    bus = EventBus()
    async with bus.subscribe() as q:
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert await q.get() == ("a", 1)
        assert await q.get() == ("b", 2)


async def test_multiple_subscribers_each_get_their_own_copy():
    bus = EventBus()
    async with bus.subscribe() as q1, bus.subscribe() as q2:
        bus.publish("x", "payload")
        assert await q1.get() == ("x", "payload")
        assert await q2.get() == ("x", "payload")


async def test_unsubscribed_queue_receives_nothing_further():
    bus = EventBus()
    async with bus.subscribe() as q:
        bus.publish("before", 1)
        assert await q.get() == ("before", 1)
    bus.publish("after", 2)          # no subscribers left; must not raise
    assert q.empty()


async def test_full_queue_drops_oldest_and_keeps_newest():
    bus = EventBus()
    async with bus.subscribe(maxsize=2) as q:
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("c", 3)          # queue was full: "a" is dropped for "c"
        assert q.qsize() == 2
        assert await q.get() == ("b", 2)
        assert await q.get() == ("c", 3)


async def test_one_slow_subscriber_does_not_block_publish_or_other_subscribers():
    bus = EventBus()
    async with bus.subscribe(maxsize=1) as slow, bus.subscribe(maxsize=100) as fast:
        for i in range(5):
            bus.publish("t", i)      # publish() is sync and must never await
        assert await fast.get() == ("t", 0)
        assert slow.qsize() == 1     # still has its most recent event, not stuck at 0