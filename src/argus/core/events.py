"""In-process pub/sub. publish() never blocks and never raises.

Call publish() only from the event-loop thread. Sampler threads should use
loop.call_soon_threadsafe(bus.publish, topic, payload).
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[tuple[str, object]]] = set()

    def publish(self, topic: str, payload: object) -> None:
        for q in tuple(self._subs):
            try:
                q.put_nowait((topic, payload))
            except asyncio.QueueFull:  # slow subscriber: drop its oldest event
                with contextlib.suppress(asyncio.QueueEmpty, asyncio.QueueFull):
                    q.get_nowait()
                    q.put_nowait((topic, payload))

    @asynccontextmanager
    async def subscribe(
        self, maxsize: int = 10_000
    ) -> AsyncIterator[asyncio.Queue[tuple[str, object]]]:
        q: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize)
        self._subs.add(q)
        try:
            yield q
        finally:
            self._subs.discard(q)