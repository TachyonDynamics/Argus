"""Batched, fail-open trace writer.

The proxy never waits on the disk: finished traces arrive over the event bus and
are written in small batches on a dedicated thread. A database error drops the
batch (and counts it) instead of affecting any request.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor

from argus.core.events import EventBus
from argus.core.models import Outcome, Trace
from argus.storage.db import Database

log = logging.getLogger(__name__)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def should_store(trace: Trace) -> bool:
    """Inference calls are always kept. Other traffic is kept only when it changes
    something (pull, create, delete, ...). Clients poll GET /api/tags and friends
    constantly; storing those would bury the requests that matter."""
    if trace.is_inference:
        return True
    return trace.method not in _SAFE_METHODS and trace.outcome is not None


class Writer:
    def __init__(
        self,
        bus: EventBus,
        db: Database,
        *,
        flush_interval_s: float = 0.5,
        max_batch: int = 200,
    ) -> None:
        self._bus = bus
        self._db = db
        self._interval = flush_interval_s
        self._max_batch = max_batch
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="argus-writer")
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self.written = 0
        self.dropped = 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="argus-writer")
        await self._ready.wait()  # subscribed before any request can finish

    async def stop(self) -> None:
        """Flush everything already published, then stop."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._executor.shutdown(wait=True)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        batch: list[dict[str, object]] = []
        deadline: float | None = None
        async with self._bus.subscribe() as q:
            self._ready.set()
            try:
                while True:
                    timeout = None if deadline is None else max(0.0, deadline - loop.time())
                    try:
                        topic, payload = await asyncio.wait_for(q.get(), timeout)
                    except asyncio.TimeoutError:
                        topic, payload = "", None
                    if topic == "trace.finished":
                        self._collect(payload, batch)
                        if batch and deadline is None:
                            deadline = loop.time() + self._interval
                    if batch and (
                        len(batch) >= self._max_batch
                        or (deadline is not None and loop.time() >= deadline)
                    ):
                        await self._flush(batch)
                        batch, deadline = [], None
            finally:  # cancelled by stop(): take what is already queued, then flush
                while not q.empty():
                    topic, payload = q.get_nowait()
                    if topic == "trace.finished":
                        self._collect(payload, batch)
                await self._flush(batch)

    def _collect(self, payload: object, batch: list[dict[str, object]]) -> None:
        try:
            if isinstance(payload, Trace) and should_store(payload):
                batch.append(payload.to_row())
        except Exception:
            self.dropped += 1
            log.exception("could not convert a trace to a row")

    async def _flush(self, batch: list[dict[str, object]]) -> None:
        if not batch:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._db.insert_requests, list(batch))
            self.written += len(batch)
        except Exception:
            self.dropped += len(batch)
            log.exception("dropped %d trace(s): database write failed", len(batch))