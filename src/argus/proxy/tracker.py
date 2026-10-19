"""Tracks in-flight requests. Argus is the only component that can see
concurrency for certain, which is what makes queueing diagnosable later."""
from __future__ import annotations

from argus.core.events import EventBus
from argus.core.models import Trace


class Tracker:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._inflight: dict[str, Trace] = {}

    def start(self, trace: Trace) -> None:
        if trace.is_inference:
            model = trace.request.model if trace.request else None
            active = [t for t in self._inflight.values() if t.is_inference]
            trace.inflight_at_start = len(active)
            # TODO: normalise "llama3" vs "llama3:latest" before comparing.
            trace.inflight_same_model = sum(
                1 for t in active if model and t.request and t.request.model == model
            )
        self._inflight[trace.id] = trace
        self._bus.publish("trace.started", trace)

    def finish(self, trace: Trace) -> None:
        self._inflight.pop(trace.id, None)
        self._bus.publish("trace.finished", trace)

    def snapshot(self) -> list[Trace]:
        return list(self._inflight.values())