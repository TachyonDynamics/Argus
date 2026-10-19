from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest_asyncio

from argus.config import Config
from argus.core.events import EventBus
from argus.core.models import Trace
from argus.proxy.app import create_proxy_app
from argus.proxy.forwarder import Forwarder, make_client
from argus.proxy.tracker import Tracker
from argus.server import make_server
from fake_ollama import FakeOllama


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ArgusHarness:
    """A real Argus proxy listening on an ephemeral port, plus a client aimed at it."""
    cfg: Config
    bus: EventBus
    tracker: Tracker
    forwarder: Forwarder
    port: int
    client: httpx.AsyncClient
    traces: list[Trace] = field(default_factory=list)

    async def trace(self, n: int = 1, timeout: float = 3.0) -> Trace:
        """Wait until n traces have finished; return the n-th one."""
        for _ in range(int(timeout / 0.01)):
            if len(self.traces) >= n:
                return self.traces[n - 1]
            await asyncio.sleep(0.01)
        raise AssertionError(f"{len(self.traces)} trace(s) finished, expected {n}")


@pytest_asyncio.fixture
async def fake() -> AsyncIterator[FakeOllama]:
    f = FakeOllama()
    await f.start()
    yield f
    await f.stop()


@pytest_asyncio.fixture
async def make_argus(tmp_path: Path) -> AsyncIterator[Callable[[str], Awaitable[ArgusHarness]]]:
    stops: list[Callable[[], Awaitable[None]]] = []

    async def build(upstream_url: str) -> ArgusHarness:
        port = free_port()
        cfg = Config(upstream_url=upstream_url, proxy_port=port, data_dir=tmp_path)
        bus = EventBus()
        tracker = Tracker(bus)
        upstream = make_client(cfg)
        forwarder = Forwarder(cfg, upstream, tracker)
        server = make_server(create_proxy_app(forwarder), host="127.0.0.1", port=port,
                             transparent=True)
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10.0)
        h = ArgusHarness(cfg, bus, tracker, forwarder, port, client)

        subscribed = asyncio.Event()

        async def collect() -> None:
            async with bus.subscribe() as q:
                subscribed.set()
                while True:
                    topic, payload = await q.get()
                    if topic == "trace.finished":
                        h.traces.append(payload)

        collector = asyncio.create_task(collect())
        await subscribed.wait()

        async def stop() -> None:
            collector.cancel()
            await client.aclose()
            server.should_exit = True
            server.force_exit = True
            await task
            await upstream.aclose()

        stops.append(stop)
        return h

    yield build
    for stop in reversed(stops):
        await stop()


@pytest_asyncio.fixture
async def argus(fake: FakeOllama, make_argus) -> ArgusHarness:
    return await make_argus(f"http://127.0.0.1:{fake.port}")