"""Composition root: builds the services and runs the proxy and admin servers
side by side on one event loop, in one process."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
from dataclasses import dataclass

import httpx
from starlette.applications import Starlette

from argus import __version__
from argus.config import Config
from argus.core.events import EventBus
from argus.integrity.service import IntegrityService
from argus.proxy.app import create_proxy_app
from argus.proxy.forwarder import Forwarder, make_client
from argus.proxy.tracker import Tracker
from argus.server import Server, make_server
from argus.storage.db import Database
from argus.storage.writer import Writer
from argus.ui.app import create_admin_app
from argus.ui.auth import load_or_create_token

log = logging.getLogger(__name__)
ADMIN_HOST = "127.0.0.1"  # the admin UI is never reachable from other machines


@dataclass
class Services:
    cfg: Config
    bus: EventBus
    tracker: Tracker
    db: Database
    writer: Writer
    client: httpx.AsyncClient
    forwarder: Forwarder
    proxy_app: Starlette
    admin_app: Starlette
    token: str
    integrity: IntegrityService


def build(cfg: Config) -> Services:
    """Create everything. Needs a running event loop (the writer creates an Event)."""
    cfg.ensure_dirs()
    bus = EventBus()
    tracker = Tracker(bus)
    db = Database(cfg.db_path)
    db.init()
    writer = Writer(bus, db)
    client = make_client(cfg)
    forwarder = Forwarder(cfg, client, tracker)
    token = load_or_create_token(cfg.data_dir)
    integrity = IntegrityService(db)
    return Services(
        cfg=cfg, bus=bus, tracker=tracker, db=db, writer=writer, client=client,
        forwarder=forwarder,
        proxy_app=create_proxy_app(forwarder),
        admin_app=create_admin_app(cfg, db, tracker, token, integrity),
        token=token,
        integrity=integrity,
    )


def port_problem(host: str, port: int) -> str | None:
    """Try to bind first, so a busy port gives a clear message instead of a traceback."""
    try:
        socket.create_server((host, port)).close()
    except OSError as exc:
        return f"{host}:{port} is not available ({exc.strerror or exc})"
    return None


def _install_stop(loop: asyncio.AbstractEventLoop, servers: tuple[Server, ...]) -> None:
    """One handler asks both servers to shut down gracefully; a second press exits
    the process immediately rather than waiting out uvicorn's graceful-shutdown
    timeout (see the comment on stop() below for why that's necessary)."""
    presses = 0

    def stop() -> None:
        nonlocal presses
        presses += 1
        if presses > 1:
            # uvicorn's own force_exit only skips its internal bookkeeping waits; the
            # final socket-level wait_closed() still blocks until a stuck connection's
            # transport actually closes, which for a genuinely hung request (a model
            # that stopped responding) only happens once the fixed 5s graceful timeout
            # fires and force-cancels it. A second, explicit "quit now" from the person
            # should not be held hostage to that, so it exits the process immediately.
            print("\nForcing an immediate exit.", flush=True)
            os._exit(1)
        for s in servers:
            s.should_exit = True

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK: Ctrl+Break on Windows
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop)
        except (NotImplementedError, RuntimeError, ValueError):
            # The Windows event loop has no add_signal_handler.
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop))


async def run(cfg: Config) -> int:
    cfg.ensure_dirs()
    for host, port, what in ((cfg.proxy_host, cfg.proxy_port, "The proxy"),
                             (ADMIN_HOST, cfg.admin_port, "The web UI")):
        if (problem := port_problem(host, port)) is not None:
            hint = ""
            if port == cfg.proxy_port:
                hint = (" Is Ollama still on its default port? Move it with the OLLAMA_HOST "
                        "environment variable (for example 127.0.0.1:11435), restart Ollama, "
                        "or pick another port with --proxy-port.")
            print(f"{what} can't start: {problem}.{hint}")
            return 2

    svc = build(cfg)
    await svc.writer.start()
    # Fire-and-forget integrity scan; cancel on shutdown so hashing cannot block exit.
    scan_task = asyncio.create_task(asyncio.to_thread(svc.integrity.scan), name="argus-integrity-scan")
    proxy = make_server(svc.proxy_app, host=cfg.proxy_host, port=cfg.proxy_port, transparent=True)
    admin = make_server(svc.admin_app, host=ADMIN_HOST, port=cfg.admin_port)
    _install_stop(asyncio.get_running_loop(), (proxy, admin))

    print(f"Argus {__version__}")
    print(f"  Proxy   http://{cfg.proxy_host}:{cfg.proxy_port}  ->  {cfg.upstream_url}")
    print(f"  Web UI  http://{ADMIN_HOST}:{cfg.admin_port}/?t={svc.token}")
    print("Press Ctrl+C to stop.", flush=True)
    try:
        await asyncio.gather(proxy.serve(), admin.serve())
    finally:
        svc.integrity.cancel()
        if not scan_task.done():
            scan_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scan_task
        await svc.writer.stop()  # flush traces still queued
        await svc.client.aclose()
        if svc.writer.dropped:
            log.warning("%d trace(s) could not be saved", svc.writer.dropped)
    return 0