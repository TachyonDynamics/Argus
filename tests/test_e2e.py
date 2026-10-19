"""Runs the real `python -m argus` entry point as a subprocess against an
in-process FakeOllama, and shuts it down the way a person actually would:
by sending it a signal. Everything else in the suite drives Argus's pieces
in-process; this is the one test that proves they still work wired together
as the shipped executable, including the Ctrl+C shutdown path.

Signal delivery to a subprocess is inherently platform-specific. This exercises
the POSIX SIGINT path; SIGBREAK (Windows Ctrl+Break, see app._install_stop) has
no equivalent here and needs a manual check on Windows.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

from conftest import free_port
from fake_ollama import HANG, Script, final, token

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="POSIX signal delivery only; see module docstring")

STARTUP_TIMEOUT = 10.0
SHUTDOWN_TIMEOUT = 10.0


class Argus:
    """A running `python -m argus` subprocess, plus the ports it's listening on."""

    def __init__(self, proc: asyncio.subprocess.Process, proxy_port: int, admin_port: int,
                data_dir: Path) -> None:
        self.proc = proc
        self.proxy_url = f"http://127.0.0.1:{proxy_port}"
        self.admin_url = f"http://127.0.0.1:{admin_port}"
        self.data_dir = data_dir

    @property
    def token(self) -> str:
        return (self.data_dir / "admin.token").read_text().strip()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "argus.db"


async def _wait_ready(url: str, timeout: float) -> None:
    async with httpx.AsyncClient() as c:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                await c.get(url, timeout=1.0)
                return
            except httpx.HTTPError:
                if asyncio.get_event_loop().time() > deadline:
                    raise
                await asyncio.sleep(0.05)


@pytest.fixture
async def running_argus(fake, tmp_path):
    proxy_port, admin_port = free_port(), free_port()
    kwargs = {}
    if sys.platform == "win32":  # lets SIGINT be delivered to this child specifically
        kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "argus",
        "--upstream", f"http://127.0.0.1:{fake.port}",
        "--proxy-port", str(proxy_port), "--admin-port", str(admin_port),
        "--data-dir", str(tmp_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **kwargs,
    )
    argus = Argus(proc, proxy_port, admin_port, tmp_path)
    try:
        await _wait_ready(f"{argus.proxy_url}/api/tags", STARTUP_TIMEOUT)
        yield argus
    finally:
        if proc.returncode is None:
            proc.send_signal(signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
            await proc.wait()


async def test_real_process_forwards_and_persists_a_request(running_argus, fake):
    fake.script = Script([token("hi"), final(eval_count=4)])
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{running_argus.proxy_url}/api/generate",
                         json={"model": "llama3", "prompt": "hi"})
    assert r.status_code == 200 and r.content == fake.sent

    async with httpx.AsyncClient() as c:                      # readiness only checked proxy; give
        for _ in range(40):                                   # the writer's flush interval a moment
            page = await c.get(f"{running_argus.admin_url}/?t={running_argus.token}",
                               follow_redirects=True)
            if "llama3" in page.text:
                break
            await asyncio.sleep(0.05)
    assert "llama3" in page.text and "4 out" in page.text


async def test_admin_ui_of_the_real_process_still_requires_the_token(running_argus):
    async with httpx.AsyncClient() as c:
        r = await c.get(running_argus.admin_url)
    assert r.status_code == 401


async def test_sigint_shuts_down_cleanly_and_flushes_pending_writes(running_argus, fake):
    fake.script = Script([token("a"), final()])
    async with httpx.AsyncClient() as c:
        await c.post(f"{running_argus.proxy_url}/api/generate", json={"model": "m", "prompt": "x"})

    running_argus.proc.send_signal(signal.SIGINT)
    out, _ = await asyncio.wait_for(running_argus.proc.communicate(), SHUTDOWN_TIMEOUT)
    assert running_argus.proc.returncode == 0, out.decode(errors="replace")

    conn = sqlite3.connect(running_argus.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= 1
    finally:
        conn.close()


async def test_second_sigint_forces_an_immediate_exit(running_argus, fake):
    """One request is left hanging (hung generation, client still attached) so the
    first SIGINT's graceful shutdown can't complete on its own. uvicorn's own
    force_exit doesn't help here: it only skips its internal bookkeeping waits, and
    still blocks on the stuck connection's socket for the full 5s graceful-shutdown
    timeout (see the comment in app._install_stop). So the second signal must exit
    the process directly, well before that 5s window would elapse."""
    fake.script = Script([token("a"), HANG])
    client = httpx.AsyncClient()
    holder = asyncio.create_task(client.post(
        f"{running_argus.proxy_url}/api/generate", json={"model": "m", "prompt": "x"}, timeout=60))
    await asyncio.sleep(0.3)  # let the request actually reach the hanging stream

    try:
        running_argus.proc.send_signal(signal.SIGINT)
        await asyncio.sleep(0.5)  # graceful shutdown is now blocked on our open connection
        assert running_argus.proc.returncode is None

        t0 = asyncio.get_event_loop().time()
        running_argus.proc.send_signal(signal.SIGINT)
        out, _ = await asyncio.wait_for(running_argus.proc.communicate(), timeout=3.0)
        elapsed = asyncio.get_event_loop().time() - t0
        assert elapsed < 2.0, f"took {elapsed:.1f}s: the second signal did not force an immediate exit"
        assert running_argus.proc.returncode == 1, out.decode(errors="replace")  # os._exit(1): forced
    finally:
        holder.cancel()
        await client.aclose()