import socket

import httpx
import pytest

from argus.app import build, port_problem, run
from argus.config import Config
from conftest import free_port
from fake_ollama import FakeOllama


def test_port_problem_none_when_free():
    assert port_problem("127.0.0.1", free_port()) is None


def test_port_problem_reports_a_bound_port():
    port = free_port()
    with socket.create_server(("127.0.0.1", port)):
        msg = port_problem("127.0.0.1", port)
        assert msg is not None and str(port) in msg and "127.0.0.1" in msg


async def test_build_wires_real_services_and_creates_the_token_file(tmp_path, fake: FakeOllama):
    cfg = Config(upstream_url=f"http://127.0.0.1:{fake.port}",
                proxy_port=free_port(), admin_port=free_port(), data_dir=tmp_path)
    svc = build(cfg)
    try:
        assert (tmp_path / "admin.token").exists()
        assert len(svc.token) >= 32
        assert (tmp_path / "argus.db").exists()
        assert svc.forwarder.cfg is cfg
        assert svc.tracker.snapshot() == []
        assert svc.integrity is not None
        # the proxy app really forwards, using the same forwarder build() wired up
        transport = httpx.ASGITransport(app=svc.proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            r = await c.get("/api/tags")
            assert r.status_code == 200
    finally:
        await svc.client.aclose()


async def test_run_reports_a_busy_proxy_port_without_raising(capsys, tmp_path):
    busy = free_port()
    with socket.create_server(("127.0.0.1", busy)):
        cfg = Config(proxy_port=busy, admin_port=free_port(), data_dir=tmp_path)
        rc = await run(cfg)
    assert rc == 2
    out = capsys.readouterr().out
    assert "proxy can't start" in out.lower() and str(busy) in out
    assert "OLLAMA_HOST" in out    # the proxy-port hint, not the generic message


async def test_run_reports_a_busy_admin_port_without_the_ollama_hint(capsys, tmp_path):
    busy = free_port()
    with socket.create_server(("127.0.0.1", busy)):
        cfg = Config(proxy_port=free_port(), admin_port=busy, data_dir=tmp_path)
        rc = await run(cfg)
    assert rc == 2
    out = capsys.readouterr().out
    assert "web ui can't start" in out.lower() and str(busy) in out
    assert "OLLAMA_HOST" not in out