import re

import httpx
import pytest
import pytest_asyncio

from argus.config import Config
from argus.core.events import EventBus
from argus.core.models import Outcome, RequestInfo, Trace
from argus.integrity.service import IntegrityService
from argus.proxy.tracker import Tracker
from argus.storage.db import Database
from argus.ui.app import create_admin_app, parse_cursor
from argus.ui.auth import COOKIE, load_or_create_token
from argus.ui.present import fmt_duration, present, tokens_per_second

PORT = 47899
TOKEN = "t" * 43


@pytest_asyncio.fixture
async def env(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "ui.db")
    db.init()
    integrity = IntegrityService(db)
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, integrity)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}") as client:
        yield client, db


@pytest_asyncio.fixture
async def authed(env):
    client, db = env
    client.cookies.set(COOKIE, TOKEN, domain="127.0.0.1")
    return client, db


def add(db, *, model="llama3", outcome=Outcome.OK, code=None, status=200, eval_count=None):
    t = Trace("POST", "/api/generate")
    t.provider, t.status_code = "ollama", status
    t.request = RequestInfo(kind="generate", model=model, stream=True)
    t.finish(outcome, code)
    row = t.to_row()
    if eval_count:
        row.update(o_eval_count=eval_count, o_eval_ns=1_000_000_000, o_prompt_count=7)
    db.insert_requests([row])


async def test_requires_token(env):
    client, _ = env
    assert (await client.get("/")).status_code == 401


async def test_token_in_url_sets_cookie_and_redirects_without_it(env):
    client, _ = env
    r = await client.get(f"/?t={TOKEN}&problems=1")
    assert r.status_code == 303 and r.headers["location"] == "/?problems=1"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    client.cookies.clear()
    assert (await client.get("/?t=wrong")).status_code == 401


async def test_redirect_never_leaves_the_site(tmp_path):
    # httpx normalises "//host/path", so drive the ASGI app with a raw scope instead.
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "ui.db")
    db.init()
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, IntegrityService(db))
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
               "scheme": "http", "path": "//evil.example/x", "raw_path": b"//evil.example/x",
               "root_path": "", "query_string": f"t={TOKEN}".encode(),
               "headers": [(b"host", f"127.0.0.1:{PORT}".encode())],
               "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", PORT)}, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    location = dict(start["headers"])[b"location"]
    assert start["status"] == 303 and location == b"/"


async def test_rejects_foreign_host_header(authed):
    client, _ = authed
    assert (await client.get("/", headers={"host": "evil.example:47899"})).status_code == 400
    assert (await client.get("/", headers={"host": f"127.0.0.1:{PORT + 1}"})).status_code == 400


async def test_state_changing_requests_need_a_matching_origin(authed):
    client, _ = authed
    assert (await client.post("/", headers={"origin": "http://evil.example"})).status_code == 403
    assert (await client.post("/")).status_code == 403


async def test_pages_carry_security_headers(authed):
    client, _ = authed
    r = await client.get("/")
    assert r.status_code == 200
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in r.headers


async def test_static_needs_no_token_but_still_checks_host(env):
    client, _ = env
    assert (await client.get("/static/app.css")).status_code == 200
    assert (await client.get("/static/app.css", headers={"host": "evil.example"})).status_code == 400


async def test_empty_state_tells_the_user_what_to_do(authed):
    client, _ = authed
    body = (await client.get("/")).text
    assert "No requests yet" in body and "11434" in body


async def test_table_shows_rows_and_escapes_untrusted_model_names(authed):
    client, db = authed
    add(db, model="<script>alert(1)</script>", eval_count=42)
    add(db, model="qwen", outcome=Outcome.UPSTREAM_UNREACHABLE, code="connection_refused", status=502)
    body = (await client.get("/")).text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "42 tok/s" in body and "7 in, 42 out" in body
    assert "Couldn&#39;t reach Ollama" in body
    assert "connection refused" in body
    problems = (await client.get("/?problems=1")).text
    assert "qwen" in problems and "&lt;script&gt;" not in problems


async def test_paging_link_and_cursor_validation(authed):
    client, db = authed
    for i in range(55):
        add(db, model=f"m{i}")
    page1 = (await client.get("/")).text
    assert "Older requests" in page1 and 'http-equiv="refresh"' in page1
    cursor = re.search(r"before=([0-9a-f_]+)", page1).group(1)
    page2 = (await client.get(f"/?before={cursor}")).text
    assert "Back to newest" in page2 and 'http-equiv="refresh"' not in page2
    assert parse_cursor("nonsense") is None and parse_cursor("1_" + "a" * 32) == (1, "a" * 32)
    assert (await client.get("/?before=' OR 1=1 --")).status_code == 200   # ignored, not executed


def test_present_formats():
    assert fmt_duration(None) == "–" and fmt_duration(400_000) == "<1 ms"
    assert fmt_duration(412_000_000) == "412 ms" and fmt_duration(9_100_000_000) == "9.1 s"
    assert fmt_duration(125_000_000_000) == "2 min 5 s"
    assert tokens_per_second({"o_eval_count": 42, "o_eval_ns": 1_000_000_000}) == 42
    assert tokens_per_second({"o_eval_count": 0, "o_eval_ns": 5}) is None
    row = {"id": "x", "started_wall_ns": 1, "outcome": "http_error", "status_code": 404,
           "error_code": "model_not_found", "model": "m", "kind": "chat"}
    p = present(row)
    assert p["tone"] == "bad" and p["detail"] == "model not found (HTTP 404)"


def test_token_is_created_once_and_reused(tmp_path):
    a = load_or_create_token(tmp_path)
    assert len(a) >= 32 and load_or_create_token(tmp_path) == a