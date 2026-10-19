import httpx
import pytest_asyncio

from argus.config import Config
from argus.core.events import EventBus
from argus.core.models import Outcome, RequestInfo, Trace, now_ns
from argus.integrity.service import IntegrityService
from argus.proxy.tracker import Tracker
from argus.storage.db import Database
from argus.ui.app import create_admin_app
from argus.ui.auth import COOKIE

PORT = 47897
TOKEN = "n" * 43


@pytest_asyncio.fixture
async def env(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "n.db")
    db.init()
    tracker = Tracker(EventBus())
    app = create_admin_app(cfg, db, tracker, TOKEN, IntegrityService(db))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}") as client:
        client.cookies.set(COOKIE, TOKEN, domain="127.0.0.1")
        yield client, db, tracker


def start_live(tracker, *, model="llama3", kind="generate", started=True):
    t = Trace("POST", "/api/generate", client_addr="127.0.0.1:6000")
    t.provider = "ollama"
    t.request = RequestInfo(kind=kind, model=model, stream=True)
    if started:
        t.stamps.sent = t.stamps.recv
        t.stamps.hdr = t.stamps.recv
        t.stamps.first_byte = t.stamps.recv
        t.stamps.last_chunk = t.stamps.recv
    tracker.start(t)
    return t


def add_finished(db, *, model="llama3", outcome=Outcome.OK, error_code=None):
    t = Trace("POST", "/api/generate")
    t.provider, t.status_code = "ollama", 200
    t.request = RequestInfo(kind="generate", model=model, stream=True)
    t.stamps.sent = t.stamps.recv
    t.stamps.first_byte = t.stamps.recv
    t.stamps.end = t.stamps.recv + 1_000_000_000
    t.finish(outcome, error_code)
    db.insert_requests([t.to_row()])


async def test_now_requires_auth(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "n2.db")
    db.init()
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, IntegrityService(db))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}") as client:
        assert (await client.get("/now")).status_code == 401


async def test_idle_state_when_nothing_is_running_or_finished(env):
    client, _, _ = env
    body = (await client.get("/now")).text
    assert "Nothing running" in body
    assert "isn't proxying any requests" in body
    assert "Nothing has finished yet" in body


async def test_a_live_generating_request_shows_up_as_a_card(env):
    client, _, tracker = env
    start_live(tracker, model="llama3")
    body = (await client.get("/now")).text
    assert "1 request running" in body
    assert "llama3" in body and "Generating" in body
    assert "127.0.0.1:6000" in body


async def test_two_live_requests_pluralise_the_headline(env):
    client, _, tracker = env
    start_live(tracker, model="a")
    start_live(tracker, model="b")
    body = (await client.get("/now")).text
    assert "2 requests running" in body


async def test_a_request_still_connecting_shows_that_phase_not_generating(env):
    client, _, tracker = env
    start_live(tracker, started=False)
    body = (await client.get("/now")).text
    assert "Connecting" in body and "Generating" not in body


async def test_finishing_a_request_removes_it_from_live_and_it_can_appear_as_recent(env):
    client, db, tracker = env
    t = start_live(tracker, model="llama3")
    assert "1 request running" in (await client.get("/now")).text
    t.stamps.end = t.stamps.recv + 1_000_000_000
    t.finish(Outcome.OK)
    tracker.finish(t)
    db.insert_requests([t.to_row()])
    body = (await client.get("/now")).text
    assert "Nothing running" in body
    assert "llama3" in body and "Completed normally" in body


async def test_recent_list_links_to_the_detail_page_and_shows_a_real_why(env):
    client, db, _ = env
    add_finished(db, outcome=Outcome.UPSTREAM_UNREACHABLE, error_code="connection_refused")
    body = (await client.get("/now")).text
    assert "Couldn&#39;t reach Ollama" in body
    import re
    m = re.search(r'/requests/([0-9a-f]{32})', body)
    assert m
    detail = await client.get(f"/requests/{m.group(1)}")
    assert detail.status_code == 200


async def test_untrusted_model_name_is_escaped_on_the_now_page(env):
    client, db, tracker = env
    start_live(tracker, model="<script>x</script>")
    add_finished(db, model="<b>y</b>")
    body = (await client.get("/now")).text
    assert "<script>x</script>" not in body and "&lt;script&gt;x&lt;/script&gt;" in body
    assert "<b>y</b>" not in body and "&lt;b&gt;y&lt;/b&gt;" in body


async def test_nav_marks_now_and_requests_as_current_on_their_own_pages(env):
    client, _, _ = env
    now_body = (await client.get("/now")).text
    req_body = (await client.get("/")).text
    assert 'href="/now" aria-current="page"' in now_body
    assert 'href="/now" ' in req_body and 'aria-current' not in req_body.split('href="/now"')[1].split(">")[0]
    assert 'href="/" aria-current="page"' in req_body
