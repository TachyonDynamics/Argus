import httpx
import pytest
import pytest_asyncio

from argus.config import Config
from argus.core.events import EventBus
from argus.core.models import FinalStats, Outcome, RequestInfo, Trace
from argus.integrity.service import IntegrityService
from argus.proxy.tracker import Tracker
from argus.storage.db import Database
from argus.ui.app import create_admin_app
from argus.ui.auth import COOKIE

PORT = 47898
TOKEN = "d" * 43
NS = 1_000_000_000


@pytest_asyncio.fixture
async def env(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path, upstream_url="http://127.0.0.1:11435")
    db = Database(tmp_path / "d.db")
    db.init()
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, IntegrityService(db))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}") as client:
        client.cookies.set(COOKIE, TOKEN, domain="127.0.0.1")
        yield client, db


def add_row(db, *, model="llama3", outcome=Outcome.OK, error_code=None, sent=0,
           first=0, end=1 * NS, load=0, prompt=0, prompt_count=None,
           eval_ns=None, eval_count=None, inflight_at_start=0, inflight_same_model=0,
           client_wait=0, max_gap=0, max_gap_at=0, conn=0):
    t = Trace("POST", "/api/generate")
    t.provider, t.status_code = "ollama", 200 if outcome == Outcome.OK else 500
    t.request = RequestInfo(kind="generate", model=model, stream=True)
    t.inflight_at_start, t.inflight_same_model = inflight_at_start, inflight_same_model
    t.client_wait_ns, t.max_gap_ns, t.max_gap_at_ns = client_wait, max_gap, max_gap_at
    t.stamps.sent = t.stamps.recv + sent
    t.stamps.first_out = t.stamps.recv + first
    t.stamps.first_byte = t.stamps.recv + first
    t.stamps.end = t.stamps.recv + end
    t.stamps.conn0, t.stamps.conn1 = t.stamps.recv, t.stamps.recv + conn
    if eval_ns is not None or load or prompt:
        t.final = FinalStats(load_ns=load, prompt_eval_ns=prompt, prompt_eval_count=prompt_count,
                             eval_ns=eval_ns, eval_count=eval_count)
    t.finish(outcome, error_code)
    row = t.to_row()
    db.insert_requests([row])
    return row["id"]


async def test_unknown_id_is_a_clean_404(env):
    client, _ = env
    r = await client.get("/requests/does-not-exist")
    assert r.status_code == 404 and "No request found" in r.text


async def test_cold_load_finding_renders_with_evidence_and_fix(env):
    client, db = env
    rid = add_row(db, sent=0, first=10 * NS, end=11 * NS, load=9 * NS, prompt=int(0.9 * NS),
                 prompt_count=50, eval_ns=int(0.1 * NS), eval_count=5)
    body = (await client.get(f"/requests/{rid}")).text
    assert "had to be loaded first" in body
    assert "Confirmed cause" in body
    assert "keep_alive" in body
    assert "measured" in body            # evidence chip label
    assert "seg-load" in body            # waterfall segment present


async def test_queueing_finding_cites_the_other_request(env):
    client, db = env
    rid = add_row(db, sent=0, first=6 * NS, end=7 * NS, load=0, prompt=int(0.1 * NS),
                 prompt_count=5, eval_ns=int(0.9 * NS), eval_count=30,
                 inflight_at_start=1, inflight_same_model=1)
    body = (await client.get(f"/requests/{rid}")).text
    assert "Waited" in body and "before Ollama started" in body
    assert "Likely cause" in body


async def test_error_row_shows_the_specific_guidance_not_a_timing_rule(env):
    client, db = env
    rid = add_row(db, outcome=Outcome.STREAM_ERROR, error_code="out_of_memory", end=2 * NS)
    body = (await client.get(f"/requests/{rid}")).text
    assert "Ran out of memory" in body
    assert "smaller model" in body
    assert "seg-load" not in body        # timing rules are suppressed, not shown as "also"


async def test_client_cancelled_never_renders_with_critical_styling(env):
    client, db = env
    rid = add_row(db, outcome=Outcome.CLIENT_CANCELLED, end=3 * NS)
    body = (await client.get(f"/requests/{rid}")).text
    assert "Stopped before finishing" in body
    assert 'sev-critical' not in body


async def test_unreachable_error_names_the_actual_upstream_url(env):
    client, db = env
    rid = add_row(db, outcome=Outcome.UPSTREAM_UNREACHABLE, error_code="connection_refused")
    body = (await client.get(f"/requests/{rid}")).text
    assert "127.0.0.1:11435" in body


async def test_unremarkable_request_shows_waterfall_but_no_why_card(env):
    client, db = env
    rid = add_row(db, sent=0, first=int(0.05 * NS), end=int(0.3 * NS), load=0,
                 prompt=int(0.05 * NS), prompt_count=5, eval_ns=int(0.25 * NS), eval_count=10)
    body = (await client.get(f"/requests/{rid}")).text
    assert "Where the time went" in body
    assert "sev-warn" not in body and "sev-critical" not in body


async def test_baseline_makes_slow_generation_visible_after_enough_history(env):
    client, db = env
    for _ in range(10):
        add_row(db, sent=0, first=0, end=1 * NS, eval_ns=1 * NS, eval_count=40)  # 40 tok/s normal
    rid = add_row(db, sent=0, first=0, end=5 * NS, eval_ns=5 * NS, eval_count=10)  # 2 tok/s
    body = (await client.get(f"/requests/{rid}")).text
    assert "Slower than usual" in body and "40" in body


async def test_untrusted_model_name_is_escaped_on_the_detail_page(env):
    client, db = env
    rid = add_row(db, model="<img src=x onerror=alert(1)>", end=2 * NS)
    body = (await client.get(f"/requests/{rid}")).text
    assert "<img src=x" not in body
    assert "&lt;img src=x" in body


async def test_requests_list_links_each_row_to_its_detail_page(env):
    client, db = env
    rid = add_row(db, end=2 * NS)
    body = (await client.get("/")).text
    assert f'/requests/{rid}' in body


async def test_detail_page_requires_auth_like_everything_else(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "d2.db")
    db.init()
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, IntegrityService(db))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}") as client:
        assert (await client.get("/requests/anything")).status_code == 401
