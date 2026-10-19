import hashlib
import re

import httpx
import orjson
import pytest_asyncio

from argus.config import Config
from argus.core.events import EventBus
from argus.integrity.service import IntegrityService
from argus.proxy.tracker import Tracker
from argus.storage.db import Database
from argus.ui.app import create_admin_app
from argus.ui.auth import COOKIE

PORT = 47896
TOKEN = "i" * 43


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Store:
    def __init__(self, base):
        self.base = base
        (base / "blobs").mkdir(parents=True)

    def blob(self, data: bytes, *, claim: bytes | None = None) -> str:
        digest = sha(claim if claim is not None else data)
        (self.base / "blobs" / f"sha256-{digest}").write_bytes(data)
        return digest

    def manifest(self, name: str, tag: str, digests: list[str]) -> None:
        p = self.base / "manifests" / "registry.ollama.ai" / "library" / name / tag
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps({"layers": [{"digest": f"sha256:{d}"} for d in digests]}))


@pytest_asyncio.fixture
async def env(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "i.db")
    db.init()
    store = Store(tmp_path / "models")
    integrity = IntegrityService(db, models_dir_override=str(store.base))
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, integrity)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}",
                                 headers={"Origin": f"http://127.0.0.1:{PORT}"}) as client:
        client.cookies.set(COOKIE, TOKEN, domain="127.0.0.1")
        yield client, store, integrity


async def test_integrity_page_requires_auth(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "i2.db")
    db.init()
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, IntegrityService(db))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}") as client:
        assert (await client.get("/integrity")).status_code == 401
        # POST with no Origin header fails the cross-origin check before the token
        # check ever runs (see ui/auth.py) - GuardMiddleware's own documented order.
        assert (await client.post("/integrity/scan")).status_code == 403
        r = await client.post("/integrity/scan", headers={"origin": f"http://127.0.0.1:{PORT}"})
        assert r.status_code == 401


async def test_before_any_scan_the_page_says_so(env):
    client, _, _ = env
    body = (await client.get("/integrity")).text
    assert "haven&#39;t been checked" in body


async def test_scan_button_populates_the_page(env):
    client, store, _ = env
    store.blob(b"weights")
    r = await client.post("/integrity/scan")
    assert r.status_code == 303 and r.headers["location"] == "/integrity?msg=scanned"
    body = (await client.get(r.headers["location"])).text
    assert "checked" in body.lower()


async def test_no_models_directory_found(tmp_path):
    cfg = Config(admin_port=PORT, data_dir=tmp_path)
    db = Database(tmp_path / "i3.db")
    db.init()
    integrity = IntegrityService(db, models_dir_override=str(tmp_path / "nope"))
    app = create_admin_app(cfg, db, Tracker(EventBus()), TOKEN, integrity)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url=f"http://127.0.0.1:{PORT}",
                                 headers={"Origin": f"http://127.0.0.1:{PORT}"}) as client:
        client.cookies.set(COOKIE, TOKEN, domain="127.0.0.1")
        await client.post("/integrity/scan")
        body = (await client.get("/integrity")).text
        assert "No Ollama model folder found" in body


async def test_corrupt_blob_shows_a_warning_and_raises_the_nav_badge(env):
    client, store, _ = env
    store.blob(b"tampered", claim=b"original")
    store.manifest("llama3", "8b", [sha(b"original")])
    await client.post("/integrity/scan")
    body = (await client.get("/integrity")).text
    assert "doesn&#39;t match its fingerprint" in body
    assert "llama3:8b" in body
    other_page = (await client.get("/now")).text
    assert re.search(r'href="/integrity"[^>]*>Integrity<span class="nav-badge">1</span>', other_page)


async def test_untrusted_model_name_is_escaped_and_shell_command_is_withheld(env):
    client, store, _ = env
    store.blob(b"bad", claim=b"good")
    store.manifest("<script>alert(1)</script>", "1", [sha(b"good")])
    await client.post("/integrity/scan")
    body = (await client.get("/integrity")).text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


async def test_approve_then_rescan_shows_approved(env):
    client, store, _ = env
    d = store.blob(b"weights")
    store.manifest("llama3", "8b", [d])
    await client.post("/integrity/scan")
    page = (await client.get("/integrity")).text
    m = re.search(r'name="digest" value="([0-9a-f]+)"', page)
    r = await client.post("/integrity/approve", data={"name": "llama3:8b", "digest": m.group(1)})
    assert r.headers["location"] == "/integrity?msg=approved"
    await client.post("/integrity/scan")
    body = (await client.get("/integrity")).text
    assert "Approved" in body and "Not approved yet" not in body


async def test_approve_all_new(env):
    client, store, _ = env
    store.manifest("a", "1", [store.blob(b"a")])
    store.manifest("b", "1", [store.blob(b"b")])
    await client.post("/integrity/scan")
    r = await client.post("/integrity/approve-all")
    assert r.headers["location"] == "/integrity?msg=approved_all&count=2"
    body = (await client.get(r.headers["location"])).text
    assert "Approved 2 models" in body


async def test_forget_a_missing_model(env):
    client, store, _ = env
    import os
    d = store.blob(b"w")
    store.manifest("gone", "1", [d])
    await client.post("/integrity/scan")
    page = (await client.get("/integrity")).text
    m = re.search(r'name="digest" value="([0-9a-f]+)"', page)
    await client.post("/integrity/approve", data={"name": "gone:1", "digest": m.group(1)})
    os.remove(store.base / "manifests" / "registry.ollama.ai" / "library" / "gone" / "1")
    await client.post("/integrity/scan")
    assert "gone:1" in (await client.get("/integrity")).text
    r = await client.post("/integrity/forget", data={"name": "gone:1"})
    assert r.headers["location"] == "/integrity?msg=forgotten"
    assert "gone:1" not in (await client.get(r.headers["location"])).text


async def test_force_rehash_endpoint_is_wired(env):
    client, store, integrity = env
    store.blob(b"w")
    r = await client.post("/integrity/scan/force")
    assert r.status_code == 303
    assert integrity.latest().forced is True


async def test_actions_are_refused_without_a_matching_origin(env):
    client, _, _ = env
    r = await client.post("/integrity/scan", headers={"origin": "http://evil.example"})
    assert r.status_code == 403


async def test_clean_scan_shows_no_nav_badge(env):
    client, store, _ = env
    store.blob(b"fine")
    await client.post("/integrity/scan")
    body = (await client.get("/now")).text
    assert "nav-badge" not in body
