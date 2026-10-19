import hashlib
import os
import threading

import orjson
import pytest

import argus.integrity.service as svc_mod
from argus.integrity.hasher import ScanCancelled
from argus.integrity.models import (BLOB_CORRUPT, BLOB_OK, BLOB_UNREADABLE, MANIFEST_APPROVED,
                                    MANIFEST_CHANGED, MANIFEST_MISSING, MANIFEST_NEW)
from argus.integrity.service import RECHECK_NS, IntegrityService, from_json, to_json
from argus.storage.db import Database


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Store:
    """A fake Ollama model store on disk."""

    def __init__(self, base):
        self.base = base
        (base / "blobs").mkdir(parents=True)

    def blob(self, data: bytes, *, claim: bytes | None = None) -> str:
        """Write a blob. Normally it is named after its own hash; pass `claim` to
        name it after DIFFERENT content, i.e. a blob whose bytes don't match its name."""
        digest = sha(claim if claim is not None else data)
        (self.base / "blobs" / f"sha256-{digest}").write_bytes(data)
        return digest

    def manifest(self, name: str, tag: str, digests: list[str], namespace: str = "library",
                extra: str = "") -> str:
        p = self.base / "manifests" / "registry.ollama.ai" / namespace / name / tag
        p.parent.mkdir(parents=True, exist_ok=True)
        body = {"schemaVersion": 2, "note": extra,
                "layers": [{"digest": f"sha256:{d}", "size": 1} for d in digests]}
        p.write_bytes(orjson.dumps(body))
        return str(p)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.init()
    return d


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "models")


@pytest.fixture
def service(db, store):
    return IntegrityService(db, models_dir_override=str(store.base))


def blob_by(result, digest):
    return next(b for b in result.blobs if b.digest == digest)


def manifest_by(result, name):
    return next(m for m in result.manifests if m.name == name)


# ---- scanning: blobs ---------------------------------------------------------

def test_no_models_directory_is_reported_calmly_not_as_an_error(db, tmp_path):
    s = IntegrityService(db, models_dir_override=str(tmp_path / "nowhere"))
    r = s.scan()
    assert r.models_dir is None and r.blobs == [] and r.warning_count == 0
    assert s.latest().models_dir is None            # and it is remembered


def test_good_blobs_verify_and_nothing_warns(store, service):
    d = store.blob(b"model weights")
    store.manifest("llama3", "8b", [d])
    r = service.scan()
    assert blob_by(r, d).status == BLOB_OK and blob_by(r, d).actual_digest == d
    assert r.warning_count == 0


def test_a_blob_whose_bytes_do_not_match_its_name_is_corrupt(store, service):
    d = store.blob(b"tampered bytes", claim=b"the original bytes")
    store.manifest("llama3", "8b", [d])
    r = service.scan()
    b = blob_by(r, d)
    assert b.status == BLOB_CORRUPT
    assert b.actual_digest == sha(b"tampered bytes") != b.digest
    assert b.models == ["llama3:8b"]                # the warning can name the model
    assert r.warning_count == 1


def test_a_shared_blob_lists_every_model_that_uses_it(store, service):
    d = store.blob(b"shared", claim=b"other")
    store.manifest("zeta", "1", [d])
    store.manifest("alpha", "2", [d])
    assert blob_by(service.scan(), d).models == ["alpha:2", "zeta:1"]


def test_an_unreferenced_blob_is_still_checked(store, service):
    d = store.blob(b"orphan", claim=b"different")
    b = blob_by(service.scan(), d)
    assert b.status == BLOB_CORRUPT and b.models == []


def test_partial_downloads_are_not_checked_at_all(store, service):
    d = sha(b"whole")
    (store.base / "blobs" / f"sha256-{d}-partial").write_bytes(b"half")
    assert service.scan().blobs == []


def test_an_unreadable_blob_is_its_own_state_never_corrupt(store, service, monkeypatch):
    d = store.blob(b"locked")
    monkeypatch.setattr(svc_mod, "sha256_file", lambda *a, **k: None)
    r = service.scan()
    assert blob_by(r, d).status == BLOB_UNREADABLE
    assert blob_by(r, d).actual_digest is None
    assert r.warning_count == 0 and [b.digest for b in r.unreadable_blobs] == [d]


# ---- scanning: manifests and approvals -----------------------------------------

def test_a_never_approved_model_is_new_and_does_not_raise_the_alarm(store, service):
    store.manifest("llama3", "8b", [store.blob(b"w")])
    r = service.scan()
    m = manifest_by(r, "llama3:8b")
    assert m.status == MANIFEST_NEW and m.approved_digest is None
    assert r.warning_count == 0 and [x.name for x in r.new_manifests] == ["llama3:8b"]


def test_approving_records_the_digest_and_survives_a_rescan(store, service):
    store.manifest("llama3", "8b", [store.blob(b"w")])
    m = manifest_by(service.scan(), "llama3:8b")
    assert service.approve("llama3:8b", m.current_digest) is True
    assert manifest_by(service.latest(), "llama3:8b").status == MANIFEST_APPROVED
    again = manifest_by(service.scan(), "llama3:8b")
    assert again.status == MANIFEST_APPROVED and again.approved_wall_ns is not None


def test_approval_survives_a_restart(db, store, service):
    store.manifest("llama3", "8b", [store.blob(b"w")])
    m = manifest_by(service.scan(), "llama3:8b")
    service.approve("llama3:8b", m.current_digest)
    fresh = IntegrityService(db, models_dir_override=str(store.base))     # a new process
    assert manifest_by(fresh.scan(), "llama3:8b").status == MANIFEST_APPROVED


def test_a_manifest_that_changes_after_approval_warns(store, service):
    d = store.blob(b"w")
    store.manifest("llama3", "8b", [d])
    service.approve("llama3:8b", manifest_by(service.scan(), "llama3:8b").current_digest)
    store.manifest("llama3", "8b", [d], extra="edited")
    r = service.scan()
    m = manifest_by(r, "llama3:8b")
    assert m.status == MANIFEST_CHANGED and m.current_digest != m.approved_digest
    assert r.warning_count == 1 and [x.name for x in r.changed_manifests] == ["llama3:8b"]


def test_re_approving_a_changed_manifest_clears_the_warning(store, service):
    d = store.blob(b"w")
    store.manifest("llama3", "8b", [d])
    service.approve("llama3:8b", manifest_by(service.scan(), "llama3:8b").current_digest)
    store.manifest("llama3", "8b", [d], extra="edited")
    changed = manifest_by(service.scan(), "llama3:8b")
    assert service.approve("llama3:8b", changed.current_digest) is True
    assert service.latest().warning_count == 0


def test_approval_is_refused_if_the_file_changed_after_the_person_looked(store, service):
    d = store.blob(b"w")
    store.manifest("llama3", "8b", [d])
    reviewed = manifest_by(service.scan(), "llama3:8b").current_digest
    store.manifest("llama3", "8b", [d], extra="swapped after review")
    assert service.approve("llama3:8b", reviewed) is False
    assert manifest_by(service.latest(), "llama3:8b").status == MANIFEST_NEW   # still unapproved


def test_approval_is_refused_for_a_digest_that_was_never_shown(store, service):
    store.manifest("llama3", "8b", [store.blob(b"w")])
    service.scan()
    assert service.approve("llama3:8b", "0" * 64) is False
    assert service.approve("no-such-model:1", "0" * 64) is False


def test_approve_before_any_scan_is_a_clean_refusal(service):
    assert service.approve("x:1", "0" * 64) is False
    assert service.approve_all_new() == 0
    assert service.forget("x:1") is False


def test_approve_all_new_leaves_changed_models_for_a_deliberate_look(store, service):
    d = store.blob(b"w")
    store.manifest("kept", "1", [d])
    service.approve("kept:1", manifest_by(service.scan(), "kept:1").current_digest)
    store.manifest("kept", "1", [d], extra="changed")          # now "changed"
    store.manifest("fresh", "1", [d])                          # now "new"
    service.scan()
    assert service.approve_all_new() == 1
    r = service.latest()
    assert manifest_by(r, "fresh:1").status == MANIFEST_APPROVED
    assert manifest_by(r, "kept:1").status == MANIFEST_CHANGED      # untouched
    assert r.warning_count == 1


def test_a_removed_model_is_missing_not_a_warning_and_can_be_forgotten(store, service):
    d = store.blob(b"w")
    path = store.manifest("llama3", "8b", [d])
    service.approve("llama3:8b", manifest_by(service.scan(), "llama3:8b").current_digest)
    os.remove(path)                                            # ollama rm
    r = service.scan()
    m = manifest_by(r, "llama3:8b")
    assert m.status == MANIFEST_MISSING and m.path is None
    assert r.warning_count == 0
    assert service.forget("llama3:8b") is True
    assert not any(x.name == "llama3:8b" for x in service.latest().manifests)
    assert not any(x.name == "llama3:8b" for x in service.scan().manifests)   # gone for good


def test_forget_only_applies_to_missing_models(store, service):
    store.manifest("llama3", "8b", [store.blob(b"w")])
    service.scan()
    assert service.forget("llama3:8b") is False


def test_namespaced_models_keep_their_namespace(store, service):
    store.manifest("tune", "latest", [store.blob(b"w")], namespace="someone")
    assert [m.name for m in service.scan().manifests] == ["someone/tune:latest"]


# ---- the verification cache ------------------------------------------------------

@pytest.fixture
def hash_calls(monkeypatch):
    real = svc_mod.sha256_file
    calls = []

    def counting(path, **kw):
        calls.append(path.name)
        return real(path, **kw)

    monkeypatch.setattr(svc_mod, "sha256_file", counting)
    return calls


def blob_calls(calls):
    return [c for c in calls if c.startswith("sha256-")]


def test_an_unchanged_blob_is_not_rehashed_on_the_next_scan(store, service, hash_calls):
    d = store.blob(b"big model")
    first = service.scan()
    assert blob_by(first, d).rehashed is True and len(blob_calls(hash_calls)) == 1
    second = service.scan()
    b = blob_by(second, d)
    assert len(blob_calls(hash_calls)) == 1                    # no second hash
    assert b.status == BLOB_OK and b.rehashed is False
    assert b.verified_wall_ns == blob_by(first, d).verified_wall_ns    # honest about when


def test_force_rehashes_everything(store, service, hash_calls):
    store.blob(b"big model")
    service.scan()
    r = service.scan(force=True)
    assert len(blob_calls(hash_calls)) == 2 and r.forced is True
    assert all(b.rehashed for b in r.blobs)


def test_a_changed_size_or_mtime_invalidates_the_cache(store, service, hash_calls):
    d = store.blob(b"original")
    service.scan()
    path = store.base / "blobs" / f"sha256-{d}"
    path.write_bytes(b"replaced with different length")
    r = service.scan()
    assert len(blob_calls(hash_calls)) == 2 and blob_by(r, d).status == BLOB_CORRUPT


def test_same_size_tampering_with_a_new_timestamp_is_caught_without_force(store, service):
    d = store.blob(b"AAAA")
    service.scan()
    path = store.base / "blobs" / f"sha256-{d}"
    path.write_bytes(b"BBBB")                                  # same size
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))   # different mtime
    assert blob_by(service.scan(), d).status == BLOB_CORRUPT


def test_known_limit_forged_timestamp_hides_tampering_until_a_full_rehash(store, service):
    """Documents the deliberate trade-off (see the module docstring): a blob with
    the same size AND a forged-back timestamp is trusted until the next full re-hash."""
    d = store.blob(b"AAAA")
    service.scan()
    path = store.base / "blobs" / f"sha256-{d}"
    original = path.stat()
    path.write_bytes(b"BBBB")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))       # forged back
    assert blob_by(service.scan(), d).status == BLOB_OK                   # not yet noticed
    assert blob_by(service.scan(force=True), d).status == BLOB_CORRUPT    # a full re-hash catches it


def test_the_cache_expires_after_the_recheck_window(store, service, db, hash_calls):
    d = store.blob(b"model")
    service.scan()
    size, mtime, verified = db.get_blob_checks()[d]
    db.replace_blob_checks([(d, size, mtime, verified - RECHECK_NS - 1)])   # verified too long ago
    service.scan()
    assert len(blob_calls(hash_calls)) == 2


def test_a_corrupt_blob_is_never_cached_as_good(store, service, hash_calls):
    store.blob(b"bad", claim=b"good")
    service.scan()
    r = service.scan()
    assert len(blob_calls(hash_calls)) == 2                    # re-hashed every time
    assert r.warning_count == 1


def test_checks_for_deleted_blobs_are_pruned(store, service, db):
    d = store.blob(b"gone soon")
    service.scan()
    assert d in db.get_blob_checks()
    os.remove(store.base / "blobs" / f"sha256-{d}")
    service.scan()
    assert db.get_blob_checks() == {}


# ---- concurrency and shutdown --------------------------------------------------------

def test_a_scan_refuses_to_run_while_another_is_in_progress(store, service):
    store.blob(b"w")
    service._lock.acquire()
    try:
        assert service.scanning is True
        assert service.scan() is None
        assert service.approve("x:1", "0" * 64) is False
        assert service.approve_all_new() == 0
        assert service.forget("x:1") is False
    finally:
        service._lock.release()
    assert service.scanning is False and service.scan() is not None


def test_cancellation_abandons_the_scan_keeps_the_last_result_and_frees_the_lock(
        store, service, monkeypatch):
    store.blob(b"w")
    good = service.scan()
    monkeypatch.setattr(svc_mod, "sha256_file",
                        lambda *a, **k: (_ for _ in ()).throw(ScanCancelled()))
    assert service.scan(force=True) is None
    assert service.scanning is False
    assert service.latest().scanned_wall_ns == good.scanned_wall_ns      # untouched


def test_cancel_sets_the_event_the_hasher_watches(service):
    assert isinstance(service._cancel, threading.Event)
    service.cancel()
    assert service._cancel.is_set()


def test_a_new_scan_clears_an_earlier_cancel_request(store, service):
    store.blob(b"w")
    service.cancel()
    assert service.scan() is not None


# ---- persistence format ------------------------------------------------------------------

def test_results_round_trip_through_json(store, service):
    d = store.blob(b"bad", claim=b"good")
    store.manifest("llama3", "8b", [d])
    r = service.scan()
    again = from_json(to_json(r))
    assert again == r


def test_garbage_in_the_cache_row_reads_as_not_scanned(service, db):
    db.save_integrity_scan(1, 5, "{not json")
    assert service.latest() is None
    db.save_integrity_scan(1, 5, '{"unexpected": "shape"}')
    assert service.latest() is None
    assert from_json("null") is None and from_json("") is None
