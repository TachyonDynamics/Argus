"""Runs integrity scans and the approve / forget workflow.

Two independent checks, matching how Ollama stores models:

1. Blob self-verification. Every blob is named after its own SHA-256, so hashing
   it and comparing to its filename needs no baseline: a mismatch means the
   file's bytes changed after it was written.
2. Manifest change detection. Each model's manifest is hashed and compared with
   what the user approved, catching a model swapped for a different (but
   individually valid) one.

Hashing multi-gigabyte files is slow, so a blob already fully hashed within the
last RECHECK_DAYS - and whose size and modification time haven't changed since -
is not hashed again unless a full re-hash is requested. That trades detection
speed for cost: same-size, same-timestamp tampering (which needs deliberate
effort to forge) can go unnoticed until the next full re-hash. Every result
records whether it was hashed this scan, so the UI can say which it was.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import orjson

from argus.integrity.hasher import ScanCancelled, sha256_file
from argus.integrity.layout import list_blobs, list_manifests, manifest_blob_digests, models_dir
from argus.integrity.models import (BLOB_CORRUPT, BLOB_OK, BLOB_UNREADABLE, MANIFEST_APPROVED,
                                    MANIFEST_CHANGED, MANIFEST_MISSING, MANIFEST_NEW,
                                    MANIFEST_UNREADABLE, BlobFinding, ManifestFinding,
                                    ScanResult)
from argus.storage.db import Database

log = logging.getLogger(__name__)

RECHECK_DAYS = 7
RECHECK_NS = RECHECK_DAYS * 24 * 3600 * 1_000_000_000


def to_json(result: ScanResult) -> str:
    return orjson.dumps(asdict(result), default=str).decode()


def from_json(text: str) -> ScanResult | None:
    """None if the stored text isn't a valid result - callers treat that as
    "not scanned yet" rather than failing."""
    try:
        d = orjson.loads(text)
        return ScanResult(
            scanned_wall_ns=d["scanned_wall_ns"],
            models_dir=Path(d["models_dir"]) if d["models_dir"] else None,
            forced=d.get("forced", False),
            blobs=[BlobFinding(
                digest=b["digest"], path=Path(b["path"]), size=b["size"], status=b["status"],
                actual_digest=b.get("actual_digest"), models=list(b.get("models", [])),
                verified_wall_ns=b.get("verified_wall_ns"), rehashed=b.get("rehashed", True),
            ) for b in d["blobs"]],
            manifests=[ManifestFinding(
                name=m["name"], path=Path(m["path"]) if m["path"] else None, status=m["status"],
                current_digest=m.get("current_digest"), approved_digest=m.get("approved_digest"),
                approved_wall_ns=m.get("approved_wall_ns"),
                blob_digests=list(m.get("blob_digests", [])),
            ) for m in d["manifests"]],
        )
    except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


class IntegrityService:
    def __init__(self, db: Database, models_dir_override: str | None = None) -> None:
        self._db = db
        self._override = models_dir_override
        self._lock = threading.Lock()       # one scan or edit at a time
        self._cancel = threading.Event()

    @property
    def scanning(self) -> bool:
        """True while a scan or an approval is in progress."""
        return self._lock.locked()

    def cancel(self) -> None:
        """Ask a running scan to stop at its next chunk. Called at shutdown so a
        background hash can never hold Argus open."""
        self._cancel.set()

    def latest(self) -> ScanResult | None:
        row = self._db.get_integrity_scan()
        return from_json(str(row["result_json"])) if row else None

    def _save(self, result: ScanResult) -> None:
        self._db.save_integrity_scan(result.scanned_wall_ns, result.warning_count, to_json(result))

    # ---- scanning ------------------------------------------------------------
    def scan(self, *, force: bool = False) -> ScanResult | None:
        """Scan every model file. Returns None if another scan (or an approval)
        is already running, or if this one was cancelled. Slow for large
        models: call it off the event loop."""
        if not self._lock.acquire(blocking=False):
            return None
        self._cancel.clear()
        try:
            return self._scan_locked(force)
        except ScanCancelled:
            log.info("integrity scan cancelled")
            return None
        finally:
            self._lock.release()

    def _scan_locked(self, force: bool) -> ScanResult:
        now = time.time_ns()
        base = models_dir(self._override)
        if base is None:
            result = ScanResult(scanned_wall_ns=now, models_dir=None, forced=force)
            self._save(result)
            self._db.replace_blob_checks([])
            return result

        manifests_on_disk = list_manifests(base)
        blob_digests_by_manifest = {m.name: manifest_blob_digests(m.path) for m in manifests_on_disk}
        models_by_blob: dict[str, list[str]] = {}
        for name, digests in blob_digests_by_manifest.items():
            for d in digests:
                models_by_blob.setdefault(d, []).append(name)

        cached = self._db.get_blob_checks()
        blobs: list[BlobFinding] = []
        checks: list[tuple[str, int, int, int]] = []
        for blob in list_blobs(base):
            models = sorted(models_by_blob.get(blob.digest, []))
            prev = cached.get(blob.digest)
            if (not force and prev is not None and prev[0] == blob.size
                    and prev[1] == blob.mtime_ns and now - prev[2] < RECHECK_NS):
                blobs.append(BlobFinding(blob.digest, blob.path, blob.size, BLOB_OK,
                                         actual_digest=blob.digest, models=models,
                                         verified_wall_ns=prev[2], rehashed=False))
                checks.append((blob.digest, blob.size, blob.mtime_ns, prev[2]))
                continue
            actual = sha256_file(blob.path, cancel=self._cancel)
            if actual is None:
                blobs.append(BlobFinding(blob.digest, blob.path, blob.size, BLOB_UNREADABLE,
                                         models=models))
            elif actual == blob.digest:
                blobs.append(BlobFinding(blob.digest, blob.path, blob.size, BLOB_OK,
                                         actual_digest=actual, models=models,
                                         verified_wall_ns=now))
                checks.append((blob.digest, blob.size, blob.mtime_ns, now))
            else:
                blobs.append(BlobFinding(blob.digest, blob.path, blob.size, BLOB_CORRUPT,
                                         actual_digest=actual, models=models))

        approvals = self._db.get_approvals()
        manifests: list[ManifestFinding] = []
        for m in manifests_on_disk:
            current = sha256_file(m.path)
            approved = approvals.get(m.name)
            if current is None:
                status = MANIFEST_UNREADABLE
            elif approved is None:
                status = MANIFEST_NEW
            elif approved[0] == current:
                status = MANIFEST_APPROVED
            else:
                status = MANIFEST_CHANGED
            manifests.append(ManifestFinding(
                name=m.name, path=m.path, status=status, current_digest=current,
                approved_digest=approved[0] if approved else None,
                approved_wall_ns=approved[1] if approved else None,
                blob_digests=blob_digests_by_manifest.get(m.name, [])))
        on_disk = {m.name for m in manifests_on_disk}
        for name, (digest, when) in approvals.items():
            if name not in on_disk:
                manifests.append(ManifestFinding(
                    name=name, path=None, status=MANIFEST_MISSING, current_digest=None,
                    approved_digest=digest, approved_wall_ns=when))

        manifests.sort(key=lambda f: f.name)
        blobs.sort(key=lambda b: (b.models or ["~"], b.digest))
        result = ScanResult(scanned_wall_ns=now, models_dir=base, blobs=blobs,
                            manifests=manifests, forced=force)
        self._save(result)
        self._db.replace_blob_checks(checks)
        return result

    # ---- approvals -----------------------------------------------------------
    def approve(self, name: str, digest: str) -> bool:
        """Approve exactly the manifest the person reviewed. Refused (False) if
        the file has changed since that scan - the digest they saw must still
        be the digest on disk - or if a scan is running."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            result = self.latest()
            if result is None:
                return False
            finding = next((m for m in result.manifests if m.name == name), None)
            if finding is None or not self._approve_one(finding, digest):
                return False
            self._save(result)
            return True
        finally:
            self._lock.release()

    def approve_all_new(self) -> int:
        """Approve every model that has never been approved (first-run
        convenience). Models whose manifest *changed* are deliberately left
        out: a change needs its own deliberate look. Returns how many were approved."""
        if not self._lock.acquire(blocking=False):
            return 0
        try:
            result = self.latest()
            if result is None:
                return 0
            count = sum(1 for f in result.new_manifests
                        if f.current_digest and self._approve_one(f, f.current_digest))
            if count:
                self._save(result)
            return count
        finally:
            self._lock.release()

    def _approve_one(self, f: ManifestFinding, digest: str) -> bool:
        if (f.path is None or f.current_digest != digest
                or f.status not in (MANIFEST_NEW, MANIFEST_CHANGED)):
            return False
        if sha256_file(f.path) != digest:      # changed since the scan the person looked at
            return False
        now = time.time_ns()
        self._db.approve_manifest(f.name, digest, now)
        f.status, f.approved_digest, f.approved_wall_ns = MANIFEST_APPROVED, digest, now
        return True

    def forget(self, name: str) -> bool:
        """Drop the approval for a model that no longer exists on disk (for
        example after `ollama rm`), so it stops being listed."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            result = self.latest()
            if result is None:
                return False
            finding = next((m for m in result.manifests
                            if m.name == name and m.status == MANIFEST_MISSING), None)
            if finding is None:
                return False
            self._db.delete_approval(name)
            result.manifests.remove(finding)
            self._save(result)
            return True
        finally:
            self._lock.release()