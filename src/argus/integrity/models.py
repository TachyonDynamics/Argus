"""Data shapes for the integrity scan. Kept separate from the SQLite storage
shape so the scan logic and its tests don't need a database at all."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Blob status: "ok" (hash matches its filename), "corrupt" (hash mismatch - the
# file's bytes changed since it was written), "unreadable" (couldn't open it,
# e.g. locked; never a pass, and never called corrupt either).
BLOB_OK = "ok"
BLOB_CORRUPT = "corrupt"
BLOB_UNREADABLE = "unreadable"

# Manifest status against its approval baseline.
MANIFEST_NEW = "new"                # never approved
MANIFEST_APPROVED = "approved"      # matches the approved digest
MANIFEST_CHANGED = "changed"        # approved once, now differs
MANIFEST_MISSING = "missing"        # was approved; the file is gone (e.g. ollama rm)
MANIFEST_UNREADABLE = "unreadable"  # present but couldn't be read


@dataclass(slots=True)
class BlobFinding:
    digest: str            # from the filename: what the file claims to be
    path: Path
    size: int
    status: str
    actual_digest: str | None = None       # what hashing found; None if unreadable
    models: list[str] = field(default_factory=list)   # which models use this file
    verified_wall_ns: int | None = None    # when its bytes were last fully hashed OK
    rehashed: bool = True                  # False: satisfied from a recent earlier check


@dataclass(slots=True)
class ManifestFinding:
    name: str               # "llama3:8b" style display name
    path: Path | None       # None only for "missing"
    status: str
    current_digest: str | None      # None for "missing" / "unreadable"
    approved_digest: str | None
    approved_wall_ns: int | None
    blob_digests: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScanResult:
    scanned_wall_ns: int
    models_dir: Path | None         # None when no models directory was found
    blobs: list[BlobFinding] = field(default_factory=list)
    manifests: list[ManifestFinding] = field(default_factory=list)
    forced: bool = False            # True when every file was re-hashed regardless of cache

    @property
    def corrupt_blobs(self) -> list[BlobFinding]:
        return [b for b in self.blobs if b.status == BLOB_CORRUPT]

    @property
    def unreadable_blobs(self) -> list[BlobFinding]:
        return [b for b in self.blobs if b.status == BLOB_UNREADABLE]

    def _manifests(self, status: str) -> list[ManifestFinding]:
        return [m for m in self.manifests if m.status == status]

    @property
    def changed_manifests(self) -> list[ManifestFinding]:
        return self._manifests(MANIFEST_CHANGED)

    @property
    def new_manifests(self) -> list[ManifestFinding]:
        return self._manifests(MANIFEST_NEW)

    @property
    def missing_manifests(self) -> list[ManifestFinding]:
        return self._manifests(MANIFEST_MISSING)

    @property
    def warning_count(self) -> int:
        """What the warning banner counts: definite integrity failures only. A
        new model awaiting review, or a model removed with `ollama rm`, is
        ordinary activity and doesn't raise an alarm."""
        return len(self.corrupt_blobs) + len(self.changed_manifests)