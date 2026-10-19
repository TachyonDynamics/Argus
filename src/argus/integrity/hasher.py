"""SHA-256 hashing, tolerant of files that vanish or can't be opened mid-read
(a file locked by another process, say). A read failure is never treated as
corruption - it's a distinct "couldn't verify" outcome, which is what
models.BLOB_UNREADABLE is for.

Hashing a large model can take a minute on a slow disk, so it can be cancelled
between chunks: Argus must never make Ctrl+C wait on a background hash."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

_CHUNK = 8 * 1024 * 1024


class ScanCancelled(Exception):
    """Raised when a cancellation was requested while hashing."""


def sha256_file(path: Path, *, chunk_size: int = _CHUNK,
                cancel: threading.Event | None = None) -> str | None:
    """The lowercase hex digest of a file's contents, or None if it couldn't
    be read at all (missing, permission denied, or vanished mid-read).

    Raises ScanCancelled if `cancel` is set between chunks."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
                if cancel is not None and cancel.is_set():
                    raise ScanCancelled
        return h.hexdigest()
    except OSError:
        return None