"""Finds Ollama's model store and lists the files that matter for integrity
checking, without needing a running Ollama process.

On-disk layout (unchanged across recent Ollama versions):
  <models>/blobs/sha256-<64 hex chars>                       one file per layer
  <models>/blobs/sha256-<...>-partial                        a download in progress
  <models>/manifests/<registry>/<namespace>/<name>/<tag>     JSON, no extension

The registry is almost always "registry.ollama.ai" and the namespace "library"
for official models; a manifest's path segments are what give a model its name,
since manifest files carry no explicit name field of their own.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import orjson

_DIGEST_RE = re.compile(r"^sha256-([0-9a-f]{64})$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 1024 * 1024   # real manifests are a few KB; refuse anything absurd


def models_dir(override: str | None = None) -> Path | None:
    """The models directory, or None if it doesn't exist.

    The first source that is set wins outright: an explicit override (tests, or
    a future setting), then OLLAMA_MODELS, then Ollama's own default under the
    user's home directory. A set-but-missing path is NOT skipped in favour of
    the next source - Ollama itself would use (or create) that path, so quietly
    scanning a different folder would report on the wrong models.

    Only the environment and filesystem are read, so this works when Ollama
    isn't running.
    """
    for candidate in (override, os.environ.get("OLLAMA_MODELS")):
        if candidate:
            p = Path(candidate)
            return p if p.is_dir() else None
    default = Path.home() / ".ollama" / "models"
    return default if default.is_dir() else None


@dataclass(slots=True)
class Blob:
    path: Path
    digest: str      # lowercase hex, from the filename - not yet verified
    size: int
    mtime_ns: int


@dataclass(slots=True)
class Manifest:
    name: str        # "llama3:8b", or "namespace/name:tag" outside the default namespace
    path: Path


def list_blobs(base: Path) -> list[Blob]:
    """Every complete blob under <models>/blobs. In-progress downloads (the
    "-partial" suffix) are excluded on purpose: they are expected to be
    incomplete, not corrupt, and hashing a file still being written would
    either fail or report a false mismatch."""
    blobs_dir = base / "blobs"
    if not blobs_dir.is_dir():
        return []
    out = []
    with os.scandir(blobs_dir) as it:
        for entry in it:
            m = _DIGEST_RE.match(entry.name)
            if not m or not entry.is_file(follow_symlinks=False):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue  # vanished between listing and stat
            out.append(Blob(path=Path(entry.path), digest=m.group(1),
                            size=st.st_size, mtime_ns=st.st_mtime_ns))
    return out


def list_manifests(base: Path) -> list[Manifest]:
    """Every manifest file under <models>/manifests, named from its path's last
    three segments (namespace/name/tag) regardless of how deep the registry
    prefix above them goes:
    .../library/<name>/<tag> -> "<name>:<tag>"
    .../<namespace>/<name>/<tag> -> "<namespace>/<name>:<tag>" """
    manifests_dir = base / "manifests"
    if not manifests_dir.is_dir():
        return []
    out = []
    for path in manifests_dir.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(manifests_dir).parts
        if len(parts) < 3:
            continue  # not deep enough to be .../namespace/name/tag; ignore stray files
        namespace, name, tag = parts[-3:]
        display = f"{name}:{tag}" if namespace == "library" else f"{namespace}/{name}:{tag}"
        out.append(Manifest(name=display, path=path))
    return out


def manifest_blob_digests(path: Path) -> list[str]:
    """The blob digests a manifest references (its config and every layer), as
    lowercase hex. A manifest that can't be read or parsed yields an empty list
    rather than an error: a malformed manifest is a finding in its own right
    (its own hash will still be checked), not a reason to abort the scan."""
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            return []
        obj = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return []
    if not isinstance(obj, dict):
        return []
    entries = [obj.get("config")]
    layers = obj.get("layers")
    if isinstance(layers, list):
        entries.extend(layers)
    out: list[str] = []
    for e in entries:
        digest = e.get("digest") if isinstance(e, dict) else None
        if isinstance(digest, str):
            hexpart = digest.removeprefix("sha256:").removeprefix("sha256-").lower()
            if _HEX64_RE.match(hexpart) and hexpart not in out:
                out.append(hexpart)
    return out