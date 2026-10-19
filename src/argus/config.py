"""Runtime configuration: a plain dataclass with ARGUS_* environment overrides."""
from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def default_data_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")  # per-user; never ProgramData
    return (Path(local) if local else Path.home() / ".local" / "share") / "Argus"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Config:
    # Convention: Ollama is moved to 11435 (OLLAMA_HOST) and Argus takes 11434,
    # so existing clients need no reconfiguration.
    upstream_url: str = "http://127.0.0.1:11435"
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 11434
    admin_port: int = 47800
    data_dir: Path = field(default_factory=default_data_dir)
    max_buffered_body: int = 64 * 1024 * 1024  # larger bodies stream through unparsed
    connect_timeout_s: float = 5.0
    write_timeout_s: float = 30.0

    @property
    def upstream_is_local(self) -> bool:
        return _is_loopback(urlsplit(self.upstream_url).hostname or "")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "argus.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        parts = urlsplit(self.upstream_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"ARGUS_UPSTREAM must be an http(s) URL, got {self.upstream_url!r}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if self.upstream_is_local and port == self.proxy_port:
            raise ValueError(
                "Upstream and proxy would share one port. Move Ollama with OLLAMA_HOST "
                "or change ARGUS_PROXY_PORT."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Config:
        d = cls()
        cfg = cls(
            upstream_url=env.get("ARGUS_UPSTREAM", d.upstream_url).rstrip("/"),
            proxy_host=env.get("ARGUS_PROXY_HOST", d.proxy_host),
            proxy_port=int(env.get("ARGUS_PROXY_PORT", d.proxy_port)),
            admin_port=int(env.get("ARGUS_ADMIN_PORT", d.admin_port)),
            data_dir=Path(env["ARGUS_DATA_DIR"]) if "ARGUS_DATA_DIR" in env else d.data_dir,
            max_buffered_body=int(env.get("ARGUS_MAX_BODY", d.max_buffered_body)),
        )
        cfg.validate()
        return cfg