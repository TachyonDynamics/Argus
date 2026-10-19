"""Tiny formatting helpers shared by the UI and the diagnostics engine.

Split out on its own so diagnostics/engine.py doesn't have to import from ui/
(diagnostics is the lower-level module; the UI reads from it, not the reverse).
"""
from __future__ import annotations

DASH = "–"


def fmt_duration(ns: int | None) -> str:
    if ns is None:
        return DASH
    ms = ns / 1e6
    if ms < 1:
        return "<1 ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} min {rest} s"


def fmt_size(n: int | None) -> str:
    if n is None:
        return DASH
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # unreachable; keeps type checkers content


def fmt_ago(seconds: float) -> str:
    """A coarse, human "how long ago". Negative values (clock changes) read as "just now"."""
    s = max(0, int(seconds))
    if s < 60:
        return "just now"
    if s < 3600:
        m = s // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if s < 86400:
        h = s // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = s // 86400
    return f"{d} day{'s' if d != 1 else ''} ago"