"""Turns database rows into what the page shows. Kept free of web code so it is
easy to test. Numbers come from measurements; nothing here guesses a cause."""
from __future__ import annotations

from datetime import datetime

from ..format import DASH, fmt_duration

# outcome -> (label, tone). Tone drives colour, but every label also says it in words.
_OUTCOMES: dict[str, tuple[str, str]] = {
    "ok": ("OK", "ok"),
    "http_error": ("Failed", "bad"),
    "stream_error": ("Failed mid-answer", "bad"),
    "stream_truncated": ("Answer cut off", "bad"),
    "client_cancelled": ("Stopped by the app", "neutral"),
    "upstream_unreachable": ("Couldn't reach Ollama", "bad"),
    "upstream_timeout": ("Ollama timed out", "bad"),
    "upstream_reset": ("Connection dropped", "bad"),
    "proxy_error": ("Argus error", "bad"),
    "unfinished": ("Unfinished", "neutral"),
}

_DETAILS: dict[str, str] = {
    "out_of_memory": "out of memory",
    "runner_crashed": "model runner crashed",
    "context_length": "context too long",
    "model_not_found": "model not found",
    "bad_request": "bad request",
    "unclassified": "unrecognized error",
    "connection_refused": "connection refused",
    "dns_failure": "name lookup failed",
    "connect_failed": "connect failed",
    "timeout": "timed out",
    "connection_reset": "connection reset",
    "stream_cut": "connection closed mid-answer",
    "no_final_chunk": "no completion marker",
    "client_disconnected": "",
    "cancelled": "",
}


def fmt_time(wall_ns: int, now: datetime | None = None) -> str:
    t = datetime.fromtimestamp(wall_ns / 1e9)  # local time
    today = (now or datetime.now()).date()
    return t.strftime("%H:%M:%S") if t.date() == today else t.strftime("%b %d, %H:%M")


def tokens_per_second(row: dict) -> float | None:
    count, ns = row.get("o_eval_count"), row.get("o_eval_ns")
    if isinstance(count, int) and isinstance(ns, int) and count > 0 and ns > 0:
        return count / (ns / 1e9)
    return None


def _detail(row: dict) -> str:
    code = row.get("error_code") or ""
    if code.startswith("http_") and code[5:].isdigit():
        return f"HTTP {code[5:]}"
    text = _DETAILS.get(code, code)
    status = row.get("status_code")
    if row.get("outcome") == "http_error" and text and isinstance(status, int):
        return f"{text} (HTTP {status})"
    return text


def _kind(row: dict) -> str:
    if row.get("kind"):
        return str(row["kind"])
    text = f"{row.get('method', '')} {row.get('path', '')}".strip()
    return text if len(text) <= 40 else text[:39] + "…"


def present(row: dict, now: datetime | None = None) -> dict[str, object]:
    label, tone = _OUTCOMES.get(str(row["outcome"]), (str(row["outcome"]), "neutral"))
    tps = tokens_per_second(row)
    p, e = row.get("o_prompt_count"), row.get("o_eval_count")
    gap = row.get("max_gap_ns")
    return {
        "id": row["id"],
        "time": fmt_time(int(row["started_wall_ns"]), now),
        "model": row.get("model") or DASH,
        "kind": _kind(row),
        "outcome": label,
        "detail": _detail(row) if tone != "ok" else "",
        "tone": tone,
        "first_token": fmt_duration(row.get("t_first_out_ns")),
        "total": fmt_duration(row.get("t_end_ns")),
        "speed": f"{tps:.0f} tok/s" if tps else DASH,
        "tokens": f"{p} in, {e} out" if isinstance(p, int) and isinstance(e, int) else DASH,
        "longest_pause": fmt_duration(gap) if gap else "None flagged",
    }


# How long a stream can go silent after producing some output before the live
# view calls it "Paused" rather than "Generating". Deliberately the same order
# of magnitude as diagnostics.engine's stall rule, but this is a coarser, live
# read - it doesn't yet know whether that silence is unusual for this model.
LIVE_STALL_NS = 2_000_000_000


def _infer_phase(trace, now_ns: int) -> str:
    """What an in-flight request looks like it's doing right now, from stamps
    alone. No process or GPU signal exists yet to say *why* a phase is slow -
    see diagnostics.engine's module docstring for the same boundary."""
    s = trace.stamps
    if s.sent is None:
        return "Connecting"
    if s.hdr is None:
        return "Waiting for Ollama"
    if s.first_byte is None:
        return "Starting"
    if s.last_chunk is not None and (now_ns - s.last_chunk) > LIVE_STALL_NS:
        return "Paused"
    return "Generating"


def present_live(trace, now_ns: int) -> dict[str, object]:
    req = trace.request
    elapsed_ns = max(0, now_ns - trace.stamps.recv)
    return {
        "id": trace.id,
        "model": (req.model if req and req.model else DASH),
        "kind": (req.kind if req else "other"),
        "client": f"from {trace.client_addr}" if trace.client_addr else "from an unknown address",
        "phase": _infer_phase(trace, now_ns),
        "elapsed": fmt_duration(elapsed_ns),
        "elapsed_ns": elapsed_ns,
    }


_CONFIDENCE_LABEL = {"confirmed": "Confirmed cause", "likely": "Likely cause",
                     "possible": "Possible cause"}


def present_finding(f) -> dict[str, object]:
    return {
        "kicker": _CONFIDENCE_LABEL.get(f.confidence, "Possible cause"),
        "severity": f.severity,
        "title": f.title,
        "explanation": f.explanation,
        "evidence": [{"text": e.text, "measured": e.measured} for e in f.evidence],
        "fix": f.fix,
        "note": f.note,
    }


def one_line_why(row: dict, diagnosis) -> str:
    """A single sentence for the recent-activity ticker on the Now page."""
    if diagnosis.primary is not None:
        return diagnosis.primary.title
    label, tone = _OUTCOMES.get(str(row["outcome"]), (str(row["outcome"]), "neutral"))
    return "Completed normally" if tone == "ok" else label


def waterfall(breakdown) -> list[dict[str, object]]:
    """One row per segment for the timing bar: proportional percentage weight, a
    formatted duration, and whether it was measured or inferred."""
    if breakdown is None or not breakdown.segments:
        return []

    total_ns = sum(s.ns for s in breakdown.segments) or 1
    cursor = 0.0
    result = []

    for s in breakdown.segments:
        pct = max((s.ns / total_ns) * 100, 0.5)
        result.append(
            {
                "kind": s.kind,
                "label": s.label,
                "duration": fmt_duration(s.ns),
                "x": round(cursor, 2),
                "w": round(pct, 2),
                "measured": s.measured,
            }
        )
        cursor += pct

    return result