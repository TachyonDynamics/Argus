"""Trace: one proxied request, stamped with monotonic timestamps.

All stamps come from time.perf_counter_ns(). Do not use time.monotonic() or the
asyncio loop clock for timing: before Python 3.13 they have ~15.6 ms resolution
on Windows. Stamps are stored absolute and exported as offsets from `recv`.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from argus.proxy.adapters.base import StreamParser

log = logging.getLogger(__name__)
now_ns = time.perf_counter_ns
GAP_BUCKETS = 32  # bucket i = gaps whose length in microseconds has bit_length i


class Outcome(str, Enum):
    OK = "ok"
    HTTP_ERROR = "http_error"
    STREAM_ERROR = "stream_error"            # in-band {"error": ...} line
    STREAM_TRUNCATED = "stream_truncated"    # upstream ended without a final chunk
    CLIENT_CANCELLED = "client_cancelled"    # not an error; never show it in red
    UPSTREAM_UNREACHABLE = "upstream_unreachable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_RESET = "upstream_reset"
    PROXY_ERROR = "proxy_error"


@dataclass(slots=True)
class RequestInfo:
    kind: str = "other"          # generate | chat | embed | embeddings | other
    model: str | None = None
    stream: bool = False
    prompt_bytes: int = 0        # request body size; a size proxy, not a token count
    num_ctx: int | None = None
    parse_ok: bool = True


@dataclass(slots=True)
class FinalStats:
    """Timing block from Ollama's final chunk (all durations in nanoseconds)."""
    total_ns: int | None = None
    load_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_ns: int | None = None
    eval_count: int | None = None
    eval_ns: int | None = None
    done_reason: str | None = None


@dataclass(slots=True)
class Stamps:
    recv: int
    body: int | None = None          # request body fully read
    conn0: int | None = None         # upstream TCP connect started / finished
    conn1: int | None = None
    tls0: int | None = None          # TLS handshake (remote upstreams only)
    tls1: int | None = None
    sent: int | None = None          # request fully handed to upstream
    hdr: int | None = None           # response headers received
    first_byte: int | None = None
    first_out: int | None = None     # first token, thinking text or tool call
    first_answer: int | None = None  # first user-visible content
    last_chunk: int | None = None
    end: int | None = None           # response fully accepted downstream


_HTTPCORE_STAMPS = {
    "connection.connect_tcp.started": "conn0",
    "connection.connect_tcp.complete": "conn1",
    "connection.start_tls.started": "tls0",
    "connection.start_tls.complete": "tls1",
    "http11.send_request_body.complete": "sent",
    "http11.receive_response_headers.complete": "hdr",
}


@dataclass
class Trace:
    method: str
    path: str
    client_addr: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    wall_start_ns: int = field(default_factory=time.time_ns)  # display anchor only
    stamps: Stamps = field(default_factory=lambda: Stamps(recv=now_ns()))
    provider: str | None = None          # adapter name; None = plain passthrough
    request: RequestInfo | None = None
    status_code: int | None = None
    outcome: Outcome | None = None
    error_code: str | None = None        # classification code, never raw error text
    final: FinalStats | None = None
    inflight_at_start: int = 0           # inference requests already running on arrival
    inflight_same_model: int = 0
    chunks: int = 0
    bytes_out: int = 0
    client_wait_ns: int = 0              # time the downstream took to accept our writes
    argus_ns: int = 0                    # time spent in Argus's own per-chunk code
    max_gap_ns: int = 0                  # longest upstream silence between chunks
    max_gap_at_ns: int = 0               # ...and when it started, as offset from recv
    gap_hist: list[int] = field(default_factory=lambda: [0] * GAP_BUCKETS)
    _ready_ns: int | None = field(default=None, init=False, repr=False)

    @property
    def is_inference(self) -> bool:
        return self.provider is not None

    # ---- stamping -------------------------------------------------------
    def stamp(self, name: str, t: int | None = None) -> None:
        """First write wins, so keep-alive reuse and retries can't overwrite."""
        if getattr(self.stamps, name) is None:
            setattr(self.stamps, name, now_ns() if t is None else t)

    async def on_httpcore(self, event: str, info: dict) -> None:
        """httpx trace hook. httpcore's async interface requires an async callback."""
        name = _HTTPCORE_STAMPS.get(event)
        if name is not None:
            self.stamp(name)

    def on_chunk(self, chunk: bytes, parser: StreamParser | None) -> None:
        """Called for every upstream chunk. Must never raise (fail-open)."""
        t = now_ns()
        try:
            s = self.stamps
            if s.first_byte is None:
                s.first_byte = t
            elif self._ready_ns is not None:
                self._record_gap(t - self._ready_ns, t)
            s.last_chunk = t
            self.chunks += 1
            self.bytes_out += len(chunk)
            if parser is not None:
                info = parser.feed(chunk)
                if info.has_output and s.first_out is None:
                    s.first_out = t
                if info.has_answer and s.first_answer is None:
                    s.first_answer = t
        except Exception:
            log.debug("on_chunk failed", exc_info=True)
        finally:
            self.argus_ns += now_ns() - t

    def after_yield(self, t_before: int) -> None:
        """Called when the downstream has accepted a chunk.

        Upstream silence is measured from this moment, so a slow client
        (backpressure) is never mistaken for a stalled model.
        """
        t = now_ns()
        self.client_wait_ns += t - t_before
        self._ready_ns = t

    def _record_gap(self, gap: int, t: int) -> None:
        if gap > self.max_gap_ns:
            self.max_gap_ns = gap
            self.max_gap_at_ns = (t - gap) - self.stamps.recv
        bucket = min(GAP_BUCKETS - 1, (gap // 1000).bit_length())
        self.gap_hist[bucket] += 1

    def finish(self, outcome: Outcome, error_code: str | None = None) -> None:
        if self.outcome is not None:  # idempotent
            return
        self.stamp("end")
        self.outcome = outcome
        if error_code:
            self.error_code = error_code

    # ---- persistence ----------------------------------------------------
    def to_row(self) -> dict[str, object]:
        s, r, f = self.stamps, self.request, self.final

        def off(v: int | None) -> int | None:
            return None if v is None else v - s.recv

        def span(a: int | None, b: int | None) -> int | None:
            return None if a is None or b is None else b - a

        return {
            "id": self.id,
            "started_wall_ns": self.wall_start_ns,
            "method": self.method,
            "path": self.path,
            "provider": self.provider,
            "kind": r.kind if r else None,
            "model": r.model if r else None,
            "stream": int(r.stream) if r else None,
            "client_addr": self.client_addr,
            "status_code": self.status_code,
            "outcome": self.outcome.value if self.outcome else "unfinished",
            "error_code": self.error_code,
            "prompt_bytes": r.prompt_bytes if r else None,
            "num_ctx": r.num_ctx if r else None,
            "inflight_at_start": self.inflight_at_start,
            "inflight_same_model": self.inflight_same_model,
            "t_body_ns": off(s.body),
            "t_sent_ns": off(s.sent),
            "t_hdr_ns": off(s.hdr),
            "t_first_byte_ns": off(s.first_byte),
            "t_first_out_ns": off(s.first_out),
            "t_first_answer_ns": off(s.first_answer),
            "t_last_chunk_ns": off(s.last_chunk),
            "t_end_ns": off(s.end),
            "conn_ns": span(s.conn0, s.conn1),
            "tls_ns": span(s.tls0, s.tls1),
            "client_wait_ns": self.client_wait_ns,
            "chunks": self.chunks,
            "bytes_out": self.bytes_out,
            "max_gap_ns": self.max_gap_ns,
            "max_gap_at_ns": self.max_gap_at_ns,
            "gap_hist": orjson.dumps(self.gap_hist).decode(),
            "o_total_ns": f.total_ns if f else None,
            "o_load_ns": f.load_ns if f else None,
            "o_prompt_count": f.prompt_eval_count if f else None,
            "o_prompt_ns": f.prompt_eval_ns if f else None,
            "o_eval_count": f.eval_count if f else None,
            "o_eval_ns": f.eval_ns if f else None,
            "done_reason": f.done_reason if f else None,
            "argus_ns": self.argus_ns,
        }


# Single source of truth for the `requests` table; storage/db.py builds its DDL
# from this, and a test asserts Trace.to_row() has exactly these keys.
ROW_COLUMNS: dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "started_wall_ns": "INTEGER NOT NULL",
    "method": "TEXT NOT NULL",
    "path": "TEXT NOT NULL",
    "provider": "TEXT",
    "kind": "TEXT",
    "model": "TEXT",
    "stream": "INTEGER",
    "client_addr": "TEXT",
    "status_code": "INTEGER",
    "outcome": "TEXT NOT NULL",
    "error_code": "TEXT",
    "prompt_bytes": "INTEGER",
    "num_ctx": "INTEGER",
    "inflight_at_start": "INTEGER",
    "inflight_same_model": "INTEGER",
    "t_body_ns": "INTEGER",
    "t_sent_ns": "INTEGER",
    "t_hdr_ns": "INTEGER",
    "t_first_byte_ns": "INTEGER",
    "t_first_out_ns": "INTEGER",
    "t_first_answer_ns": "INTEGER",
    "t_last_chunk_ns": "INTEGER",
    "t_end_ns": "INTEGER",
    "conn_ns": "INTEGER",
    "tls_ns": "INTEGER",
    "client_wait_ns": "INTEGER",
    "chunks": "INTEGER",
    "bytes_out": "INTEGER",
    "max_gap_ns": "INTEGER",
    "max_gap_at_ns": "INTEGER",
    "gap_hist": "TEXT",
    "o_total_ns": "INTEGER",
    "o_load_ns": "INTEGER",
    "o_prompt_count": "INTEGER",
    "o_prompt_ns": "INTEGER",
    "o_eval_count": "INTEGER",
    "o_eval_ns": "INTEGER",
    "done_reason": "TEXT",
    "argus_ns": "INTEGER",
}