"""Ollama native API: /api/generate, /api/chat, /api/embed, /api/embeddings.

Streaming responses are NDJSON; chunk boundaries never align with line
boundaries, so lines are buffered. Prompts and responses are never stored here.
"""
from __future__ import annotations

import re

import orjson

from argus.core.models import FinalStats, RequestInfo

from .base import ChunkInfo

_KINDS = {
    "/api/generate": "generate",
    "/api/chat": "chat",
    "/api/embed": "embed",
    "/api/embeddings": "embeddings",
}
_STREAMING_KINDS = frozenset({"generate", "chat"})
_MAX_BUFFER = 16 * 1024 * 1024

# Order matters: most specific first.
_ERROR_PATTERNS = (
    ("out_of_memory", re.compile(
        r"out of memory|cuda error|requires more system memory|insufficient (?:vram|memory)", re.I)),
    ("runner_crashed", re.compile(r"runner process|exit status|unexpectedly stopped", re.I)),
    ("context_length", re.compile(r"context (?:length|window)", re.I)),
    ("model_not_found", re.compile(r"not found|try pulling|file does not exist", re.I)),
    ("bad_request", re.compile(r"invalid|unmarshal|missing|required", re.I)),
)


def classify_error_text(text: str) -> str:
    for code, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return code
    return "unclassified"


def _int(v: object) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _final_from(obj: dict) -> FinalStats:
    reason = obj.get("done_reason")
    return FinalStats(
        total_ns=_int(obj.get("total_duration")),
        load_ns=_int(obj.get("load_duration")),
        prompt_eval_count=_int(obj.get("prompt_eval_count")),
        prompt_eval_ns=_int(obj.get("prompt_eval_duration")),
        eval_count=_int(obj.get("eval_count")),
        eval_ns=_int(obj.get("eval_duration")),
        done_reason=reason if isinstance(reason, str) else None,
    )


class OllamaStreamParser:
    def __init__(self, req: RequestInfo, status_code: int) -> None:
        self._kind = req.kind
        self._stream = req.stream
        self._ok = status_code < 400
        self._buf = bytearray()
        self._overflow = False
        self._seen = False
        self.final: FinalStats | None = None
        self.error_code: str | None = None

    def feed(self, chunk: bytes) -> ChunkInfo:
        info = ChunkInfo()
        if not chunk:
            return info
        if not self._stream and not self._seen:
            # Non-streamed reply: the first byte only arrives once generation is done.
            self._seen = True
            info.has_output = self._ok
            info.has_answer = self._ok and self._kind in _STREAMING_KINDS
        if self._overflow:
            return info
        self._buf += chunk
        if self._stream:
            while (i := self._buf.find(b"\n")) >= 0:
                line = bytes(self._buf[:i])
                del self._buf[: i + 1]
                self._handle(line, info)
        if len(self._buf) > _MAX_BUFFER:
            self._buf.clear()
            self._overflow = True
        return info

    def finish(self) -> None:
        """Parse whatever is left: a non-streamed body, or a last line with no newline."""
        if self._buf and not self._overflow:
            self._handle(bytes(self._buf), ChunkInfo())
        self._buf.clear()

    def _handle(self, line: bytes, info: ChunkInfo) -> None:
        line = line.strip()
        if not line:
            return
        try:
            obj = orjson.loads(line)
        except orjson.JSONDecodeError:
            return
        if not isinstance(obj, dict):
            return

        err = obj.get("error")
        if err:
            self.error_code = classify_error_text(err if isinstance(err, str) else str(err))
            return

        if self._kind == "generate":
            if obj.get("response") or obj.get("thinking"):
                info.has_output = True
            if obj.get("response"):
                info.has_answer = True
        elif self._kind == "chat":
            msg = obj.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if content or msg.get("thinking") or msg.get("tool_calls"):
                    info.has_output = True
                if content:
                    info.has_answer = True

        # Generate/chat end with done:true; embed replies carry total_duration instead.
        if obj.get("done") is True or "total_duration" in obj:
            self.final = _final_from(obj)


class OllamaAdapter:
    name = "ollama"

    def matches(self, method: str, path: str) -> bool:
        return method == "POST" and path.rstrip("/") in _KINDS

    def parse_request(self, body: bytes, path: str) -> RequestInfo:
        kind = _KINDS[path.rstrip("/")]
        default_stream = kind in _STREAMING_KINDS
        try:
            obj = orjson.loads(body)
        except orjson.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict):
            return RequestInfo(kind=kind, stream=default_stream, prompt_bytes=len(body), parse_ok=False)

        s = obj.get("stream")
        stream = default_stream if s is None else bool(s)
        if kind not in _STREAMING_KINDS:
            stream = False
        opts = obj.get("options")
        num_ctx = _int(opts.get("num_ctx")) if isinstance(opts, dict) else None
        model = obj.get("model")
        return RequestInfo(
            kind=kind,
            model=model if isinstance(model, str) else None,
            stream=stream,
            prompt_bytes=len(body),
            num_ctx=num_ctx,
        )

    def new_stream_parser(self, request: RequestInfo, status_code: int) -> OllamaStreamParser:
        return OllamaStreamParser(request, status_code)