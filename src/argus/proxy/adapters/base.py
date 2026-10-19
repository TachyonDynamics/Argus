from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from argus.core.models import FinalStats, RequestInfo


@dataclass(slots=True)
class ChunkInfo:
    has_output: bool = False   # token, thinking text or tool call
    has_answer: bool = False   # user-visible content


class StreamParser(Protocol):
    final: FinalStats | None
    error_code: str | None     # classification code only; raw error text is never kept

    def feed(self, chunk: bytes) -> ChunkInfo: ...
    def finish(self) -> None: ...


class ProviderAdapter(Protocol):
    name: str

    def matches(self, method: str, path: str) -> bool: ...
    def parse_request(self, body: bytes, path: str) -> RequestInfo: ...
    def new_stream_parser(self, request: RequestInfo, status_code: int) -> StreamParser: ...