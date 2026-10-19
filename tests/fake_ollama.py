"""A scripted stand-in for Ollama, run in-process on an ephemeral port.

Tests set `fake.script` to control exactly when bytes arrive, so timing
assertions compare against known delays instead of a real model.
"""
from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field

import orjson
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from argus.server import make_server

Step = tuple  # ("sleep", s) | ("line", dict) | ("raw", bytes) | ("abort",) | ("hang",)


def sleep(seconds: float) -> Step:
    return ("sleep", seconds)


def token(text: str) -> Step:
    return ("line", {"model": "m", "created_at": "t", "response": text, "done": False})


def chat_token(text: str) -> Step:
    return ("line", {"model": "m", "message": {"role": "assistant", "content": text}, "done": False})


def final(*, load_ns: int = 0, prompt_ns: int = 1_000_000, eval_ns: int = 10_000_000,
          prompt_count: int = 5, eval_count: int = 3, total_ns: int | None = None) -> Step:
    return ("line", {
        "model": "m", "created_at": "t", "response": "", "done": True, "done_reason": "stop",
        "total_duration": total_ns if total_ns is not None else load_ns + prompt_ns + eval_ns,
        "load_duration": load_ns,
        "prompt_eval_count": prompt_count, "prompt_eval_duration": prompt_ns,
        "eval_count": eval_count, "eval_duration": eval_ns,
    })


def error_line(message: str) -> Step:
    return ("line", {"error": message})


def raw(data: bytes) -> Step:
    return ("raw", data)


ABORT: Step = ("abort",)   # drop the connection without finishing the chunked body
HANG: Step = ("hang",)     # go silent forever (until the peer disconnects)


@dataclass
class Script:
    steps: list[Step] = field(default_factory=list)
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    media_type: str = "application/x-ndjson"


class FakeOllama:
    def __init__(self) -> None:
        self.script = Script()
        self.received: list[dict] = []   # one entry per request seen
        self.sent = b""                  # bytes emitted by the most recent script run
        self.cancelled = 0               # streams cut short by the peer disconnecting
        self.port = 0
        self._sock: socket.socket | None = None
        self._server = None
        self._task: asyncio.Task | None = None
        self.app = Starlette(routes=[
            Route("/echo", self._echo, methods=["GET", "POST"]),
            Route("/api/blobs/{digest}", self._blob, methods=["POST"]),
            Route("/api/tags", self._tags, methods=["GET"]),
            Route("/api/{name}", self._scripted, methods=["POST"]),
        ])

    async def start(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self._server = make_server(self.app, host="127.0.0.1", port=self.port, transparent=True)
        self._server.config.log_level = "critical"
        self._task = asyncio.create_task(self._server.serve(sockets=[self._sock]))
        while not self._server.started:
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        assert self._server is not None and self._task is not None
        self._server.should_exit = True
        self._server.force_exit = True
        await self._task

    async def wait_cancelled(self, n: int = 1, timeout: float = 3.0) -> None:
        for _ in range(int(timeout / 0.02)):
            if self.cancelled >= n:
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"fake saw {self.cancelled} cancelled stream(s), expected {n}")

    # ---- endpoints ---------------------------------------------------------
    async def _record(self, request: Request, body: bytes | None = None) -> None:
        self.received.append({
            "method": request.method,
            "path": request.url.path,
            "headers": {k: v for k, v in request.headers.items()},
            "raw_headers": list(request.headers.raw),
            "body": await request.body() if body is None else body,
        })

    async def _scripted(self, request: Request) -> Response:
        await self._record(request)
        script = self.script
        self.sent = b""
        return StreamingResponse(self._play(script), status_code=script.status,
                                 headers=script.headers, media_type=script.media_type)

    async def _play(self, script: Script):
        try:
            for step in script.steps:
                kind = step[0]
                if kind == "sleep":
                    await asyncio.sleep(step[1])
                elif kind in ("line", "raw"):
                    data = orjson.dumps(step[1]) + b"\n" if kind == "line" else step[1]
                    self.sent += data
                    yield data
                elif kind == "abort":
                    raise RuntimeError("scripted abort")
                elif kind == "hang":
                    await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    async def _echo(self, request: Request) -> Response:
        await self._record(request)
        resp = JSONResponse({"headers": dict(request.headers)})
        resp.raw_headers.append((b"set-cookie", b"a=1; Path=/"))
        resp.raw_headers.append((b"set-cookie", b"b=2; Path=/"))
        resp.raw_headers.append((b"access-control-allow-origin", b"http://example.test"))
        resp.raw_headers.append((b"x-fake", b"yes"))
        return resp

    async def _blob(self, request: Request) -> Response:
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
        await self._record(request, body=b"")
        return JSONResponse({"received": total})

    async def _tags(self, request: Request) -> Response:
        await self._record(request)
        return JSONResponse({"models": []})