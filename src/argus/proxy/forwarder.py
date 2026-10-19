"""Transparent streaming forwarder. It observes traffic and never modifies it.

Run uvicorn with server_header=False and date_header=False so the upstream's own
Server and Date headers pass through instead of being duplicated.
"""
from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import AsyncIterator
from email.utils import formatdate

import anyio
import httpx
import orjson
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response, StreamingResponse

from argus.config import Config
from argus.core.models import Outcome, RequestInfo, Trace, now_ns
from argus.proxy import adapters
from argus.proxy.adapters.base import ProviderAdapter, StreamParser
from argus.proxy.headers import request_headers, response_headers
from argus.proxy.tracker import Tracker

log = logging.getLogger(__name__)


class _ClientGone(Exception):
    pass


def make_client(cfg: Config) -> httpx.AsyncClient:
    local = cfg.upstream_is_local
    return httpx.AsyncClient(
        base_url=cfg.upstream_url,
        # Local upstream: no keep-alive, so every request gets a real connect stamp
        # and there are no stale connections after Ollama restarts.
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=0 if local else 20),
        # Models can be silent for minutes: never time out reads, only flag stalls.
        timeout=httpx.Timeout(connect=cfg.connect_timeout_s, read=None,
                              write=cfg.write_timeout_s, pool=None),
        follow_redirects=False,
        http2=False,
        trust_env=not local,  # don't let a system proxy hijack loopback traffic
    )


def classify_exception(exc: BaseException, *, streaming: bool) -> tuple[Outcome, str]:
    root: BaseException = exc
    for _ in range(8):  # follow the cause chain to the OS-level error
        nxt = root.__cause__ or root.__context__
        if nxt is None:
            break
        root = nxt
    if isinstance(root, ConnectionRefusedError):
        return Outcome.UPSTREAM_UNREACHABLE, "connection_refused"
    if isinstance(root, socket.gaierror):
        return Outcome.UPSTREAM_UNREACHABLE, "dns_failure"
    if isinstance(exc, httpx.TimeoutException):
        return Outcome.UPSTREAM_TIMEOUT, "timeout"
    if isinstance(exc, httpx.ConnectError):
        return Outcome.UPSTREAM_UNREACHABLE, "connect_failed"
    if isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError, httpx.CloseError)):
        return (Outcome.STREAM_TRUNCATED, "stream_cut") if streaming else (
            Outcome.UPSTREAM_RESET, "connection_reset")
    return Outcome.PROXY_ERROR, type(exc).__name__


def _client_addr(request: Request) -> str | None:
    c = request.client
    return f"{c.host}:{c.port}" if c else None


def _content_length(request: Request) -> int | None:
    try:
        n = int(request.headers["content-length"])
    except (KeyError, ValueError):
        return None
    return n if n >= 0 else None


def _has_body(request: Request) -> bool:
    n = _content_length(request)
    return (n is not None and n > 0) or "transfer-encoding" in request.headers


def _target(request: Request) -> str:
    raw = request.scope.get("raw_path") or request.url.path.encode("utf-8")
    qs = request.scope.get("query_string", b"")
    return raw.decode("latin-1") + ("?" + qs.decode("latin-1") if qs else "")


def _parse_request(adapter: ProviderAdapter, body: bytes, path: str) -> RequestInfo:
    try:
        return adapter.parse_request(body, path)
    except Exception:
        log.debug("parse_request failed", exc_info=True)
        return RequestInfo(kind="other", prompt_bytes=len(body), parse_ok=False)


def _finish_parser(parser: StreamParser | None) -> None:
    if parser is not None:
        try:
            parser.finish()
        except Exception:
            log.debug("parser.finish failed", exc_info=True)


async def _wait_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)


class Forwarder:
    def __init__(self, cfg: Config, client: httpx.AsyncClient, tracker: Tracker) -> None:
        self.cfg = cfg
        self.client = client
        self.tracker = tracker

    async def handle(self, request: Request) -> Response:
        path = request.url.path
        trace = Trace(request.method, path, client_addr=_client_addr(request))
        adapter = adapters.match(request.method, path)
        trace.provider = adapter.name if adapter else None

        # Small inference bodies are buffered and parsed. Anything else (model blobs
        # for `ollama create`, unknown lengths) streams through unbuffered.
        clen = _content_length(request)
        buffered = adapter is not None and clen is not None and clen <= self.cfg.max_buffered_body
        content: bytes | AsyncIterator[bytes] | None
        if buffered:
            assert adapter is not None
            try:
                content = await request.body()
            except ClientDisconnect:
                return Response(status_code=499)
            trace.stamp("body")
            trace.request = _parse_request(adapter, content, path)
        elif _has_body(request):
            content = request.stream()
        else:
            content = None

        self.tracker.start(trace)
        req = self.client.build_request(
            request.method,
            _target(request),
            headers=request_headers(request.headers.raw, buffered=buffered,
                                    identity=adapter is not None),
            content=content,
            extensions={"trace": trace.on_httpcore},
        )

        try:
            # Watch for client disconnect while waiting on a slow load or prompt eval,
            # so Ollama's work is cancelled just as it would be without Argus.
            resp = await self._send(request, req, watch=buffered)
        except _ClientGone:
            self._finish(trace, Outcome.CLIENT_CANCELLED, "client_disconnected")
            return Response(status_code=499)
        except asyncio.CancelledError:
            self._finish(trace, Outcome.CLIENT_CANCELLED, "cancelled")
            raise
        except Exception as exc:
            outcome, code = classify_exception(exc, streaming=False)
            if outcome is Outcome.PROXY_ERROR:
                log.exception("unexpected error forwarding %s %s", request.method, path)
            err = self._upstream_error(outcome, code)
            trace.status_code = err.status_code
            self._finish(trace, outcome, code)
            return err

        trace.status_code = resp.status_code
        trace.stamp("hdr")
        parser = (adapter.new_stream_parser(trace.request, resp.status_code)
                  if adapter and trace.request else None)

        async def stream() -> AsyncIterator[bytes]:
            # If we're torn down early (client left), this default stands.
            # Note: cancellation arrives as CancelledError while awaiting upstream,
            # or as GeneratorExit if the generator was suspended at a yield.
            outcome, code = Outcome.CLIENT_CANCELLED, "client_disconnected"
            try:
                async for chunk in resp.aiter_raw():
                    trace.on_chunk(chunk, parser)
                    t0 = now_ns()
                    yield chunk
                    trace.after_yield(t0)
                _finish_parser(parser)
                outcome, code = self._success_outcome(trace, parser)
            except httpx.HTTPError as exc:  # upstream died mid-stream
                outcome, code = classify_exception(exc, streaming=True)
                raise  # abort so the client sees the same truncation it would see directly
            except Exception as exc:
                outcome, code = Outcome.PROXY_ERROR, type(exc).__name__
                raise
            finally:
                if parser is not None:
                    trace.final = parser.final
                with anyio.CancelScope(shield=True):
                    await resp.aclose()  # closing the connection cancels generation upstream
                self._finish(trace, outcome, code)

        out = StreamingResponse(stream(), status_code=resp.status_code)
        out.raw_headers = response_headers(resp.headers.raw)
        return out

    async def _send(self, request: Request, req: httpx.Request, *, watch: bool) -> httpx.Response:
        if not watch:
            return await self.client.send(req, stream=True)
        send = asyncio.ensure_future(self.client.send(req, stream=True))
        watcher = asyncio.ensure_future(_wait_disconnect(request))
        try:
            await asyncio.wait({send, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if send.done():
                return send.result()
            send.cancel()
            result = (await asyncio.gather(send, return_exceptions=True))[0]
            if isinstance(result, httpx.Response):  # lost a race: don't leak the connection
                await result.aclose()
            raise _ClientGone
        finally:
            watcher.cancel()
            if not send.done():
                send.cancel()

    @staticmethod
    def _success_outcome(trace: Trace, parser: StreamParser | None) -> tuple[Outcome, str | None]:
        status = trace.status_code or 0
        code = parser.error_code if parser else None
        if status >= 400:
            return Outcome.HTTP_ERROR, code or f"http_{status}"
        if code:
            return Outcome.STREAM_ERROR, code  # in-band error, HTTP 200
        req = trace.request
        if (parser is not None and req is not None and req.parse_ok
                and req.kind in ("generate", "chat") and parser.final is None):
            return Outcome.STREAM_TRUNCATED, "no_final_chunk"
        return Outcome.OK, None

    @staticmethod
    def _upstream_error(outcome: Outcome, code: str) -> Response:
        detail = {
            Outcome.UPSTREAM_UNREACHABLE: "could not reach the upstream server",
            Outcome.UPSTREAM_TIMEOUT: "the upstream server timed out",
            Outcome.UPSTREAM_RESET: "the upstream server closed the connection",
        }.get(outcome, "the proxy hit an internal error")
        return Response(
            orjson.dumps({"error": f"Argus: {detail} ({code})"}),
            status_code=504 if outcome is Outcome.UPSTREAM_TIMEOUT else 502,
            media_type="application/json",
            headers={"date": formatdate(usegmt=True)},
        )

    def _finish(self, trace: Trace, outcome: Outcome, code: str | None) -> None:
        try:
            trace.finish(outcome, code)
            self.tracker.finish(trace)
        except Exception:
            log.exception("finishing trace failed")