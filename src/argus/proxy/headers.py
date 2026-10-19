"""Header filtering. Works on raw (bytes, bytes) pairs to preserve duplicates
(e.g. several Set-Cookie headers) and original casing."""
from __future__ import annotations

from collections.abc import Iterable

Raw = list[tuple[bytes, bytes]]

_HOP_BY_HOP = frozenset({
    b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"te", b"trailer", b"transfer-encoding", b"upgrade",
})


def _named_in_connection(raw: Iterable[tuple[bytes, bytes]]) -> set[bytes]:
    named: set[bytes] = set()
    for k, v in raw:
        if k.lower() == b"connection":
            named.update(t.strip().lower() for t in v.split(b",") if t.strip())
    return named


def filter_headers(raw: Iterable[tuple[bytes, bytes]], drop: Iterable[bytes] = ()) -> Raw:
    pairs = list(raw)
    skip = _HOP_BY_HOP | _named_in_connection(pairs) | {d.lower() for d in drop}
    return [(k, v) for k, v in pairs if k.lower() not in skip]


def request_headers(
    raw: Iterable[tuple[bytes, bytes]], *, buffered: bool, identity: bool
) -> Raw:
    """Headers for the upstream request.

    Host is dropped so httpx sets it from the upstream URL. If we buffered the
    body, httpx computes Content-Length itself. On inference paths we ask for an
    uncompressed body so streamed chunks stay parseable.
    """
    drop = [b"host"]
    if buffered:
        drop.append(b"content-length")
    if identity:
        drop.append(b"accept-encoding")
    out = filter_headers(raw, drop)
    if identity:
        out.append((b"accept-encoding", b"identity"))
    return out


def response_headers(raw: Iterable[tuple[bytes, bytes]]) -> Raw:
    """Headers for the client response. Everything else is forwarded untouched
    (CORS, Content-Type, Content-Length, Set-Cookie, ...)."""
    return filter_headers(raw)