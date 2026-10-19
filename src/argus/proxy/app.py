from __future__ import annotations

from starlette.applications import Starlette
from starlette.routing import Route

from argus.proxy.forwarder import Forwarder

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def create_proxy_app(forwarder: Forwarder) -> Starlette:
    """Catch-all route: every path is forwarded, so new Ollama endpoints just work.
    The httpx client is created by the caller (composition root), not in a lifespan,
    so tests can build the app without lifespan support."""
    return Starlette(routes=[Route("/{path:path}", forwarder.handle, methods=_METHODS)])