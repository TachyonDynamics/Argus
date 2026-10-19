"""uvicorn wiring shared by the application and the tests."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import uvicorn


class Server(uvicorn.Server):
    """A uvicorn server that leaves signal handling to the caller.

    Two servers share one process, and uvicorn's own handlers would each replace
    the previous one, so only one server would ever see Ctrl-C. app.run() installs
    a single handler that stops both.
    """

    def install_signal_handlers(self) -> None:  # uvicorn < 0.30
        pass

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:  # uvicorn >= 0.30
        yield


def make_server(app, *, host: str, port: int, transparent: bool = False) -> Server:
    """`transparent=True` is for the proxy listener: it must not add its own Server or
    Date headers (the upstream's pass through) and must not rewrite request.client
    from X-Forwarded-For (client attribution depends on the real source port)."""
    cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
        ws="none",
        timeout_graceful_shutdown=5,
        server_header=not transparent,
        date_header=not transparent,
        proxy_headers=not transparent,
    )
    return Server(cfg)