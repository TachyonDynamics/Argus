from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from urllib.parse import urlencode, urlparse

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

COOKIE = "argus_session"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_CSP = "default-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"


def load_or_create_token(data_dir: Path) -> str:
    path = data_dir / "admin.token"
    try:
        token = path.read_text(encoding="ascii").strip()
        if len(token) >= 32:
            return token
    except FileNotFoundError:
        pass
    token = secrets.token_urlsafe(32)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(token, encoding="ascii")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    return token


def _plain(text: str, status: int) -> Response:
    return PlainTextResponse(text, status_code=status)


def _harden(resp: Response) -> Response:
    resp.headers["Content-Security-Policy"] = _CSP
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


class GuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, token: str, port: int) -> None:
        super().__init__(app)
        self._token = token.encode("ascii")
        self._hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        self._origins = {f"http://{h}" for h in self._hosts}

    def _valid(self, supplied: str) -> bool:
        return hmac.compare_digest(supplied.encode("utf-8", "ignore"), self._token)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        host = request.headers.get("host", "").lower()
        if host not in self._hosts:
            return _harden(_plain("Unexpected Host header.", 400))

        if request.method not in _SAFE_METHODS:
            origin = request.headers.get("origin", "").lower()

            if not origin and "referer" in request.headers:
                parsed = urlparse(request.headers.get("referer", ""))
                if parsed.scheme and parsed.netloc:
                    origin = f"{parsed.scheme}://{parsed.netloc}".lower()

            if origin and origin not in self._origins:
                return _harden(_plain("Cross-origin request refused.", 403))

            fetch_site = request.headers.get("sec-fetch-site", "").lower()
            if fetch_site and fetch_site not in {"same-origin", "same-site", "none"}:
                return _harden(_plain("Cross-origin request refused.", 403))

        if request.url.path.startswith("/static/"):
            return _harden(await call_next(request))

        supplied = request.query_params.get("t")
        if supplied and self._valid(supplied):
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != "t"]
            target = request.url.path + ("?" + urlencode(rest) if rest else "")
            if not target.startswith("/") or target.startswith("//"):
                target = "/"
            resp = RedirectResponse(target, status_code=303)
            resp.set_cookie(
                COOKIE,
                self._token.decode("ascii"),
                httponly=True,
                samesite="strict",
                path="/",
            )
            return _harden(resp)

        if not self._valid(request.cookies.get(COOKIE, "")):
            return _harden(
                _plain(
                    "Open Argus from its tray icon, or from the link Argus printed at startup.",
                    401,
                )
            )

        return _harden(await call_next(request))