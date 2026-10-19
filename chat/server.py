"""
Tiny chat UI server for Argus + Ollama.

Serves the web UI and proxies API calls to the Argus proxy (default 11434)
so the browser never hits CORS issues.

Any page open in the person's browser can send requests to 127.0.0.1, so
binding to loopback alone isn't enough - see ui/auth.py's GuardMiddleware,
which this mirrors: every request's Host header is checked (defeats DNS
rebinding, where a malicious page tricks the browser into treating this
server as same-origin) and, for anything that isn't a plain read, the
Origin header must match too (defeats a cross-origin POST/DELETE, which a
browser will still send even though it won't let the page read the reply -
for a request with side effects, sending it at all is the attack).

Usage:
  python server.py
  # open http://127.0.0.1:8790
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_CSP = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"


class Handler(BaseHTTPRequestHandler):
    upstream: str = DEFAULT_UPSTREAM
    allowed_hosts: frozenset[str] = frozenset()
    allowed_origins: frozenset[str] = frozenset()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()
        self.wfile.write(body)

    def _guarded(self, method: str) -> bool:
        """True if this request is allowed to proceed. Mirrors GuardMiddleware:
        the Host header is checked on every request, Origin only on requests
        that aren't a plain, side-effect-free read."""
        host = (self.headers.get("Host") or "").lower()
        if host not in self.allowed_hosts:
            self._send(400, b"Unexpected Host header.", "text/plain; charset=utf-8")
            return False
        if method not in _SAFE_METHODS:
            origin = (self.headers.get("Origin") or "").lower()
            if origin not in self.allowed_origins:
                self._send(403, b"Cross-origin request refused.", "text/plain; charset=utf-8")
                return False
        return True

    def do_GET(self) -> None:
        if not self._guarded("GET"):
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            data = (ROOT / "index.html").read_bytes()
            self._send(200, data, "text/html; charset=utf-8")
            return
        if path.startswith("/api/"):
            self._proxy("GET", path, None)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if not self._guarded("POST"):
            return
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        self._proxy("POST", path, body)

    def do_DELETE(self) -> None:
        if not self._guarded("DELETE"):
            return
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/"):
            self._proxy("DELETE", path, None)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _proxy(self, method: str, path: str, body: bytes | None) -> None:
        url = self.upstream.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        req = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=600) as resp:
                data = resp.read()
                ctype = resp.headers.get("Content-Type", "application/json")
                self._send(resp.status, data, ctype)
        except HTTPError as e:
            data = e.read() if e.fp else str(e).encode()
            ctype = e.headers.get("Content-Type", "application/json") if e.headers else "application/json"
            self._send(e.code, data, ctype)
        except URLError as e:
            msg = json.dumps({"error": f"Cannot reach Argus/Ollama at {self.upstream}: {e.reason}"})
            self._send(502, msg.encode(), "application/json")
        except Exception as e:
            msg = json.dumps({"error": str(e)})
            self._send(500, msg.encode(), "application/json")


def _local_hosts(port: int) -> frozenset[str]:
    return frozenset(f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]"))


def main() -> None:
    p = argparse.ArgumentParser(description="Argus chat UI")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="Argus proxy URL (default http://127.0.0.1:11434)")
    args = p.parse_args()

    Handler.upstream = args.upstream
    Handler.allowed_hosts = _local_hosts(args.port)
    Handler.allowed_origins = frozenset(f"http://{h}" for h in Handler.allowed_hosts)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Chat UI  http://{args.host}:{args.port}")
    print(f"Upstream {args.upstream}  (Argus proxy)")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
