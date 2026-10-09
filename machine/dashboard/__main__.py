"""The read-only dashboard (L16.2) on http://127.0.0.1:8050 - ideas and states, versions, the last runs, the heartbeat,
errors.

    .venv\\Scripts\\python -m machine.dashboard                 on this PC only (127.0.0.1:8050)
    .venv\\Scripts\\python -m machine.dashboard --check         print the six boxes once, start no server
    docker compose up                                          the tests + this page from Docker (compose.yaml)

Three addresses only: / (the page), /status.json (the same, as data), /health ("ok", for Docker's health check). GET
and HEAD only - anything else is refused (405). It serves no file from disk, writes nothing, sends nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from machine.dashboard import mask, page, read

HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
                               "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
TEXT = "text/plain; charset=utf-8"


class Handler(BaseHTTPRequestHandler):
    server_version = "idea-machine-dashboard"
    sys_version = ""

    def do_GET(self) -> None:
        self._answer(body=True)

    def do_HEAD(self) -> None:
        self._answer(body=False)

    def _refuse(self) -> None:
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, b"read only\n", TEXT, body=True, allow="GET, HEAD")

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _refuse

    def _answer(self, *, body: bool) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                data, kind = page.html_page(read.latest()).encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/status.json":
                data = json.dumps(read.latest(), ensure_ascii=False, indent=1).encode("utf-8")
                kind = "application/json; charset=utf-8"
            elif path == "/health":
                data, kind = b"ok\n", TEXT
            else:
                data, kind = b"not here\n", TEXT
                return self._send(HTTPStatus.NOT_FOUND, data, kind, body=body)
        except Exception as err:                          # a bug: say so, masked, and keep serving
            data = mask.every_digit(f"the page could not be made: {type(err).__name__}: {err}")[:300].encode("utf-8")
            return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, data, TEXT, body=body)
        self._send(HTTPStatus.OK, data, kind, body=body)

    def _send(self, status: HTTPStatus, data: bytes, kind: str, *, body: bool, allow: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        if allow:
            self.send_header("Allow", allow)
        for k, v in HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        if sys.stderr is None or self.path == "/health":   # pythonw has no console; Docker asks /health every minute
            return
        sys.stderr.write(f"{read.now():%Y-%m-%d %H:%M:%S} {fmt % args}\n")


def server(host: str = "127.0.0.1", port: int = 8050) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.dashboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this PC only (in Docker: 0.0.0.0, and "
                                                        "compose.yaml opens it to this PC only)")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--check", action="store_true", help="print the six boxes once and stop")
    a = ap.parse_args(argv)
    if a.check:
        for x in read.snapshot()["now"]:
            print(f"{page.WORD.get(x['state'], x['state']):9} {x['name']}: {x['value']} - {x['line']}")
        return 0
    srv = server(a.host, a.port)
    print(f"The dashboard: http://127.0.0.1:{srv.server_address[1]} (read only; Ctrl+C stops it)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
