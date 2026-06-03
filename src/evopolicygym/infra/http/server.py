"""Stdlib HTTP binding for the agent-facing Service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any
from urllib.parse import urlparse

from .api import Service, SubmitRequest

Json = Any


@dataclass(slots=True)
class Server:
    """Small stdlib HTTP server for one EvoPolicyGym Service."""

    service: Service
    host: str = "127.0.0.1"
    port: int = 0
    _http: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> Server:
        if self._http is not None:
            return self
        self._http = ThreadingHTTPServer((self.host, self.port), _handler(self.service))
        self._thread = Thread(target=self._http.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._http is None:
            return
        self._http.shutdown()
        self._http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._http = None
        self._thread = None

    def __enter__(self) -> Server:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def url(self) -> str:
        if self._http is None:
            raise RuntimeError("server is not started")
        host, port = self._http.server_address
        return f"http://{host}:{port}"


def serve(service: Service, *, host: str = "127.0.0.1", port: int = 0) -> Server:
    """Start a stdlib HTTP server for one Service."""

    return Server(service=service, host=host, port=port).start()


def _handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "EvoPolicyGym/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/info":
                self._json(HTTPStatus.OK, service.info().body())
            elif path == "/task":
                task = service.task_doc()
                self._bytes(HTTPStatus.OK, task.text.encode("utf-8"), task.media)
            else:
                self._json(HTTPStatus.NOT_FOUND, _error("not_found", "unknown endpoint"))

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/finalize":
                self._json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    _error("method_not_allowed", "finalize is server-owned"),
                )
                return
            if path != "/submit":
                self._json(HTTPStatus.NOT_FOUND, _error("not_found", "unknown endpoint"))
                return

            body = self._read_json()
            if not isinstance(body, dict) or "env_instances" not in body:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    _error("bad_request", "body must contain env_instances"),
                )
                return

            try:
                response = service.submit(SubmitRequest(body["env_instances"]))
            except ValueError as exc:
                self._json(HTTPStatus.CONFLICT, _error("run_closed", str(exc)))
                return
            self._json(HTTPStatus(response.code), response.body())

        def log_message(self, format: str, *args: object) -> None:
            return None

        def _read_json(self) -> Json:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return None
            data = self.rfile.read(length)
            try:
                return json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                return None

        def _json(self, status: HTTPStatus, data: Json) -> None:
            body = json.dumps(data, sort_keys=True).encode("utf-8")
            self._bytes(status, body, "application/json")

        def _bytes(self, status: HTTPStatus, body: bytes, content: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", f"{content}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _error(status: str, message: str) -> dict[str, Json]:
    return {"status": status, "error": {"message": message}}
