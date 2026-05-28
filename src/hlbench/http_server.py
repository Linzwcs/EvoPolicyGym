"""Stdlib HTTP wrapper around ``Server``.

Three endpoints, mirroring SPEC §3:

  GET  /info       → JSON of ``Server.info()``
  POST /submit     → body: ``{"env_instances": [...]}``;
                     response: ``{"submit_id", "status", "summary"}``
  POST /finalize   → response: full ``run.json`` body

Sync only — POST /submit blocks until the submit completes (episodes
are seconds-to-minutes per CLAUDE.md invariant 8). One Server per
process; ``serve()`` runs the loop and never returns until interrupted.

Stdlib only — no fastapi/uvicorn. The agent-facing protocol is small
enough that ``http.server.BaseHTTPRequestHandler`` is the right size.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hlbench.core.server import Server


log = logging.getLogger(__name__)


class HlbenchHandler(BaseHTTPRequestHandler):
    """One handler per request. The shared Server lives on the
    ThreadingHTTPServer instance (``self.server.hlbench_server``)."""

    # --- common helpers ------------------------------------------------

    def _send_json(self, status: int, body: Any) -> None:
        payload = json.dumps(body, default=_jsonable).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise _BadRequest(f"invalid JSON body: {e}") from e

    def _server(self) -> Server:
        return self.server.hlbench_server  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # Route through stdlib logging so callers can mute it.
        log.info("[%s] " + fmt, self.address_string(), *args)

    # --- routing -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        try:
            if self.path == "/info":
                self._send_json(200, self._server().info())
                return
            if self.path == "/health":
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"error": "not_found", "path": self.path})
        except _BadRequest as e:
            self._send_json(400, {"error": "bad_request", "message": str(e)})
        except Exception as e:  # pragma: no cover (defensive)
            log.exception("unhandled error on GET %s", self.path)
            self._send_json(500, {"error": "internal", "message": str(e)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/submit":
                body = self._read_json_body()
                ids = body.get("env_instances")
                if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                    raise _BadRequest(
                        "POST /submit requires {'env_instances': [int, ...]}"
                    )
                result = self._server().submit(ids)
                self._send_json(200, {
                    "submit_id": result.submit_id,
                    "status": result.status,
                    "summary": result.summary,
                })
                return

            if self.path == "/finalize":
                # Body is allowed to be empty; ignore if present.
                self._read_json_body()
                result = self._server().finalize()
                self._send_json(200, asdict(result))
                return

            self._send_json(404, {"error": "not_found", "path": self.path})

        except _BadRequest as e:
            self._send_json(400, {"error": "bad_request", "message": str(e)})
        except RuntimeError as e:
            # Server raises this when submit() is called post-finalize.
            self._send_json(409, {"error": "conflict", "message": str(e)})
        except Exception as e:  # pragma: no cover (defensive)
            log.exception("unhandled error on POST %s", self.path)
            self._send_json(500, {"error": "internal", "message": str(e)})


class _BadRequest(Exception):
    """Maps to HTTP 400."""


def _jsonable(value: Any) -> Any:
    """Fallback encoder: handles Path and dataclasses-as-asdict-output."""
    from pathlib import Path
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


# --------------------------- public entry point ---------------------------


class HlbenchHTTPServer:
    """Background-friendly wrapper around ``ThreadingHTTPServer``.

    Use as a context manager for tests::

        with HlbenchHTTPServer(server, port=0) as http:
            url = f"http://{http.host}:{http.port}"
            ...

    Or call ``.serve_forever_blocking()`` from a CLI for a long-running
    process.
    """

    def __init__(
        self, server: Server, *, host: str = "127.0.0.1", port: int = 8765,
    ) -> None:
        self._http = ThreadingHTTPServer((host, port), HlbenchHandler)
        # Stash the hlbench Server on the http server so handlers can find it.
        self._http.hlbench_server = server  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._http.server_address[0]  # type: ignore[return-value]

    @property
    def port(self) -> int:
        return self._http.server_address[1]  # type: ignore[index]

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("HTTP server already started")
        self._thread = threading.Thread(
            target=self._http.serve_forever, name="hlbench-http", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._http.shutdown()
        self._http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def serve_forever_blocking(self) -> None:
        try:
            self._http.serve_forever()
        finally:
            self._http.server_close()

    # Context manager: spawn / shut down a background thread.
    def __enter__(self) -> "HlbenchHTTPServer":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()
