"""Unix-socket transport for one active Submission Session."""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

from .._protocol.session import (
    SESSION_FRAMES,
    SESSION_PROTOCOL,
)
from ._session import (
    FinishReceipt,
    SessionError,
    SubmissionReceipt,
    SubmissionSession,
)


def send_session_message(
    connection: socket.socket,
    message: dict[str, object],
) -> None:
    connection.sendall(SESSION_FRAMES.encode(message))


def receive_session_message(connection: socket.socket) -> dict[str, object]:
    length = SESSION_FRAMES.decode_header(_receive_exact(connection, 4))
    return SESSION_FRAMES.decode_payload(_receive_exact(connection, length))


class UnixSessionGateway:
    """Serve one-request-per-connection Session messages over a Unix socket."""

    def __init__(self, socket_path: Path, session: SubmissionSession) -> None:
        self._socket_path = socket_path
        self._session = session
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.terminal = threading.Event()

    def start(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self._socket_path))
            os.chmod(self._socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(0.1)
        except BaseException:
            listener.close()
            self._socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener
        thread = threading.Thread(
            target=self._serve,
            name="evopolicygym-agent-session",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def close(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            listener.close()
        thread = self._thread
        if thread is not None:
            thread.join()
        self._socket_path.unlink(missing_ok=True)

    def _serve(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(1.0)
                try:
                    request = receive_session_message(connection)
                except Exception:
                    response = _error(
                        "invalid_request",
                        "the Session request could not be decoded",
                    )
                else:
                    try:
                        response = _handle_request(self._session, request)
                    except Exception:
                        self._session.fail()
                        response = _error(
                            "session_failed",
                            "the Host Session failed",
                        )
                try:
                    send_session_message(connection, response)
                except (OSError, ValueError):
                    pass
            if self._session.terminal_reason is not None:
                self.terminal.set()


def _handle_request(
    session: SubmissionSession,
    request: dict[str, object],
) -> dict[str, object]:
    if session.terminal_reason is not None:
        return _error("session_closed", "the Agent Session is already closed")
    if request.get("protocol") != SESSION_PROTOCOL:
        return _error("protocol_mismatch", "unsupported Agent Session protocol")
    method = request.get("method")
    if method == "submit":
        if set(request) != {"protocol", "method", "episodes"}:
            return _error("invalid_request", "submit request fields are invalid")
        submit_outcome = session.submit(request["episodes"])
        if isinstance(submit_outcome, SessionError):
            return _error(submit_outcome.code, submit_outcome.message)
        assert isinstance(submit_outcome, SubmissionReceipt)
        return {
            "protocol": SESSION_PROTOCOL,
            "ok": True,
            "result": {
                "submission_id": submit_outcome.submission_id,
                "program_digest": submit_outcome.program_digest,
                "score": submit_outcome.score,
                "episodes_used": submit_outcome.episodes_used,
                "episodes_remaining": submit_outcome.episodes_remaining,
                "feedback": (
                    "feedback/submissions/"
                    f"{submit_outcome.submission_id}/feedback.json"
                ),
            },
        }
    if method == "finish":
        if set(request) != {"protocol", "method", "submission_id"}:
            return _error("invalid_request", "finish request fields are invalid")
        finish_outcome = session.finish(request["submission_id"])
        if isinstance(finish_outcome, SessionError):
            return _error(finish_outcome.code, finish_outcome.message)
        assert isinstance(finish_outcome, FinishReceipt)
        return {
            "protocol": SESSION_PROTOCOL,
            "ok": True,
            "result": {
                "submission_id": finish_outcome.submission_id,
                "program_digest": finish_outcome.program_digest,
            },
        }
    return _error("invalid_request", "unknown Session method")


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Session frame ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _error(code: str, message: str) -> dict[str, object]:
    return {
        "protocol": SESSION_PROTOCOL,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


__all__: list[str] = []
