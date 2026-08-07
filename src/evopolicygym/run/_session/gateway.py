"""Host-side Unix-socket Gateway for one active Submission Session."""

from __future__ import annotations

import os
import queue
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..._protocol.session import SESSION_PROTOCOL
from .client import receive_session_message, send_session_message
from .outcomes import (
    FinishReceipt,
    SessionError,
    SubmissionReceipt,
)
from .service import SubmissionSession


class UnixSessionGateway:
    """Serve one-request-per-connection Session messages over a Unix socket."""

    def __init__(self, socket_path: Path, session: SubmissionSession) -> None:
        self._socket_path = socket_path
        self._session = session
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._requests: queue.Queue[_PendingRequest] = queue.Queue()
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

    def dispatch_next(self, *, timeout_seconds: float) -> bool:
        """Handle at most one decoded request on the calling Host thread."""

        if type(timeout_seconds) is not float or timeout_seconds < 0.0:
            raise ValueError("timeout_seconds must be a non-negative float")
        try:
            pending = self._requests.get(timeout=timeout_seconds)
        except queue.Empty:
            return False
        try:
            pending.response = _handle_request(self._session, pending.request)
        except Exception:
            self._session.fail()
            pending.response = _error(
                "session_failed",
                "the Host Session failed",
            )
        finally:
            pending.ready.set()
        return True

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
                    pending = _PendingRequest(request=request)
                    self._requests.put(pending)
                    while not pending.ready.wait(timeout=0.1):
                        if self._stop.is_set():
                            return
                    pending_response = pending.response
                    assert pending_response is not None
                    response = pending_response
                try:
                    send_session_message(connection, response)
                except (OSError, ValueError):
                    pass
            if self._session.agent_authority_closed:
                self.terminal.set()


@dataclass(slots=True)
class _PendingRequest:
    request: dict[str, object]
    response: dict[str, object] | None = None
    ready: threading.Event = field(default_factory=threading.Event)


def _handle_request(
    session: SubmissionSession,
    request: dict[str, object],
) -> dict[str, object]:
    if session.agent_authority_closed:
        return _error("session_closed", "the Agent Session is already closed")
    if request.get("protocol") != SESSION_PROTOCOL:
        return _error("protocol_mismatch", "unsupported Agent Session protocol")
    method = request.get("method")
    if method == "submit":
        if set(request) != {"protocol", "method", "episode_indices"}:
            return _error("invalid_request", "submit request fields are invalid")
        submit_outcome = session.submit(request["episode_indices"])
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
                "episode_indices": list(
                    submit_outcome.episode_indices
                ),
                "episodes_used": submit_outcome.episodes_used,
                "episodes_remaining": submit_outcome.episodes_remaining,
                "feedback": (
                    "feedback/submissions/"
                    f"{submit_outcome.submission_id}/feedback.json"
                ),
            },
        }
    if method == "finish":
        if set(request) != {"protocol", "method", "submission_ids"}:
            return _error("invalid_request", "finish request fields are invalid")
        finish_outcome = session.finish(request["submission_ids"])
        if isinstance(finish_outcome, SessionError):
            return _error(finish_outcome.code, finish_outcome.message)
        assert isinstance(finish_outcome, FinishReceipt)
        return {
            "protocol": SESSION_PROTOCOL,
            "ok": True,
            "result": {
                "candidate_submission_ids": list(
                    finish_outcome.candidate_submission_ids
                ),
                "agent_authority_closed": True,
            },
        }
    return _error("invalid_request", "unknown Session method")


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
