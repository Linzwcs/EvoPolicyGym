"""Host controller for one fresh Episode-local Policy process."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

from ...._protocol.policy import decode_policy_value, encode_policy_value
from ....evaluation._service import (
    PolicyRuntimeCleanupError,
    PolicyRuntimeError,
)
from ....policy import PolicyContext, PolicyValue
from ....program import Program
from ....results import PolicyFailureCode
from .stream import read_policy_message, write_policy_message


class PolicyProcessError(PolicyRuntimeError):
    """Private base class for Policy process control failures."""


class PolicyProcessTimeout(PolicyProcessError):
    """The Policy process did not answer before its deadline."""


class PolicyProcessProtocolError(PolicyProcessError):
    """The Policy process violated the private framed protocol."""


class PolicyProcessCleanupError(PolicyRuntimeCleanupError):
    """The Policy process could not be reaped."""


class PolicyProcess:
    """Host controller for exactly one Episode-local Policy process."""

    def __init__(self, program: Program, context: PolicyContext) -> None:
        if type(program) is not Program:
            raise TypeError("program must be Program")
        if type(context) is not PolicyContext:
            raise TypeError("context must be PolicyContext")
        self._program = program
        self._context = context
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._stderr: BinaryIO | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._messages: queue.Queue[dict[str, object] | BaseException] = queue.Queue()

    def start(self, *, timeout_seconds: float) -> PolicyFailureCode | None:
        if self._process is not None:
            raise RuntimeError("Policy process was already started")
        temporary = tempfile.TemporaryDirectory(
            prefix="evopolicygym-episode-",
            ignore_cleanup_errors=True,
        )
        self._temporary = temporary
        root = Path(temporary.name)
        program_directory = root / "program"
        self._program.write_to(program_directory)
        stderr = (root / "policy.stderr.log").open("wb")
        self._stderr = stderr

        environment = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        import_roots = [str(Path(__file__).resolve().parents[3])]
        python_path = os.environ.get("PYTHONPATH")
        if python_path:
            import_roots.extend(
                str(Path(item).resolve())
                for item in python_path.split(os.pathsep)
                if item
            )
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(import_roots))
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.execution.process.policy.worker",
                ],
                cwd=program_directory,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as error:
            self._release_files()
            raise PolicyProcessProtocolError("Policy process could not start") from error
        self._process = process
        assert process.stdout is not None
        reader = threading.Thread(
            target=self._read_messages,
            args=(process.stdout,),
            name="evopolicygym-policy-reader",
            daemon=True,
        )
        self._reader = reader
        reader.start()

        try:
            self._send(_context_message(self._context))
            response = self._receive(timeout_seconds=timeout_seconds)
        except PolicyProcessTimeout:
            return "timeout"
        except PolicyProcessError:
            return "protocol_error"
        if response.get("type") == "ready":
            return None
        return _failure_from_response(response)

    def act(
        self,
        observation: PolicyValue,
        *,
        timeout_seconds: float,
    ) -> tuple[PolicyValue | None, PolicyFailureCode | None]:
        try:
            self._send(
                {
                    "type": "act",
                    "observation": encode_policy_value(observation),
                }
            )
            response = self._receive(timeout_seconds=timeout_seconds)
        except PolicyProcessTimeout:
            return None, "timeout"
        except PolicyProcessError:
            return None, "protocol_error"

        if response.get("type") == "action":
            try:
                action = decode_policy_value(response.get("value"))
            except (TypeError, ValueError, RecursionError):
                return None, "protocol_error"
            return action, None
        return None, _failure_from_response(response)

    def close(self) -> None:
        process = self._process
        cleanup_failed = False
        if process is not None:
            if process.poll() is None:
                try:
                    self._send({"type": "close"})
                except PolicyProcessError:
                    pass
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    _signal_process(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        _signal_process(process, signal.SIGKILL)
                        try:
                            process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            cleanup_failed = True
            if process.poll() is None:
                cleanup_failed = True

        reader = self._reader
        if reader is not None:
            reader.join(timeout=0.25)
        self._release_files()
        if cleanup_failed:
            raise PolicyProcessCleanupError("Policy process could not be reaped")

    def _send(self, message: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise PolicyProcessProtocolError("Policy process is not available")
        try:
            write_policy_message(cast(BinaryIO, process.stdin), message)
        except (BrokenPipeError, OSError) as error:
            raise PolicyProcessProtocolError("Policy process write failed") from error

    def _receive(self, *, timeout_seconds: float) -> dict[str, object]:
        if timeout_seconds <= 0:
            raise PolicyProcessTimeout()
        try:
            item = self._messages.get(timeout=timeout_seconds)
        except queue.Empty:
            raise PolicyProcessTimeout() from None
        if isinstance(item, BaseException):
            raise PolicyProcessProtocolError("Policy process read failed") from item
        return item

    def _read_messages(self, stream: BinaryIO) -> None:
        try:
            while True:
                self._messages.put(read_policy_message(stream))
        except BaseException as error:
            self._messages.put(error)

    def _release_files(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        stderr = self._stderr
        if stderr is not None:
            try:
                stderr.close()
            except OSError:
                pass
        temporary = self._temporary
        if temporary is not None:
            temporary.cleanup()
        self._stderr = None
        self._temporary = None


class ProcessPolicyRuntimeFactory:
    def create(
        self,
        program: Program,
        context: PolicyContext,
    ) -> PolicyProcess:
        return PolicyProcess(program, context)


def _signal_process(
    process: subprocess.Popen[bytes],
    selected_signal: signal.Signals,
) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, selected_signal)
        elif selected_signal == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _context_message(context: PolicyContext) -> dict[str, object]:
    return {
        "type": "context",
        "observation_space": encode_policy_value(context.observation_space),
        "action_space": encode_policy_value(context.action_space),
        "metadata": encode_policy_value(dict(context.metadata)),
        "policy_seed": str(context.policy_seed),
    }


def _failure_from_response(response: Mapping[str, object]) -> PolicyFailureCode:
    if response.get("type") != "error":
        return "protocol_error"
    code = response.get("code")
    if code in {"exception", "timeout", "invalid_action", "protocol_error"}:
        return code
    return "protocol_error"


__all__: list[str] = []
