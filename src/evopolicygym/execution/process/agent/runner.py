"""Process implementation of the Program Evolution AgentRunner contract."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from ....errors import AgentRunError
from ....run._agent import AgentOutcome, TerminalSignal


class ProcessAgentRunner:
    """Run one non-isolated command-line Coding Agent and reap its process tree."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        workspace: Path,
        environment: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        command_tuple = tuple(command)
        if (
            not command_tuple
            or any(type(item) is not str or not item for item in command_tuple)
        ):
            raise ValueError("command must contain non-empty text arguments")
        if not isinstance(workspace, Path):
            raise TypeError("workspace must be Path")
        if not isinstance(environment, Mapping) or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        ):
            raise TypeError("environment must map text names to text values")
        if not isinstance(stdout_path, Path) or not isinstance(stderr_path, Path):
            raise TypeError("Agent log destinations must be Path values")
        self._command = command_tuple
        self._workspace = workspace
        self._environment = dict(environment)
        self._stdout_path = stdout_path
        self._stderr_path = stderr_path

    def run(
        self,
        terminal: TerminalSignal,
        *,
        timeout_seconds: float,
    ) -> AgentOutcome:
        process: subprocess.Popen[bytes] | None = None
        with (
            self._stdout_path.open("xb") as stdout,
            self._stderr_path.open("xb") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    self._command,
                    cwd=self._workspace,
                    env=self._environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError as error:
                return AgentOutcome(
                    returncode=None,
                    start_failed=True,
                    start_error_type=type(error).__name__,
                    start_errno=error.errno,
                )

            try:
                outcome = _wait_for_agent(
                    process,
                    terminal,
                    timeout_seconds=timeout_seconds,
                )
                _terminate_process_group(process)
                return AgentOutcome(
                    returncode=process.returncode,
                    timed_out=outcome == "timed_out",
                    stopped_after_terminal=(
                        outcome == "stopped_after_terminal"
                    ),
                )
            finally:
                _terminate_process_group(process)


def build_agent_environment(
    socket_path: Path,
    workspace: Path,
    *,
    inherited_names: Sequence[str],
) -> dict[str, str]:
    """Build the minimal environment visible to one local command Agent."""

    package_root = str(Path(__file__).resolve().parents[4])
    command_directory = str(Path(sys.executable).parent.resolve(strict=True))
    inherited_path = os.environ.get("PATH", os.defpath)
    command_path = os.pathsep.join(
        dict.fromkeys((command_directory, *inherited_path.split(os.pathsep)))
    )
    import_roots = [package_root]
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        import_roots.extend(
            str(Path(item).resolve())
            for item in inherited_python_path.split(os.pathsep)
            if item
        )
    environment = {
        "PATH": command_path,
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(import_roots)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "EVOPOLICYGYM_SESSION_SOCKET": os.path.relpath(
            socket_path,
            start=workspace,
        ),
        "EVOPOLICYGYM_WORKSPACE": str(workspace.resolve(strict=True)),
    }
    for name in inherited_names:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


def _wait_for_agent(
    process: subprocess.Popen[bytes],
    terminal: TerminalSignal,
    *,
    timeout_seconds: float,
) -> Literal["exited", "timed_out", "stopped_after_terminal"]:
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if terminal.wait(timeout=0.02):
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                _terminate_process(process)
                return "stopped_after_terminal"
            return "exited"
        if time.monotonic() >= deadline:
            _terminate_process(process)
            return "timed_out"
    process.wait()
    return "exited"


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
    except ProcessLookupError:
        return
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as error:
        raise AgentRunError("Coding Agent process could not be reaped") from error


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Ensure descendants cannot outlive the command Agent on POSIX."""

    if process.poll() is None:
        _terminate_process(process)
    if os.name != "posix":
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


__all__: list[str] = []
