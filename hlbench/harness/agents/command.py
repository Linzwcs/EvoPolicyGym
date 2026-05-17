"""Generic command-based agent backend."""

from __future__ import annotations

import subprocess
import time
import os
from dataclasses import dataclass
from pathlib import Path

from hlbench.core.paths import REPO_ROOT


@dataclass(frozen=True)
class CommandResult:
    backend: str
    name: str
    command: tuple[str, ...]
    returncode: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_record(self, *, stdout_path: str | None = None, stderr_path: str | None = None) -> dict[str, object]:
        record: dict[str, object] = {
            "backend": self.backend,
            "name": self.name,
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
        }
        if stdout_path is not None:
            record["stdout_path"] = stdout_path
        if stderr_path is not None:
            record["stderr_path"] = stderr_path
        return record


class CommandAgent:
    def __init__(
        self,
        command: list[str] | tuple[str, ...] | None = None,
        timeout_seconds: int = 1800,
        *,
        backend: str = "command",
        name: str = "custom",
    ) -> None:
        self.command = tuple(command or ())
        self.timeout_seconds = timeout_seconds
        self.backend = backend
        self.name = name

    def run(self, workspace: Path) -> CommandResult:
        if not self.command:
            return CommandResult(
                backend=self.backend,
                name=self.name,
                command=self.command,
                returncode=0,
                timed_out=False,
                duration_seconds=0.0,
                stdout="",
                stderr="agent command skipped",
            )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(self.command),
                cwd=workspace,
                env=_agent_env(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return CommandResult(
                backend=self.backend,
                name=self.name,
                command=self.command,
                returncode=completed.returncode,
                timed_out=False,
                duration_seconds=time.monotonic() - started,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                backend=self.backend,
                name=self.name,
                command=self.command,
                returncode=124,
                timed_out=True,
                duration_seconds=time.monotonic() - started,
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr),
            )


def _agent_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join([src_path, existing])
    return env


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
