"""Run workspace layout, retained invocation, events, and terminal record."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .._version import __version__
from ..agents import AgentInvocation
from ..errors import AgentRunError
from ..execution.process.agent.runner import AgentExit
from ..program import Program
from ..results import RunResult
from . import RunConfig

_RUN_RECORD_SCHEMA = "evopolicygym/run-record/v1"
_RUN_EVENT_SCHEMA = "evopolicygym/run-event/v1"
_INVOCATION_SCHEMA = "evopolicygym/agent-invocation/v1"
_SESSION_SOCKET_VARIABLE = "EVOPOLICYGYM_SESSION_SOCKET"
_WORKSPACE_VARIABLE = "EVOPOLICYGYM_WORKSPACE"


@dataclass(frozen=True, slots=True)
class RunDirectoryPaths:
    root: Path
    workspace: Path
    program: Path
    feedback: Path
    initial: Path
    submissions: Path
    agent: Path
    control: Path
    socket: Path
    events: Path

    @classmethod
    def under(cls, root: Path) -> RunDirectoryPaths:
        workspace = root / "workspace"
        control = root / "control"
        return cls(
            root=root,
            workspace=workspace,
            program=workspace / "program",
            feedback=workspace / "feedback",
            initial=root / "initial",
            submissions=root / "submissions",
            agent=root / "agent",
            control=control,
            socket=control / "session.sock",
            events=root / "events.jsonl",
        )


class WorkspaceProgramSource:
    """Capture the mutable Program candidate in one active Run workspace."""

    def __init__(self, directory: Path) -> None:
        if not isinstance(directory, Path):
            raise TypeError("directory must be Path")
        self._directory = directory

    def capture(self) -> Program:
        return Program.from_directory(self._directory)


class RunDirectoryRecorder:
    """Append Host events and atomically commit the terminal Run manifest."""

    def __init__(
        self,
        *,
        paths: RunDirectoryPaths,
        benchmark_id: str,
        initial_program: Program,
        config: RunConfig,
        agent_identity: Mapping[str, str],
    ) -> None:
        self._paths = paths
        self._benchmark_id = benchmark_id
        self._initial_program = initial_program
        self._config = config
        self._agent_identity = dict(agent_identity)
        self._events: TextIO | None = None

    def __enter__(self) -> RunDirectoryRecorder:
        self._events = self._paths.events.open("x", encoding="utf-8")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        del exception_type, exception, traceback
        events = self._events
        if events is not None:
            events.close()
        self._events = None

    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        events = self._events
        if events is None:
            raise RuntimeError("Run recorder is not open")
        append_event(events, event, fields)

    def commit(self, result: RunResult, agent_exit: AgentExit) -> None:
        _write_run_manifest(
            self._paths.root / "run.json",
            result,
            benchmark_id=self._benchmark_id,
            initial_program=self._initial_program,
            config=self._config,
            agent_exit=agent_exit,
            agent_identity=self._agent_identity,
        )


def prepare_run_directory(
    root: Path,
    initial_program: Program,
) -> RunDirectoryPaths:
    if not isinstance(root, Path):
        raise TypeError("run directory must be Path")
    if root.exists() or root.is_symlink():
        raise AgentRunError("run_directory must not already exist")
    if not root.parent.is_dir():
        raise AgentRunError("run_directory parent does not exist")
    root.mkdir(mode=0o700)
    paths = RunDirectoryPaths.under(root)
    for directory in (
        paths.workspace,
        paths.feedback,
        paths.initial,
        paths.submissions,
        paths.agent,
        paths.control,
    ):
        directory.mkdir(mode=0o700)
    initial_program.write_to(paths.initial / "program")
    initial_program.write_to(paths.program)
    _make_tree_read_only(paths.initial / "program")
    return paths


def retain_agent_invocation(
    paths: RunDirectoryPaths,
    invocation: AgentInvocation,
) -> None:
    if invocation.instructions is not None:
        _write_text_file(
            paths.agent / "instructions.md",
            invocation.instructions,
        )
    _write_invocation(paths.agent / "invocation.json", invocation)


def remove_control_directory(control: Path) -> None:
    try:
        control.rmdir()
    except OSError:
        pass


def append_event(
    stream: TextIO,
    event: str,
    fields: Mapping[str, object],
) -> None:
    document = {
        "schema": _RUN_EVENT_SCHEMA,
        "time_unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "event": event,
        **fields,
    }
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    stream.write(payload + "\n")
    stream.flush()


def _write_invocation(
    path: Path,
    invocation: AgentInvocation,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema": _INVOCATION_SCHEMA,
            "agent": dict(invocation.identity),
            "command": list(invocation.recorded_command),
            "cwd": "workspace",
            "environment": {
                "fixed_names": [
                    "PATH",
                    "PYTHONPATH",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONUNBUFFERED",
                    _SESSION_SOCKET_VARIABLE,
                    _WORKSPACE_VARIABLE,
                ],
                "inherited_allowlist": list(
                    invocation.inherited_environment
                ),
            },
            "instructions": (
                "agent/instructions.md"
                if invocation.instructions is not None
                else None
            ),
            "stdout": "agent/stdout.log",
            "stdout_media_type": invocation.stdout_media_type,
            "stderr": "agent/stderr.log",
        },
    )


def _write_run_manifest(
    path: Path,
    result: RunResult,
    *,
    benchmark_id: str,
    initial_program: Program,
    config: RunConfig,
    agent_exit: AgentExit,
    agent_identity: dict[str, str],
) -> None:
    submissions = [
        {
            "submission_id": item.submission_id,
            "program_digest": item.program_digest,
            "episodes_used": item.episodes_used,
            "episodes_remaining": item.episodes_remaining,
            "score": item.feedback.score,
            "record": f"submissions/{item.submission_id}",
        }
        for item in result.submissions
    ]
    _write_json_atomic(
        path,
        {
            "schema": _RUN_RECORD_SCHEMA,
            "library_version": __version__,
            "benchmark": {"id": benchmark_id},
            "initial_program": {
                "digest": initial_program.digest,
                "record": "initial/program",
            },
            "workspace": {
                "root": "workspace",
                "program": "workspace/program",
                "feedback": "workspace/feedback",
            },
            "events": "events.jsonl",
            "config": {
                "split": config.split,
                "seed": config.seed,
                "max_submissions": config.max_submissions,
                "episode_budget": config.episode_budget,
                "max_episodes_per_submission": (
                    config.max_episodes_per_submission
                ),
                "episode_timeout_seconds": config.episode_timeout_seconds,
                "agent_timeout_seconds": config.agent_timeout_seconds,
            },
            "agent": {
                **agent_identity,
                "invocation": "agent/invocation.json",
                "stdout": "agent/stdout.log",
                "stderr": "agent/stderr.log",
                "timed_out": agent_exit.timed_out,
                "stopped_after_terminal": (
                    agent_exit.stopped_after_terminal
                ),
                "start_failed": agent_exit.start_failed,
                "returncode": agent_exit.returncode,
            },
            "terminal_reason": result.terminal_reason,
            "final_submission_id": result.final_submission_id,
            "submissions": submissions,
        },
    )


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    except OSError as error:
        raise AgentRunError("Run record could not be committed") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_file(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400)
    except OSError as error:
        raise AgentRunError("Agent instructions could not be retained") from error


def _make_tree_read_only(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        path = Path(directory)
        for name in files:
            os.chmod(path / name, 0o400)
        os.chmod(path, 0o500)


__all__: list[str] = []
