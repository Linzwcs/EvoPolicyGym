"""Run workspace layout, retained invocation, events, and terminal record."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TextIO, cast

from .._version import __version__
from ..agents import AgentInvocation
from ..benchmark import BenchmarkSpec
from ..errors import AgentRunError
from ..execution.process.agent.runner import AgentExit
from ..program import Program
from ..results import RunResult
from . import RunConfig
from ._json import encode_public_json_value
from .progress import RunEvent, RunEventValue, RunObserver

_RUN_RECORD_SCHEMA = "evopolicygym/run-record/v5"
_RUN_EVENT_SCHEMA = "evopolicygym/run-event/v1"
_INVOCATION_SCHEMA = "evopolicygym/agent-invocation/v1"
_VALIDATION_REPORT_SCHEMA = "evopolicygym/validation-report/v2"
_ASSESSMENT_REPORT_SCHEMA = "evopolicygym/assessment-report/v2"
_SESSION_SOCKET_VARIABLE = "EVOPOLICYGYM_SESSION_SOCKET"
_WORKSPACE_VARIABLE = "EVOPOLICYGYM_WORKSPACE"


@dataclass(frozen=True, slots=True)
class RunDirectoryPaths:
    root: Path
    workspace: Path
    skill: Path
    program: Path
    analysis: Path
    feedback: Path
    initial: Path
    submissions: Path
    agent: Path
    validation: Path
    assessment: Path
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
            skill=workspace / "skill",
            program=workspace / "program",
            analysis=workspace / "analysis",
            feedback=workspace / "feedback",
            initial=root / "initial",
            submissions=root / "submissions",
            agent=root / "agent",
            validation=root / "validation",
            assessment=root / "assessment",
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
        benchmark_spec: BenchmarkSpec,
        initial_program: Program,
        config: RunConfig,
        agent_identity: Mapping[str, str],
        observer: RunObserver | None = None,
    ) -> None:
        if type(benchmark_spec) is not BenchmarkSpec:
            raise TypeError("benchmark_spec must be BenchmarkSpec")
        self._paths = paths
        self._benchmark_spec = benchmark_spec
        self._initial_program = initial_program
        self._config = config
        self._agent_identity = dict(agent_identity)
        self._observer = observer
        self._events: TextIO | None = None
        self._event_lock = Lock()

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
        with self._event_lock:
            events = self._events
            if events is None:
                raise RuntimeError("Run recorder is not open")
            published = append_event(events, event, fields)
            observer = self._observer
            if observer is not None:
                try:
                    observer.on_event(published)
                except Exception:
                    self._observer = None

    def commit(self, result: RunResult, agent_exit: AgentExit) -> None:
        if result.validation is not None:
            _write_validation_report(
                self._paths.validation,
                result,
            )
        if result.assessment is not None:
            _write_assessment_report(
                self._paths.assessment,
                result,
            )
        _write_run_manifest(
            self._paths.root / "run.json",
            result,
            benchmark_spec=self._benchmark_spec,
            initial_program=self._initial_program,
            config=self._config,
            agent_exit=agent_exit,
            agent_identity=self._agent_identity,
        )


def prepare_run_directory(
    root: Path,
    initial_program: Program,
    *,
    agent_skill: str | None = None,
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
        paths.analysis,
        paths.feedback,
        paths.initial,
        paths.submissions,
        paths.agent,
        paths.control,
    ):
        directory.mkdir(mode=0o700)
    initial_program.write_to(paths.initial / "program")
    initial_program.write_to(paths.program)
    if agent_skill is not None:
        paths.skill.mkdir(mode=0o700)
        _write_read_only_text_file(
            paths.skill / "SKILL.md",
            agent_skill,
            error_message="Benchmark Agent skill could not be retained",
        )
        os.chmod(paths.skill, 0o500)
    _make_tree_read_only(paths.initial / "program")
    return paths


def retain_agent_invocation(
    paths: RunDirectoryPaths,
    invocation: AgentInvocation,
) -> None:
    if invocation.instructions is not None:
        _write_read_only_text_file(
            paths.agent / "instructions.md",
            invocation.instructions,
            error_message="Agent instructions could not be retained",
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
) -> RunEvent:
    normalized: dict[str, RunEventValue] = {}
    for name, value in fields.items():
        if type(value) not in {str, int, float, bool, type(None)}:
            raise TypeError("Run event fields must contain JSON scalar values")
        normalized[name] = cast(RunEventValue, value)
    published = RunEvent(
        name=event,
        time_unix_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        fields=normalized,
    )
    document = {
        "schema": _RUN_EVENT_SCHEMA,
        "time_unix_ns": published.time_unix_ns,
        "monotonic_ns": published.monotonic_ns,
        "event": published.name,
        **published.fields,
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
    return published


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
    benchmark_spec: BenchmarkSpec,
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
            "benchmark": {
                "id": benchmark_spec.id,
                "environment": {
                    "digest": benchmark_spec.environment_digest,
                    "parameters": encode_public_json_value(
                        benchmark_spec.environment_parameters
                    ),
                },
            },
            "initial_program": {
                "digest": initial_program.digest,
                "record": "initial/program",
            },
            "workspace": {
                "root": "workspace",
                "program": "workspace/program",
                "analysis": "workspace/analysis",
                "feedback": "workspace/feedback",
                **(
                    {"skill": "workspace/skill/SKILL.md"}
                    if (path.parent / "workspace" / "skill" / "SKILL.md").is_file()
                    else {}
                ),
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
                "bulk_feedback_retention_bytes": (
                    config.bulk_feedback_retention_bytes
                ),
                "use_benchmark_skill": config.use_benchmark_skill,
                "episode_timeout_seconds": config.episode_timeout_seconds,
                "agent_timeout_seconds": config.agent_timeout_seconds,
                "validation": (
                    None
                    if config.validation is None
                    else {
                        "split": config.validation.split,
                        "episodes_per_candidate": (
                            config.validation.episodes_per_candidate
                        ),
                        "max_candidates": (
                            config.validation.max_candidates
                        ),
                    }
                ),
                "assessment": (
                    None
                    if config.assessment is None
                    else {
                        "split": config.assessment.split,
                        "episodes": config.assessment.episodes,
                    }
                ),
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
            "candidate_submission_ids": list(
                result.candidate_submission_ids
            ),
            "validation": (
                None
                if result.validation is None
                else {"report": "validation/report.json"}
            ),
            "assessment": (
                None
                if result.assessment is None
                else {"report": "assessment/report.json"}
            ),
            "submissions": submissions,
        },
    )


def _write_validation_report(
    directory: Path,
    result: RunResult,
) -> None:
    validation = result.validation
    assert validation is not None
    try:
        directory.mkdir(mode=0o700)
    except OSError as error:
        raise AgentRunError(
            "Validation record could not be committed"
        ) from error
    _write_json_atomic(
        directory / "report.json",
        {
            "schema": _VALIDATION_REPORT_SCHEMA,
            "split": validation.split,
            "episodes_per_candidate": (
                validation.episodes_per_candidate
            ),
            "primary_metric": validation.primary_metric,
            "score_direction": validation.score_direction,
            "candidates": [
                {
                    "submission_id": candidate.submission_id,
                    "program_digest": candidate.program_digest,
                    "score": candidate.score,
                    "episodes": candidate.episodes,
                    "policy_failures": candidate.policy_failures,
                    "feedback_content": encode_public_json_value(
                        candidate.feedback_content
                    ),
                }
                for candidate in validation.candidates
            ],
            "selected_submission_id": (
                validation.selected_submission_id
            ),
        },
        error_message="Validation record could not be committed",
    )
    try:
        os.chmod(directory, 0o500)
    except OSError as error:
        raise AgentRunError(
            "Validation record could not be committed"
        ) from error


def _write_assessment_report(
    directory: Path,
    result: RunResult,
) -> None:
    assessment = result.assessment
    assert assessment is not None
    try:
        directory.mkdir(mode=0o700)
    except OSError as error:
        raise AgentRunError(
            "Assessment record could not be committed"
        ) from error
    _write_json_atomic(
        directory / "report.json",
        {
            "schema": _ASSESSMENT_REPORT_SCHEMA,
            "submission_id": assessment.submission_id,
            "program_digest": assessment.program_digest,
            "split": assessment.split,
            "episodes": assessment.episodes,
            "primary_metric": assessment.primary_metric,
            "score_direction": assessment.score_direction,
            "score": assessment.score,
            "policy_failures": assessment.policy_failures,
            "feedback_content": encode_public_json_value(
                assessment.feedback_content
            ),
        },
        error_message="Assessment record could not be committed",
    )
    try:
        os.chmod(directory, 0o500)
    except OSError as error:
        raise AgentRunError(
            "Assessment record could not be committed"
        ) from error


def _write_json_atomic(
    path: Path,
    document: dict[str, object],
    *,
    error_message: str = "Run record could not be committed",
) -> None:
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
        raise AgentRunError(error_message) from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_read_only_text_file(
    path: Path,
    content: str,
    *,
    error_message: str,
) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400)
    except OSError as error:
        raise AgentRunError(error_message) from error


def _make_tree_read_only(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        path = Path(directory)
        for name in files:
            os.chmod(path / name, 0o400)
        os.chmod(path, 0o500)


__all__: list[str] = []
