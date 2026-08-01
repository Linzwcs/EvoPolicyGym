"""Coordinate persisted Run events, reports, and the terminal manifest."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import TextIO

from ...benchmark import BenchmarkSpec
from ...program import Program
from ...results import RunResult
from ...skills import AgentSkill
from .. import RunConfig
from .._agent import AgentOutcome
from .._workspace import RunDirectoryPaths
from ..progress import RunObserver
from .events import append_event
from .manifest import write_run_manifest
from .reports import write_assessment_report, write_validation_report


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
        skills: tuple[AgentSkill, ...] = (),
        observer: RunObserver | None = None,
    ) -> None:
        if type(benchmark_spec) is not BenchmarkSpec:
            raise TypeError("benchmark_spec must be BenchmarkSpec")
        self._paths = paths
        self._benchmark_spec = benchmark_spec
        self._initial_program = initial_program
        self._config = config
        self._agent_identity = dict(agent_identity)
        self._skills = skills
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

    def commit(self, result: RunResult, agent_outcome: AgentOutcome) -> None:
        if result.validation is not None:
            write_validation_report(
                self._paths.validation,
                result.validation,
            )
        if result.assessment is not None:
            write_assessment_report(
                self._paths.assessment,
                result.assessment,
            )
        write_run_manifest(
            self._paths.root / "run.json",
            result,
            benchmark_spec=self._benchmark_spec,
            initial_program=self._initial_program,
            config=self._config,
            skills=self._skills,
            agent_outcome=agent_outcome,
            agent_identity=self._agent_identity,
        )

__all__: list[str] = []
