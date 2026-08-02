"""Execution-independent Program Evolution lifecycle coordination."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ..errors import EvaluationError
from ..program import Program
from ..results import (
    AssessmentResult,
    RunResult,
    RunTerminalReason,
    SubmissionResult,
)
from ._agent import AgentOutcome, AgentRunner, TerminalSignal
from ._selection.validation import CandidateSelection


class SessionGateway(Protocol):
    @property
    def terminal(self) -> TerminalSignal:
        ...

    def start(self) -> None:
        ...

    def close(self) -> None:
        ...


class EvolutionSession(Protocol):
    @property
    def submissions(self) -> tuple[SubmissionResult, ...]:
        ...

    @property
    def candidate_submission_ids(self) -> tuple[str, ...]:
        ...

    @property
    def terminal_reason(self) -> RunTerminalReason | None:
        ...

    @property
    def authority_exhausted(self) -> bool:
        ...


class CandidateSelectionService(Protocol):
    def select(
        self,
        submissions: tuple[SubmissionResult, ...],
        candidate_submission_ids: tuple[str, ...],
    ) -> CandidateSelection:
        ...


class FinalAssessmentService(Protocol):
    def assess(
        self,
        submission: SubmissionResult,
    ) -> AssessmentResult | None:
        ...


class RunRecorder(Protocol):
    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        ...

    def commit(self, result: RunResult, agent_outcome: AgentOutcome) -> None:
        ...


class ProgramEvolutionRun:
    """Coordinate one Agent-driven Program improvement lifecycle."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        initial_program: Program,
        session: EvolutionSession,
        gateway: SessionGateway,
        agent_runner: AgentRunner,
        candidate_selector: CandidateSelectionService,
        final_assessor: FinalAssessmentService,
        recorder: RunRecorder,
        agent_timeout_seconds: float,
    ) -> None:
        if type(benchmark_id) is not str or not benchmark_id:
            raise ValueError("benchmark_id must be non-empty text")
        if type(initial_program) is not Program:
            raise TypeError("initial_program must be Program")
        if agent_timeout_seconds <= 0:
            raise ValueError("agent_timeout_seconds must be positive")
        self._benchmark_id = benchmark_id
        self._initial_program = initial_program
        self._session = session
        self._gateway = gateway
        self._agent_runner = agent_runner
        self._candidate_selector = candidate_selector
        self._final_assessor = final_assessor
        self._recorder = recorder
        self._agent_timeout_seconds = agent_timeout_seconds

    def execute(self) -> RunResult:
        """Execute the Run, commit its result, and return a detached value."""

        agent_outcome: AgentOutcome | None = None
        try:
            self._gateway.start()
            self._recorder.record_event(
                "agent_started",
                {
                    "benchmark_id": self._benchmark_id,
                    "initial_program_digest": self._initial_program.digest,
                },
            )
            agent_outcome = self._agent_runner.run(
                self._gateway.terminal,
                timeout_seconds=self._agent_timeout_seconds,
            )
            self._record_agent_outcome(agent_outcome)
        finally:
            self._gateway.close()

        assert agent_outcome is not None
        candidate_ids = self._session.candidate_submission_ids
        selection: CandidateSelection | None = None
        assessment: AssessmentResult | None = None
        terminal_reason = _terminal_reason(self._session, agent_outcome)
        if candidate_ids and self._session.terminal_reason is None:
            try:
                selection = self._candidate_selector.select(
                    self._session.submissions,
                    candidate_ids,
                )
            except EvaluationError:
                terminal_reason = "validation_failed"
                self._recorder.record_event(
                    "validation_failed",
                    {"candidate_count": len(candidate_ids)},
                )
            else:
                self._recorder.record_event(
                    "final_submission_selected",
                    {
                        "submission_id": selection.submission.submission_id,
                        "program_digest": (
                            selection.submission.program_digest
                        ),
                    },
                )
                try:
                    assessment = self._final_assessor.assess(
                        selection.submission
                    )
                except EvaluationError:
                    terminal_reason = "assessment_failed"
                    self._recorder.record_event(
                        "assessment_failed",
                        {
                            "submission_id": (
                                selection.submission.submission_id
                            ),
                        },
                    )
                else:
                    terminal_reason = "finished"
                    self._recorder.record_event(
                        "run_finished",
                        {
                            "submission_id": (
                                selection.submission.submission_id
                            ),
                            "program_digest": (
                                selection.submission.program_digest
                            ),
                        },
                    )
        result = RunResult(
            final_program=(
                None if selection is None else selection.submission.program
            ),
            final_submission_id=(
                None
                if selection is None
                else selection.submission.submission_id
            ),
            submissions=self._session.submissions,
            terminal_reason=terminal_reason,
            candidate_submission_ids=candidate_ids,
            validation=(
                None if selection is None else selection.validation
            ),
            assessment=assessment,
        )
        self._recorder.commit(result, agent_outcome)
        return result

    def _record_agent_outcome(self, agent_outcome: AgentOutcome) -> None:
        if agent_outcome.start_failed:
            fields: dict[str, object] = {}
            if agent_outcome.start_error_type is not None:
                fields["error_type"] = agent_outcome.start_error_type
            if agent_outcome.start_errno is not None:
                fields["errno"] = agent_outcome.start_errno
            self._recorder.record_event("agent_start_failed", fields)
            return
        if agent_outcome.timed_out:
            self._recorder.record_event("agent_timeout", {})
            return
        if agent_outcome.stopped_after_terminal:
            self._recorder.record_event(
                "agent_stopped_after_terminal",
                {"returncode": agent_outcome.returncode},
            )
            return
        self._recorder.record_event(
            "agent_exited",
            {"returncode": agent_outcome.returncode},
        )


def _terminal_reason(
    session: EvolutionSession,
    agent_outcome: AgentOutcome,
) -> RunTerminalReason:
    if session.terminal_reason is not None:
        return session.terminal_reason
    if (
        agent_outcome.timed_out
        or agent_outcome.stopped_after_terminal
        or agent_outcome.start_failed
        or agent_outcome.returncode not in {0, None}
    ):
        return "agent_failed"
    if session.authority_exhausted:
        return "budget_exhausted"
    return "agent_exited"


__all__: list[str] = []
