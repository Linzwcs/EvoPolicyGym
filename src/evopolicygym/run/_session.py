"""Submission accounting, receipts, and final-selection rules."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ..benchmark import Benchmark
from ..errors import EvaluationError, ProgramError
from ..evaluation import EvaluationConfig
from ..program import Program
from ..results import (
    EpisodeSummary,
    EvaluationResult,
    RunTerminalReason,
    SubmissionResult,
)
from . import RunConfig

_SUBMISSION_SEED_DOMAIN = b"evopolicygym/submission-seed/v1\0"


@dataclass(frozen=True, slots=True)
class SessionError:
    """Sanitized rejection returned to one Coding Agent request."""

    code: str
    message: str

    def __post_init__(self) -> None:
        for name in ("code", "message"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Agent-visible receipt for one committed Submission."""

    submission_id: str
    program_digest: str
    score: float
    episodes_used: int
    episodes_remaining: int


@dataclass(frozen=True, slots=True)
class FinishReceipt:
    """Agent-visible receipt selecting the final Program."""

    submission_id: str
    program_digest: str


type SubmissionOutcome = SubmissionReceipt | SessionError
type FinishOutcome = FinishReceipt | SessionError


class ProgramSource(Protocol):
    def capture(self) -> Program:
        ...


class ProgramEvaluator(Protocol):
    def evaluate(
        self,
        program: Program,
        benchmark: Benchmark,
        config: EvaluationConfig,
        *,
        episode_completed: (
            Callable[[int, int, EpisodeSummary], None] | None
        ) = None,
    ) -> EvaluationResult:
        ...


class SubmissionPublisher(Protocol):
    def commit(self, result: SubmissionResult) -> None:
        ...


class EventRecorder(Protocol):
    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        ...


class SubmissionSession:
    """Single-owner authoritative state for bounded Program submissions."""

    def __init__(
        self,
        *,
        programs: ProgramSource,
        evaluator: ProgramEvaluator,
        publisher: SubmissionPublisher,
        benchmark: Benchmark,
        config: RunConfig,
        recorder: EventRecorder,
    ) -> None:
        self._programs = programs
        self._evaluator = evaluator
        self._publisher = publisher
        self._benchmark = benchmark
        self._config = config
        self._recorder = recorder
        self._episodes_remaining = config.episode_budget
        self._submissions: list[SubmissionResult] = []
        self._final_submission_id: str | None = None
        self._terminal_reason: RunTerminalReason | None = None

    @property
    def submissions(self) -> tuple[SubmissionResult, ...]:
        return tuple(self._submissions)

    @property
    def final_submission_id(self) -> str | None:
        return self._final_submission_id

    @property
    def final_program(self) -> Program | None:
        identifier = self._final_submission_id
        if identifier is None:
            return None
        return next(
            item.program
            for item in self._submissions
            if item.submission_id == identifier
        )

    @property
    def terminal_reason(self) -> RunTerminalReason | None:
        return self._terminal_reason

    @property
    def authority_exhausted(self) -> bool:
        return (
            self._episodes_remaining == 0
            or len(self._submissions) >= self._config.max_submissions
        )

    def submit(self, episodes: object) -> SubmissionOutcome:
        if self._terminal_reason is not None:
            return _error("session_closed", "the Agent Session is already closed")
        if type(episodes) is not int or episodes <= 0:
            return _error("invalid_request", "episodes must be a positive integer")
        if len(self._submissions) >= self._config.max_submissions:
            return _error("submission_limit", "the submission limit is exhausted")
        submission_limit = self._config.max_episodes_per_submission
        if submission_limit is not None and episodes > submission_limit:
            return _error(
                "episode_limit",
                "episodes exceeds max_episodes_per_submission",
            )
        if episodes > self._episodes_remaining:
            return _error("budget_exhausted", "insufficient Episode budget")

        try:
            program = self._programs.capture()
        except ProgramError:
            self._recorder.record_event(
                "submission_rejected",
                {"reason": "program_invalid"},
            )
            return _error(
                "program_invalid",
                "the workspace Policy could not be captured",
            )

        ordinal = len(self._submissions) + 1
        submission_id = f"submission-{ordinal:06d}"
        self._episodes_remaining -= episodes
        self._recorder.record_event(
            "evaluation_started",
            {
                "submission_id": submission_id,
                "program_digest": program.digest,
                "episodes": episodes,
                "episodes_remaining": self._episodes_remaining,
            },
        )
        try:
            def episode_completed(
                completed: int,
                total: int,
                summary: EpisodeSummary,
            ) -> None:
                self._recorder.record_event(
                    "episode_completed",
                    {
                        "submission_id": submission_id,
                        "completed": completed,
                        "total": total,
                        "status": summary.status,
                    },
                )

            evaluation = self._evaluator.evaluate(
                program,
                self._benchmark,
                EvaluationConfig(
                    split=self._config.split,
                    episodes=episodes,
                    seed=_submission_seed(self._config.seed, ordinal),
                    episode_timeout_seconds=self._config.episode_timeout_seconds,
                ),
                episode_completed=episode_completed,
            )
        except EvaluationError:
            self._terminal_reason = "evaluation_failed"
            self._recorder.record_event(
                "evaluation_failed",
                {
                    "submission_id": submission_id,
                    "episodes_remaining": self._episodes_remaining,
                },
            )
            return _error(
                "evaluation_failed",
                "trusted evaluation failed; the reserved budget was consumed",
            )

        result = SubmissionResult(
            submission_id=submission_id,
            program=program,
            episodes_used=episodes,
            episodes_remaining=self._episodes_remaining,
            feedback=evaluation.feedback,
            episodes=evaluation.episodes,
        )
        try:
            self._publisher.commit(result)
        except Exception:
            self._terminal_reason = "evaluation_failed"
            self._recorder.record_event(
                "publication_failed",
                {
                    "submission_id": submission_id,
                    "episodes_remaining": self._episodes_remaining,
                },
            )
            return _error(
                "publication_failed",
                "public Feedback could not be committed",
            )

        self._submissions.append(result)
        self._recorder.record_event(
            "submission_published",
            {
                "submission_id": submission_id,
                "program_digest": program.digest,
                "score": result.feedback.score,
                "episodes_remaining": self._episodes_remaining,
            },
        )
        return SubmissionReceipt(
            submission_id=submission_id,
            program_digest=program.digest,
            score=result.feedback.score,
            episodes_used=episodes,
            episodes_remaining=self._episodes_remaining,
        )

    def finish(self, submission_id: object) -> FinishOutcome:
        if self._terminal_reason is not None:
            return _error("session_closed", "the Agent Session is already closed")
        selected = tuple(
            item
            for item in self._submissions
            if item.submission_id == submission_id
        )
        if type(submission_id) is not str or len(selected) != 1:
            return _error(
                "unknown_submission",
                "finish must select a published submission",
            )
        assert isinstance(submission_id, str)
        self._final_submission_id = submission_id
        self._terminal_reason = "finished"
        program = selected[0].program
        self._recorder.record_event(
            "run_finished",
            {
                "submission_id": submission_id,
                "program_digest": program.digest,
            },
        )
        return FinishReceipt(
            submission_id=submission_id,
            program_digest=program.digest,
        )

    def fail(self) -> None:
        """Close admission after an unexpected Host-side gateway fault."""

        if self._terminal_reason is None:
            self._terminal_reason = "evaluation_failed"


def _submission_seed(run_seed: int, ordinal: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SUBMISSION_SEED_DOMAIN)
    digest.update(run_seed.to_bytes(8, "big"))
    digest.update(ordinal.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _error(code: str, message: str) -> SessionError:
    return SessionError(code=code, message=message)


__all__: list[str] = []
