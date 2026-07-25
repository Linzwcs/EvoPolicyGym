"""Submission accounting, receipts, and atomic candidate handoff."""

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
    """Agent-visible receipt transferring candidate selection to the Host."""

    candidate_submission_ids: tuple[str, ...]


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
        self._candidate_submission_ids: tuple[str, ...] | None = None
        self._terminal_reason: RunTerminalReason | None = None

    @property
    def submissions(self) -> tuple[SubmissionResult, ...]:
        return tuple(self._submissions)

    @property
    def candidate_submission_ids(self) -> tuple[str, ...]:
        return self._candidate_submission_ids or ()

    @property
    def terminal_reason(self) -> RunTerminalReason | None:
        return self._terminal_reason

    @property
    def agent_authority_closed(self) -> bool:
        return (
            self._candidate_submission_ids is not None
            or self._terminal_reason is not None
        )

    @property
    def authority_exhausted(self) -> bool:
        return (
            self._episodes_remaining == 0
            or len(self._submissions) >= self._config.max_submissions
        )

    def submit(self, episodes: object) -> SubmissionOutcome:
        if self.agent_authority_closed:
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

    def finish(self, submission_ids: object) -> FinishOutcome:
        if self.agent_authority_closed:
            return _error("session_closed", "the Agent Session is already closed")

        if type(submission_ids) is not list or not submission_ids:
            return self._reject_finish(
                "invalid_request",
                "finish requires a non-empty submission_ids list",
            )
        if any(
            type(submission_id) is not str or not submission_id
            for submission_id in submission_ids
        ):
            return self._reject_finish(
                "invalid_request",
                "submission_ids must contain non-empty text",
                candidate_count=len(submission_ids),
            )

        identifiers = tuple(submission_ids)
        candidate_limit = (
            1
            if self._config.validation is None
            else self._config.validation.max_candidates
        )
        if len(identifiers) > candidate_limit:
            return self._reject_finish(
                "candidate_limit",
                "finish exceeds the candidate limit",
                candidate_count=len(identifiers),
            )
        if len(identifiers) != len(set(identifiers)):
            return self._reject_finish(
                "duplicate_submission",
                "finish candidates must be unique",
                candidate_count=len(identifiers),
            )

        published = {item.submission_id for item in self._submissions}
        if any(identifier not in published for identifier in identifiers):
            return self._reject_finish(
                "unknown_submission",
                "finish candidates must be published submissions",
                candidate_count=len(identifiers),
            )

        self._recorder.record_event(
            "finish_requested",
            {"candidate_count": len(identifiers)},
        )
        self._candidate_submission_ids = identifiers
        return FinishReceipt(candidate_submission_ids=identifiers)

    def fail(self) -> None:
        """Close admission after an unexpected Host-side gateway fault."""

        if not self.agent_authority_closed:
            self._terminal_reason = "evaluation_failed"

    def _reject_finish(
        self,
        code: str,
        message: str,
        *,
        candidate_count: int | None = None,
    ) -> SessionError:
        fields: dict[str, object] = {"reason": code}
        if candidate_count is not None:
            fields["candidate_count"] = candidate_count
        self._recorder.record_event("finish_rejected", fields)
        return _error(code, message)


def _submission_seed(run_seed: int, ordinal: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SUBMISSION_SEED_DOMAIN)
    digest.update(run_seed.to_bytes(8, "big"))
    digest.update(ordinal.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _error(code: str, message: str) -> SessionError:
    return SessionError(code=code, message=message)


__all__: list[str] = []
