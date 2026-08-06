"""Submission accounting and atomic candidate handoff."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from ..._protocol.session import SESSION_MAX_EPISODE_INDICES
from ...benchmark import Benchmark, BenchmarkSpec
from ...errors import EvaluationError, ProgramError
from ...evaluation._inputs import EpisodeInput
from ...program import Program
from ...results import (
    EpisodeSummary,
    EvaluationResult,
    RunTerminalReason,
    SubmissionResult,
)
from .. import RunConfig
from .outcomes import (
    FinishOutcome,
    FinishReceipt,
    SessionError,
    SubmissionOutcome,
    SubmissionReceipt,
)


class ProgramSource(Protocol):
    def capture(self) -> Program:
        ...


class ProgramEvaluator(Protocol):
    def evaluate_episodes(
        self,
        program: Program,
        benchmark: Benchmark,
        episode_inputs: tuple[EpisodeInput, ...],
        *,
        episode_timeout_seconds: float,
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
        spec: BenchmarkSpec,
        config: RunConfig,
        recorder: EventRecorder,
        episode_pool: tuple[EpisodeInput, ...],
    ) -> None:
        if type(spec) is not BenchmarkSpec:
            raise TypeError("spec must be BenchmarkSpec")
        if (
            type(episode_pool) is not tuple
            or not episode_pool
            or any(
                type(episode) is not EpisodeInput
                for episode in episode_pool
            )
        ):
            raise TypeError(
                "episode_pool must be a non-empty tuple of EpisodeInput values"
            )
        if len(episode_pool) != config.episode_pool_size:
            raise ValueError(
                "episode_pool must match config.episode_pool_size"
            )
        self._programs = programs
        self._evaluator = evaluator
        self._publisher = publisher
        self._benchmark = benchmark
        self._spec = spec
        self._config = config
        self._recorder = recorder
        self._episode_pool = episode_pool
        self._submission_episode_limit = min(
            SESSION_MAX_EPISODE_INDICES,
            len(episode_pool),
            (
                SESSION_MAX_EPISODE_INDICES
                if config.max_episodes_per_submission is None
                else config.max_episodes_per_submission
            ),
        )
        if (
            config.finish_budget_policy == "require_budget_exhaustion"
            and config.episode_budget
            > config.max_submissions * self._submission_episode_limit
        ):
            raise ValueError(
                "finish_budget_policy requires an Episode budget that can be "
                "exhausted within max_submissions"
            )
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

    def submit(self, episode_indices: object) -> SubmissionOutcome:
        if self.agent_authority_closed:
            return _error("session_closed", "the Agent Session is already closed")
        if type(episode_indices) is not list or not episode_indices:
            return _error(
                "invalid_request",
                "episode_indices must be a non-empty list",
            )
        if any(
            type(index) is not int
            or not 0 <= index < len(self._episode_pool)
            for index in episode_indices
        ):
            return _error(
                "invalid_request",
                "episode_indices contains an invalid pool index",
            )
        selected_indices = tuple(episode_indices)
        if any(
            previous >= current
            for previous, current in zip(
                selected_indices,
                selected_indices[1:],
                strict=False,
            )
        ):
            return _error(
                "invalid_request",
                "episode_indices must be strictly increasing",
            )
        episodes = len(selected_indices)
        if episodes > SESSION_MAX_EPISODE_INDICES:
            return _error(
                "episode_limit",
                "episode_indices exceeds the Session protocol limit",
            )
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
        if self._config.finish_budget_policy == "require_budget_exhaustion":
            remaining_after = self._episodes_remaining - episodes
            submission_slots_after = (
                self._config.max_submissions - len(self._submissions) - 1
            )
            future_capacity = (
                submission_slots_after * self._submission_episode_limit
            )
            if remaining_after > future_capacity:
                minimum_episodes = self._episodes_remaining - future_capacity
                return _error(
                    "budget_allocation",
                    "this submission is too small to exhaust the Episode "
                    "budget within the remaining submission limit; select at "
                    f"least {minimum_episodes} Episodes",
                )

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
                "episode_indices": _render_indices(selected_indices),
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
                        "episode_index": selected_indices[completed - 1],
                        "status": summary.status,
                    },
                )

            evaluation = self._evaluator.evaluate_episodes(
                program,
                self._benchmark,
                tuple(
                    self._episode_pool[index]
                    for index in selected_indices
                ),
                episode_timeout_seconds=self._config.episode_timeout_seconds,
                episode_completed=episode_completed,
            )
            if (
                type(evaluation) is not EvaluationResult
                or evaluation.benchmark_id != self._spec.id
                or evaluation.environment_digest
                != self._spec.environment_digest
                or evaluation.program_digest != program.digest
                or len(evaluation.episodes) != episodes
            ):
                raise EvaluationError(
                    "Evaluation returned a mismatched result"
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
            episode_indices=selected_indices,
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
            episode_indices=selected_indices,
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
        if (
            self._config.finish_budget_policy
            == "require_budget_exhaustion"
            and self._episodes_remaining > 0
        ):
            remaining_message = (
                "1 Episode budget unit remains."
                if self._episodes_remaining == 1
                else (
                    f"{self._episodes_remaining} Episode budget units remain."
                )
            )
            return self._reject_finish(
                "budget_remaining",
                f"finish is not available: {remaining_message} "
                "Continue evaluating or confirming candidate Programs, then "
                "retry finish.",
                candidate_count=len(identifiers),
                episodes_remaining=self._episodes_remaining,
            )

        self._recorder.record_event(
            "finish_requested",
            {
                "candidate_count": len(identifiers),
                "episodes_remaining": self._episodes_remaining,
            },
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
        episodes_remaining: int | None = None,
    ) -> SessionError:
        fields: dict[str, object] = {"reason": code}
        if candidate_count is not None:
            fields["candidate_count"] = candidate_count
        if episodes_remaining is not None:
            fields["episodes_remaining"] = episodes_remaining
        self._recorder.record_event("finish_rejected", fields)
        return _error(code, message)


def _render_indices(indices: tuple[int, ...]) -> str:
    return ",".join(str(index) for index in indices)


def _error(code: str, message: str) -> SessionError:
    return SessionError(code=code, message=message)


__all__: list[str] = []
