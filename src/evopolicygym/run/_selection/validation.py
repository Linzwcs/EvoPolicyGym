"""Host-side candidate validation after Coding Agent authority has closed."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ...benchmark import Benchmark, BenchmarkSpec
from ...errors import EvaluationError
from ...evaluation import EvaluationConfig
from ...program import Program
from ...results import (
    EpisodeSummary,
    EvaluationResult,
    SubmissionResult,
    ValidationCandidateResult,
    ValidationResult,
)
from .. import RunConfig

_VALIDATION_SEED_DOMAIN = b"evopolicygym/validation-seed/v1\0"


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


class EventRecorder(Protocol):
    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """Internal detached selection returned to the Run coordinator."""

    submission: SubmissionResult
    validation: ValidationResult | None


class CandidateSelector:
    """Evaluate an ordered candidate set and select it deterministically."""

    def __init__(
        self,
        *,
        evaluator: ProgramEvaluator,
        benchmark: Benchmark,
        spec: BenchmarkSpec,
        config: RunConfig,
        recorder: EventRecorder,
    ) -> None:
        self._evaluator = evaluator
        self._benchmark = benchmark
        self._spec = spec
        self._config = config
        self._recorder = recorder

    def select(
        self,
        submissions: tuple[SubmissionResult, ...],
        candidate_submission_ids: tuple[str, ...],
    ) -> CandidateSelection:
        candidates = _resolve_candidates(
            submissions,
            candidate_submission_ids,
        )
        validation_config = self._config.validation
        if validation_config is None:
            if len(candidates) != 1:
                raise EvaluationError(
                    "finish requires exactly one candidate without Validation"
                )
            return CandidateSelection(
                submission=candidates[0],
                validation=None,
            )

        evaluation_config = EvaluationConfig(
            split=validation_config.split,
            episodes=validation_config.episodes_per_candidate,
            seed=_validation_seed(self._config.seed),
            episode_timeout_seconds=self._config.episode_timeout_seconds,
        )
        self._recorder.record_event(
            "validation_started",
            {
                "split": validation_config.split,
                "candidate_count": len(candidates),
                "episodes_per_candidate": (
                    validation_config.episodes_per_candidate
                ),
                "primary_metric": self._spec.primary_metric,
                "score_direction": self._spec.score_direction,
            },
        )

        aggregate: list[ValidationCandidateResult] = []
        for candidate in candidates:
            identifier = candidate.submission_id
            self._recorder.record_event(
                "validation_candidate_started",
                {
                    "submission_id": identifier,
                    "episodes": validation_config.episodes_per_candidate,
                },
            )

            def episode_completed(
                completed: int,
                total: int,
                summary: EpisodeSummary,
                *,
                submission_id: str = identifier,
            ) -> None:
                self._recorder.record_event(
                    "validation_episode_completed",
                    {
                        "submission_id": submission_id,
                        "completed": completed,
                        "total": total,
                        "status": summary.status,
                    },
                )

            try:
                evaluated = self._evaluator.evaluate(
                    candidate.program,
                    self._benchmark,
                    evaluation_config,
                    episode_completed=episode_completed,
                )
            except EvaluationError:
                raise
            except Exception:
                raise EvaluationError(
                    "trusted Validation evaluation failed"
                ) from None
            if (
                type(evaluated) is not EvaluationResult
                or evaluated.benchmark_id != self._spec.id
                or evaluated.environment_digest
                != self._spec.environment_digest
                or evaluated.program_digest != candidate.program_digest
                or len(evaluated.episodes)
                != validation_config.episodes_per_candidate
            ):
                raise EvaluationError(
                    "Validation evaluation returned an invalid result"
                )

            policy_failures = sum(
                episode.status == "policy_failed"
                for episode in evaluated.episodes
            )
            result = ValidationCandidateResult(
                submission_id=identifier,
                program_digest=candidate.program_digest,
                score=evaluated.feedback.score,
                episodes=len(evaluated.episodes),
                policy_failures=policy_failures,
                feedback_content=evaluated.feedback.content,
            )
            aggregate.append(result)
            self._recorder.record_event(
                "validation_candidate_completed",
                {
                    "submission_id": identifier,
                    "score": result.score,
                    "policy_failures": result.policy_failures,
                },
            )

        selected_index = min(
            range(len(aggregate)),
            key=lambda index: _selection_key(
                aggregate[index],
                index=index,
                score_direction=self._spec.score_direction,
            ),
        )
        selected = candidates[selected_index]
        validation = ValidationResult(
            split=validation_config.split,
            episodes_per_candidate=validation_config.episodes_per_candidate,
            primary_metric=self._spec.primary_metric,
            score_direction=self._spec.score_direction,
            candidates=tuple(aggregate),
            selected_submission_id=selected.submission_id,
        )
        return CandidateSelection(
            submission=selected,
            validation=validation,
        )


def _resolve_candidates(
    submissions: tuple[SubmissionResult, ...],
    identifiers: tuple[str, ...],
) -> tuple[SubmissionResult, ...]:
    if not identifiers:
        raise EvaluationError("finish did not provide candidates")
    by_identifier = {
        submission.submission_id: submission for submission in submissions
    }
    try:
        return tuple(by_identifier[identifier] for identifier in identifiers)
    except KeyError:
        raise EvaluationError(
            "finish selected an unavailable submission"
        ) from None


def _selection_key(
    candidate: ValidationCandidateResult,
    *,
    index: int,
    score_direction: str,
) -> tuple[float, int, int]:
    score = (
        -candidate.score
        if score_direction == "maximize"
        else candidate.score
    )
    return (score, candidate.policy_failures, index)


def _validation_seed(run_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(_VALIDATION_SEED_DOMAIN)
    digest.update(run_seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


__all__: list[str] = []
