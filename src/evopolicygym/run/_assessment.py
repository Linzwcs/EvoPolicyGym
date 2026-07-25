"""Held-out final-Program assessment after candidate selection."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Protocol

from ..benchmark import Benchmark, BenchmarkSpec
from ..errors import EvaluationError
from ..evaluation import EvaluationConfig
from ..program import Program
from ..results import (
    AssessmentResult,
    EpisodeSummary,
    EvaluationResult,
    SubmissionResult,
)
from . import RunConfig

_ASSESSMENT_SEED_DOMAIN = b"evopolicygym/assessment-seed/v1\0"


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


class ProgramAssessor:
    """Evaluate the selected Program once without influencing selection."""

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

    def assess(
        self,
        submission: SubmissionResult,
    ) -> AssessmentResult | None:
        assessment_config = self._config.assessment
        if assessment_config is None:
            return None

        identifier = submission.submission_id
        self._recorder.record_event(
            "assessment_started",
            {
                "submission_id": identifier,
                "split": assessment_config.split,
                "episodes": assessment_config.episodes,
                "primary_metric": self._spec.primary_metric,
                "score_direction": self._spec.score_direction,
            },
        )

        def episode_completed(
            completed: int,
            total: int,
            summary: EpisodeSummary,
        ) -> None:
            self._recorder.record_event(
                "assessment_episode_completed",
                {
                    "submission_id": identifier,
                    "completed": completed,
                    "total": total,
                    "status": summary.status,
                },
            )

        try:
            evaluated = self._evaluator.evaluate(
                submission.program,
                self._benchmark,
                EvaluationConfig(
                    split=assessment_config.split,
                    episodes=assessment_config.episodes,
                    seed=_assessment_seed(self._config.seed),
                    episode_timeout_seconds=(
                        self._config.episode_timeout_seconds
                    ),
                ),
                episode_completed=episode_completed,
            )
        except EvaluationError:
            raise
        except Exception:
            raise EvaluationError(
                "trusted Assessment evaluation failed"
            ) from None
        if (
            type(evaluated) is not EvaluationResult
            or evaluated.benchmark_id != self._spec.id
            or evaluated.environment_digest
            != self._spec.environment_digest
            or evaluated.program_digest != submission.program_digest
            or len(evaluated.episodes) != assessment_config.episodes
        ):
            raise EvaluationError(
                "Assessment evaluation returned an invalid result"
            )

        result = AssessmentResult(
            submission_id=identifier,
            program_digest=submission.program_digest,
            split=assessment_config.split,
            episodes=len(evaluated.episodes),
            primary_metric=self._spec.primary_metric,
            score_direction=self._spec.score_direction,
            score=evaluated.feedback.score,
            policy_failures=sum(
                episode.status == "policy_failed"
                for episode in evaluated.episodes
            ),
        )
        self._recorder.record_event(
            "assessment_completed",
            {
                "submission_id": identifier,
                "score": result.score,
                "policy_failures": result.policy_failures,
            },
        )
        return result


def _assessment_seed(run_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(_ASSESSMENT_SEED_DOMAIN)
    digest.update(run_seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


__all__: list[str] = []
