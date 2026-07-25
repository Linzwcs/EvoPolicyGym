from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from evopolicygym.authoring import (
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
)
from evopolicygym.benchmark import Benchmark
from evopolicygym.errors import EvaluationError
from evopolicygym.evaluation import EvaluationConfig
from evopolicygym.program import Program
from evopolicygym.results import (
    EpisodeSummary,
    EvaluationResult,
    Feedback,
    SubmissionResult,
)
from evopolicygym.run import RunConfig, ValidationConfig
from evopolicygym.run._validation import CandidateSelector


class StubBenchmark:
    def __init__(
        self,
        *,
        score_direction: Literal["maximize", "minimize"] = "maximize",
    ) -> None:
        self._spec = BenchmarkSpec(
            id="example/validation-v1",
            description="Validation selection fixture.",
            observation_space=None,
            action_space=None,
            metadata={},
            max_episode_steps=1,
            primary_metric="reward",
            score_direction=score_direction,
        )

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        del split
        return tuple(
            EpisodeSpec(environment_seed=seed + index)
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        del episode
        raise AssertionError("the fake evaluator owns evaluation")

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        del episodes
        raise AssertionError("the fake evaluator owns evaluation")


class FakeEvaluator:
    def __init__(
        self,
        outcomes: Mapping[str, tuple[float, int]],
        *,
        fail_digest: str | None = None,
        mismatch_environment: bool = False,
    ) -> None:
        self.outcomes = dict(outcomes)
        self.fail_digest = fail_digest
        self.mismatch_environment = mismatch_environment
        self.configs: list[EvaluationConfig] = []

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
        self.configs.append(config)
        if program.digest == self.fail_digest:
            raise EvaluationError("trusted fixture failure")
        score, failures = self.outcomes[program.digest]
        episodes = tuple(
            EpisodeSummary(
                status="policy_failed",
                reward=None,
                steps=0,
                failure="exception",
            )
            if index < failures
            else EpisodeSummary(
                status="completed",
                reward=score,
                steps=1,
            )
            for index in range(config.episodes)
        )
        if episode_completed is not None:
            for index, episode in enumerate(episodes, start=1):
                episode_completed(index, len(episodes), episode)
        return EvaluationResult(
            benchmark_id=benchmark.spec.id,
            environment_digest=(
                "sha256:" + "0" * 64
                if self.mismatch_environment
                else benchmark.spec.environment_digest
            ),
            program_digest=program.digest,
            feedback=Feedback(score=score),
            episodes=episodes,
        )


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        self.events.append((event, dict(fields)))


def make_submission(
    root: Path,
    ordinal: int,
) -> SubmissionResult:
    source = root / f"program-{ordinal}"
    source.mkdir()
    (source / "policy.py").write_text(
        f"VALUE = {ordinal}\n"
        "def make_policy(context):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        submission_id=f"submission-{ordinal:06d}",
        program=Program.from_directory(source),
        episodes_used=1,
        episodes_remaining=10 - ordinal,
        feedback=Feedback(score=float(ordinal)),
        episodes=(
            EpisodeSummary(
                status="completed",
                reward=float(ordinal),
                steps=1,
            ),
        ),
    )


class CandidateSelectorTests(unittest.TestCase):
    def test_primary_score_outranks_policy_failure_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = tuple(
                make_submission(root, ordinal)
                for ordinal in range(1, 5)
            )
            ordered = (
                submissions[1],
                submissions[0],
                submissions[2],
                submissions[3],
            )
            outcomes = {
                ordered[0].program_digest: (7.0, 0),
                ordered[1].program_digest: (7.0, 0),
                ordered[2].program_digest: (7.0, 1),
                ordered[3].program_digest: (8.0, 2),
            }
            evaluator = FakeEvaluator(outcomes)
            recorder = FakeRecorder()
            benchmark = StubBenchmark()
            selector = CandidateSelector(
                evaluator=evaluator,
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    max_submissions=4,
                    validation=ValidationConfig(
                        episodes_per_candidate=3,
                        max_candidates=4,
                    ),
                    seed=41,
                ),
                recorder=recorder,
            )

            selection = selector.select(
                submissions,
                tuple(item.submission_id for item in ordered),
            )

        self.assertEqual(
            selection.submission.submission_id,
            ordered[3].submission_id,
        )
        self.assertIsNotNone(selection.validation)
        assert selection.validation is not None
        self.assertEqual(
            tuple(
                item.submission_id
                for item in selection.validation.candidates
            ),
            tuple(item.submission_id for item in ordered),
        )
        self.assertEqual(len(evaluator.configs), 4)
        self.assertTrue(
            all(config == evaluator.configs[0] for config in evaluator.configs)
        )
        self.assertEqual(evaluator.configs[0].split, "validation")
        self.assertEqual(evaluator.configs[0].episodes, 3)
        self.assertEqual(
            [name for name, _ in recorder.events].count(
                "validation_episode_completed"
            ),
            12,
        )

    def test_fewer_policy_failures_break_a_score_tie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_submission(root, 1)
            second = make_submission(root, 2)
            benchmark = StubBenchmark()
            selector = CandidateSelector(
                evaluator=FakeEvaluator(
                    {
                        first.program_digest: (4.0, 1),
                        second.program_digest: (4.0, 0),
                    }
                ),
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    max_submissions=2,
                    validation=ValidationConfig(
                        episodes_per_candidate=2,
                        max_candidates=2,
                    ),
                ),
                recorder=FakeRecorder(),
            )

            selection = selector.select(
                (first, second),
                (first.submission_id, second.submission_id),
            )

        self.assertEqual(
            selection.submission.submission_id,
            second.submission_id,
        )

    def test_minimize_and_finish_order_break_exact_ties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_submission(root, 1)
            second = make_submission(root, 2)
            third = make_submission(root, 3)
            outcomes = {
                first.program_digest: (2.0, 0),
                second.program_digest: (1.0, 1),
                third.program_digest: (1.0, 1),
            }
            evaluator = FakeEvaluator(outcomes)
            recorder = FakeRecorder()
            benchmark = StubBenchmark(score_direction="minimize")
            selector = CandidateSelector(
                evaluator=evaluator,
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    max_submissions=3,
                    validation=ValidationConfig(
                        episodes_per_candidate=2,
                        max_candidates=3,
                    ),
                ),
                recorder=recorder,
            )

            selection = selector.select(
                (first, second, third),
                (
                    third.submission_id,
                    second.submission_id,
                    first.submission_id,
                ),
            )

        self.assertEqual(
            selection.submission.submission_id,
            third.submission_id,
        )

    def test_validation_failure_does_not_return_a_partial_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_submission(root, 1)
            second = make_submission(root, 2)
            benchmark = StubBenchmark()
            selector = CandidateSelector(
                evaluator=FakeEvaluator(
                    {
                        first.program_digest: (1.0, 0),
                        second.program_digest: (2.0, 0),
                    },
                    fail_digest=second.program_digest,
                ),
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    max_submissions=2,
                    validation=ValidationConfig(
                        episodes_per_candidate=2,
                        max_candidates=2,
                    ),
                ),
                recorder=FakeRecorder(),
            )

            with self.assertRaises(EvaluationError):
                selector.select(
                    (first, second),
                    (first.submission_id, second.submission_id),
                )

    def test_validation_rejects_a_different_environment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = make_submission(Path(temporary), 1)
            benchmark = StubBenchmark()
            selector = CandidateSelector(
                evaluator=FakeEvaluator(
                    {submission.program_digest: (1.0, 0)},
                    mismatch_environment=True,
                ),
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    validation=ValidationConfig(
                        episodes_per_candidate=1,
                    ),
                ),
                recorder=FakeRecorder(),
            )

            with self.assertRaises(EvaluationError):
                selector.select(
                    (submission,),
                    (submission.submission_id,),
                )

    def test_without_validation_the_single_candidate_is_not_reevaluated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = make_submission(Path(temporary), 1)
            evaluator = FakeEvaluator({})
            recorder = FakeRecorder()
            benchmark = StubBenchmark()
            selector = CandidateSelector(
                evaluator=evaluator,
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(),
                recorder=recorder,
            )

            selection = selector.select(
                (submission,),
                (submission.submission_id,),
            )

        self.assertEqual(selection.submission, submission)
        self.assertIsNone(selection.validation)
        self.assertEqual(evaluator.configs, [])
        self.assertEqual(recorder.events, [])


if __name__ == "__main__":
    unittest.main()
