from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

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
from evopolicygym.run import AssessmentConfig, RunConfig
from evopolicygym.run._selection.assessment import (
    ProgramAssessor,
    _assessment_seed,
)
from evopolicygym.run._selection.validation import _validation_seed


class StubBenchmark:
    @property
    def spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(
            id="example/assessment-v1",
            description="Assessment fixture.",
            observation_space=None,
            action_space=None,
            metadata={},
            max_episode_steps=1,
            primary_metric="reward",
            score_direction="maximize",
        )

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
        *,
        fail: bool = False,
        mismatch_environment: bool = False,
    ) -> None:
        self.fail = fail
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
        if self.fail:
            raise EvaluationError("trusted fixture failure")
        episodes = tuple(
            EpisodeSummary(
                status="policy_failed",
                reward=None,
                steps=0,
                failure="exception",
            )
            if index == 0
            else EpisodeSummary(
                status="completed",
                reward=5.0,
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
            feedback=Feedback(score=5.0),
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


def make_submission(root: Path) -> SubmissionResult:
    source = root / "program"
    source.mkdir()
    (source / "policy.py").write_text(
        "def make_policy(context):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        submission_id="submission-000001",
        program=Program.from_directory(source),
        episode_indices=(0,),
        episodes_used=1,
        episodes_remaining=0,
        feedback=Feedback(score=1.0),
        episodes=(
            EpisodeSummary(
                status="completed",
                reward=1.0,
                steps=1,
            ),
        ),
    )


class ProgramAssessorTests(unittest.TestCase):
    def test_assessment_evaluates_only_the_selected_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = make_submission(Path(temporary))
            evaluator = FakeEvaluator()
            recorder = FakeRecorder()
            benchmark = StubBenchmark()
            assessor = ProgramAssessor(
                evaluator=evaluator,
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    assessment=AssessmentConfig(
                        split="held-out",
                        episodes=3,
                    ),
                    seed=19,
                ),
                recorder=recorder,
            )

            result = assessor.assess(submission)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.submission_id, submission.submission_id)
        self.assertEqual(result.program_digest, submission.program_digest)
        self.assertEqual(result.split, "held-out")
        self.assertEqual(result.episodes, 3)
        self.assertEqual(result.score, 5.0)
        self.assertEqual(result.policy_failures, 1)
        self.assertEqual(len(evaluator.configs), 1)
        self.assertEqual(evaluator.configs[0].split, "held-out")
        self.assertEqual(evaluator.configs[0].episodes, 3)
        self.assertEqual(
            [name for name, _ in recorder.events],
            [
                "assessment_started",
                "assessment_episode_completed",
                "assessment_episode_completed",
                "assessment_episode_completed",
                "assessment_completed",
            ],
        )

    def test_assessment_uses_an_independent_seed_domain(self) -> None:
        self.assertNotEqual(
            _assessment_seed(7),
            _validation_seed(7),
        )

    def test_no_assessment_config_performs_no_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = make_submission(Path(temporary))
            evaluator = FakeEvaluator()
            recorder = FakeRecorder()
            benchmark = StubBenchmark()

            result = ProgramAssessor(
                evaluator=evaluator,
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(),
                recorder=recorder,
            ).assess(submission)

        self.assertIsNone(result)
        self.assertEqual(evaluator.configs, [])
        self.assertEqual(recorder.events, [])

    def test_trusted_assessment_failure_produces_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = make_submission(Path(temporary))
            benchmark = StubBenchmark()
            assessor = ProgramAssessor(
                evaluator=FakeEvaluator(fail=True),
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    assessment=AssessmentConfig(episodes=2),
                ),
                recorder=FakeRecorder(),
            )

            with self.assertRaises(EvaluationError):
                assessor.assess(submission)

    def test_assessment_rejects_a_different_environment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = make_submission(Path(temporary))
            benchmark = StubBenchmark()
            assessor = ProgramAssessor(
                evaluator=FakeEvaluator(mismatch_environment=True),
                benchmark=benchmark,
                spec=benchmark.spec,
                config=RunConfig(
                    assessment=AssessmentConfig(episodes=2),
                ),
                recorder=FakeRecorder(),
            )

            with self.assertRaises(EvaluationError):
                assessor.assess(submission)


if __name__ == "__main__":
    unittest.main()
