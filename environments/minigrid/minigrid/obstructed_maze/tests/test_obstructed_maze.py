from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_obstructed_maze import (
    ObstructedMazeBenchmark,
    ObstructedMazeConfig,
    baseline_program,
)


class ObstructedMazeTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = ObstructedMazeBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/ObstructedMaze-v0/success-rate-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 288)
        full = ObstructedMazeBenchmark(
            ObstructedMazeConfig(profile="Full-v1")
        )
        self.assertEqual(full.spec.max_episode_steps, 3600)
        self.assertNotEqual(
            benchmark.spec.environment_digest,
            full.spec.environment_digest,
        )
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertEqual(train, repeated)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )

    def test_environment_contract_and_invalid_actions(self) -> None:
        benchmark = ObstructedMazeBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 5, 6),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)
        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertIsInstance(observation["image"], TensorValue)
            self.assertTrue(
                str(observation["mission"]).startswith("pick up the ")
            )
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_scenario_override_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObstructedMazeBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"room_size": 8})
            )

    def test_feedback_keeps_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = ObstructedMazeBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"scenario"', trace)

    def test_baseline_completes_public_progress_ladder(self) -> None:
        profiles = (
            "1Dl-v0",
            "1Dlh-v0",
            "1Dlhb-v0",
            "2Dl-v0",
            "2Dlh-v0",
            "2Dlhb-v0",
            "2Dlhb-v1",
            "1Q-v0",
            "1Q-v1",
            "2Q-v0",
            "2Q-v1",
            "Full-v0",
            "Full-v1",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                result = evaluate(
                    baseline_program(),
                    ObstructedMazeBenchmark(
                        ObstructedMazeConfig(profile=profile)
                    ),
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="validation",
                        episodes=1,
                        seed=5,
                        episode_timeout_seconds=10,
                    ),
                )
                self.assertEqual(result.feedback.score, 1.0)
                self.assertEqual(
                    result.feedback.artifacts[0].name,
                    "trace.jsonl",
                )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "pick up the blue ball",
    }


if __name__ == "__main__":
    unittest.main()
