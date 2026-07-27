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

from minigrid_red_blue_doors import (
    RedBlueDoorsBenchmark,
    RedBlueDoorsConfig,
    baseline_program,
)


class RedBlueDoorsBenchmarkTests(unittest.TestCase):
    def test_profiles_and_identity(self) -> None:
        default = RedBlueDoorsBenchmark()
        small = RedBlueDoorsBenchmark(
            RedBlueDoorsConfig(profile="6x6")
        )
        self.assertEqual(
            default.spec.id,
            "minigrid/RedBlueDoors-v0/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1_280)
        self.assertEqual(small.spec.max_episode_steps, 720)
        self.assertNotEqual(
            default.spec.environment_digest,
            small.spec.environment_digest,
        )
        with self.assertRaises(ValueError):
            RedBlueDoorsConfig(profile="7x7")

    def test_episode_planning_is_split_scoped(self) -> None:
        benchmark = RedBlueDoorsBenchmark()
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
        benchmark = RedBlueDoorsBenchmark(
            RedBlueDoorsConfig(profile="6x6")
        )
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
            image = observation["image"]
            self.assertIsInstance(image, TensorValue)
            assert isinstance(image, TensorValue)
            self.assertEqual(image.shape, (7, 7, 3))
            self.assertEqual(
                observation["mission"],
                "open the red door then the blue door",
            )
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_scenario_override_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RedBlueDoorsBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "6x6"},
                )
            )

    def test_feedback_keeps_episode_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = RedBlueDoorsBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"profile"', trace)

    def test_baseline_solves_every_profile(self) -> None:
        for profile in ("6x6", "8x8"):
            with self.subTest(profile=profile):
                benchmark = RedBlueDoorsBenchmark(
                    RedBlueDoorsConfig(profile=profile)
                )
                result = evaluate(
                    baseline_program(),
                    benchmark,
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="validation",
                        episodes=8,
                        seed=5,
                        episode_timeout_seconds=10,
                    ),
                )
                self.assertEqual(result.feedback.score, 1.0)
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(
                    result.feedback.content["red_door_opened_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["order_error_rate"],
                    0.0,
                )
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
        "mission": "open the red door then the blue door",
    }


if __name__ == "__main__":
    unittest.main()
