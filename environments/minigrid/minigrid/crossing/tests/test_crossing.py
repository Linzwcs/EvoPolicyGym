from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import EpisodeRecord, EpisodeSpec
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_crossing import (
    CrossingBenchmark,
    CrossingConfig,
    baseline_program,
)


class CrossingTests(unittest.TestCase):
    def test_profiles_define_identity(self) -> None:
        default = CrossingBenchmark()
        wall = CrossingBenchmark(CrossingConfig(profile="wall-S9-N3"))
        large = CrossingBenchmark(
            CrossingConfig(profile="lava-S11-N5")
        )
        self.assertEqual(
            default.spec.id,
            "minigrid/Crossing-v0/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 324)
        self.assertEqual(large.spec.max_episode_steps, 484)
        self.assertNotEqual(
            default.spec.environment_digest,
            wall.spec.environment_digest,
        )
        with self.assertRaises(ValueError):
            CrossingConfig(profile="S9-N3")

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = CrossingBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=1, scenario={"size": 11})
            )

    def test_feedback_privacy(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = (
            CrossingBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        )
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_baseline_solves_all_profiles_without_hazard(self) -> None:
        profiles = (
            "lava-S9-N1",
            "lava-S9-N2",
            "lava-S9-N3",
            "lava-S11-N5",
            "wall-S9-N1",
            "wall-S9-N2",
            "wall-S9-N3",
            "wall-S11-N5",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = CrossingBenchmark(
                    CrossingConfig(profile=profile)
                )
                result = evaluate(
                    baseline_program(),
                    benchmark,
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="validation",
                        episodes=6,
                        seed=5,
                        episode_timeout_seconds=10,
                    ),
                )
                self.assertEqual(result.feedback.score, 1.0)
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(
                    result.feedback.content["hazard_rate"],
                    0.0,
                )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "avoid the lava and get to the green goal square",
    }


if __name__ == "__main__":
    unittest.main()
