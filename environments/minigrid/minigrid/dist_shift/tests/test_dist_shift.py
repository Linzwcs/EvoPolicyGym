from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import EpisodeRecord, EpisodeSpec
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_dist_shift import (
    DistShiftBenchmark,
    DistShiftConfig,
    baseline_program,
)


class DistShiftTests(unittest.TestCase):
    def test_profiles_define_identity(self) -> None:
        default = DistShiftBenchmark()
        shifted = DistShiftBenchmark(DistShiftConfig(profile="shift2"))
        self.assertEqual(
            default.spec.id,
            "minigrid/DistShift-v0/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 252)
        self.assertNotEqual(
            default.spec.environment_digest,
            shifted.spec.environment_digest,
        )
        with self.assertRaises(ValueError):
            DistShiftConfig(profile="shift3")

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = DistShiftBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=1, scenario={"strip2_row": 5})
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
            DistShiftBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        )
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_baseline_solves_all_profiles_without_hazard(self) -> None:
        profiles = ("shift1", "shift2")
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = DistShiftBenchmark(
                    DistShiftConfig(profile=profile)
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
        "mission": "get to the green goal square",
    }


if __name__ == "__main__":
    unittest.main()
