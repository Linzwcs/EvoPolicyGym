from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import EpisodeRecord, EpisodeSpec
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_empty import (
    EmptyBenchmark,
    EmptyConfig,
    baseline_program,
)


class EmptyTests(unittest.TestCase):
    def test_profiles_define_upstream_identity(self) -> None:
        default = EmptyBenchmark()
        random = EmptyBenchmark(EmptyConfig(profile="6x6-random"))
        self.assertEqual(
            default.spec.id,
            "minigrid/Empty-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 256)
        self.assertEqual(
            default.spec.metadata["environment"],
            "MiniGrid-Empty-8x8-v0",
        )
        self.assertEqual(
            default.spec.environment_parameters["image_channel_order"],
            ["object", "color", "state"],
        )
        self.assertEqual(
            default.spec.environment_parameters["direction_encoding"],
            {"east": 0, "south": 1, "west": 2, "north": 3},
        )
        self.assertEqual(
            default.spec.environment_parameters["success_reward_formula"],
            "1 - 0.9*step_count/max_episode_steps",
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            random.spec.environment_digest,
        )

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = EmptyBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )
        with self.assertRaises(ValueError):
            benchmark.make_environment(EpisodeSpec(environment_seed=1, scenario={"size": 11}))

    def test_feedback_privacy(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = EmptyBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_step_feedback_exposes_horizon_and_interaction_diagnostics(
        self,
    ) -> None:
        environment = EmptyBenchmark().make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step(6)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["step_count"], 1)
            self.assertEqual(step.metrics["remaining_steps"], 255)
            self.assertEqual(step.metrics["done_count"], 1)
            self.assertEqual(step.metrics["ineffective_action"], True)
            self.assertEqual(step.metrics["ineffective_action_fraction"], 1.0)
            self.assertEqual(step.metrics["unique_observation_count"], 1)
            self.assertEqual(
                step.metrics["success_reward_at_this_step"],
                1.0 - 0.9 / 256.0,
            )
            self.assertEqual(step.metrics["terminal_reason"], "none")
        finally:
            environment.close()

    def test_baseline_solves_all_profiles(self) -> None:
        profiles = (
            "5x5",
            "5x5-random",
            "6x6",
            "6x6-random",
            "8x8",
            "16x16",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = EmptyBenchmark(EmptyConfig(profile=profile))
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
                content = result.feedback.content
                assert isinstance(content, dict)
                self.assertEqual(content["success_rate"], 1.0)
                self.assertEqual(result.feedback.score, content["mean_return"])
                self.assertEqual(
                    content["episodes_goal_found_but_not_reached"],
                    0,
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
