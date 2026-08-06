from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_lava_gap import (
    LavaGapBenchmark,
    LavaGapConfig,
    baseline_program,
)


class LavaGapTests(unittest.TestCase):
    def test_profiles_define_identity(self) -> None:
        default = LavaGapBenchmark()
        small = LavaGapBenchmark(LavaGapConfig(profile="S5"))
        self.assertEqual(
            default.spec.id,
            "minigrid/LavaGap-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 196)
        self.assertEqual(small.spec.max_episode_steps, 100)
        self.assertEqual(
            default.spec.environment_parameters["image_axis_order"],
            ["view_x", "view_y", "channel"],
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
            small.spec.environment_digest,
        )
        with self.assertRaises(ValueError):
            LavaGapConfig(profile="S9")

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = LavaGapBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )
        with self.assertRaises(ValueError):
            benchmark.make_environment(EpisodeSpec(environment_seed=1, scenario={"size": 11}))

    def test_step_feedback_exposes_safety_and_exploration(self) -> None:
        benchmark = LavaGapBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(6, 0, 1, 2),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step(6)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["step_count"], 1)
            self.assertEqual(step.metrics["remaining_steps"], 195)
            self.assertEqual(step.metrics["done_count"], 1)
            self.assertEqual(step.metrics["ineffective_action"], True)
            self.assertEqual(step.metrics["unique_observation_count"], 1)
            self.assertIsInstance(step.metrics["goal_found"], bool)
            self.assertIsInstance(step.metrics["hazard_found"], bool)
            self.assertEqual(step.metrics["hazard_entered"], False)
            self.assertEqual(step.metrics["terminal_reason"], "none")
        finally:
            environment.close()

    def test_feedback_privacy(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = LavaGapBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_real_hazard_and_time_limit_outcomes_are_distinct(self) -> None:
        benchmark = LavaGapBenchmark(LavaGapConfig(profile="S5"))
        hazard_episode = EpisodeSpec(environment_seed=0)
        hazard_environment = benchmark.make_environment(hazard_episode)
        try:
            initial_observation = hazard_environment.reset()
            hazard_step = hazard_environment.step(2)
        finally:
            hazard_environment.close()
        self.assertTrue(hazard_step.terminated)
        self.assertFalse(hazard_step.truncated)
        self.assertEqual(hazard_step.reward, 0.0)
        self.assertIsInstance(hazard_step.metrics, dict)
        assert isinstance(hazard_step.metrics, dict)
        self.assertEqual(hazard_step.metrics["hazard_entered"], True)
        self.assertEqual(hazard_step.metrics["hazard_found"], True)
        self.assertEqual(hazard_step.metrics["terminal_reason"], "hazard")

        hazard_record = EpisodeRecord(
            episode=hazard_episode,
            policy_seed=0,
            initial_observation=initial_observation,
            transitions=(Transition(action=2, step=hazard_step),),
        )
        hazard_feedback = benchmark.feedback((hazard_record,))
        self.assertEqual(hazard_feedback.score, 0.0)
        self.assertIsInstance(hazard_feedback.content, dict)
        assert isinstance(hazard_feedback.content, dict)
        self.assertEqual(hazard_feedback.content["hazard_found_rate"], 1.0)
        self.assertEqual(hazard_feedback.content["hazard_entry_rate"], 1.0)
        hazard_documents = tuple(
            json.loads(line) for line in hazard_feedback.artifacts[0].read_bytes().splitlines()
        )
        self.assertEqual(hazard_documents[0]["outcome"], "hazard")

        timeout_environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            timeout_environment.reset()
            timeout_step = timeout_environment.step(6)
            for _ in range(benchmark.spec.max_episode_steps - 1):
                timeout_step = timeout_environment.step(6)
        finally:
            timeout_environment.close()
        self.assertFalse(timeout_step.terminated)
        self.assertTrue(timeout_step.truncated)
        self.assertIsInstance(timeout_step.metrics, dict)
        assert isinstance(timeout_step.metrics, dict)
        self.assertEqual(timeout_step.metrics["remaining_steps"], 0)
        self.assertEqual(timeout_step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_all_profiles_without_hazard(self) -> None:
        profiles = ("S5", "S6", "S7")
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = LavaGapBenchmark(LavaGapConfig(profile=profile))
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
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(
                    result.feedback.content["hazard_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["hazard_entry_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["episodes_goal_found_but_not_reached"],
                    0,
                )
                documents = tuple(
                    json.loads(line)
                    for line in result.feedback.artifacts[0].read_bytes().splitlines()
                )
                episodes = tuple(
                    document for document in documents if document["type"] == "episode"
                )
                self.assertTrue(episodes)
                self.assertTrue(all(document["outcome"] == "success" for document in episodes))


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
