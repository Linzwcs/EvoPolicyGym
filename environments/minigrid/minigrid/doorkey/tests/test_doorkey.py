from __future__ import annotations

import json
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

from minigrid_doorkey import (
    DoorKeyBenchmark,
    DoorKeyConfig,
    baseline_program,
)


class DoorKeyBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = DoorKeyBenchmark()
        small = DoorKeyBenchmark(DoorKeyConfig(profile="5x5"))
        large = DoorKeyBenchmark(DoorKeyConfig(profile="16x16"))

        self.assertEqual(
            default.spec.id,
            "minigrid/DoorKey-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 640)
        self.assertEqual(small.spec.max_episode_steps, 250)
        self.assertEqual(large.spec.max_episode_steps, 2_560)
        self.assertNotEqual(
            default.spec.environment_digest,
            small.spec.environment_digest,
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            large.spec.environment_digest,
        )
        self.assertEqual(
            default.spec.environment_parameters["profile"],
            "8x8",
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

        exposed = default.spec.environment_parameters["state_encoding"]
        self.assertIsInstance(exposed, dict)
        assert isinstance(exposed, dict)
        exposed["locked"] = 100
        fresh = default.spec.environment_parameters["state_encoding"]
        self.assertIsInstance(fresh, dict)
        assert isinstance(fresh, dict)
        self.assertEqual(fresh["locked"], 2)

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            DoorKeyConfig(profile="10x10")
        with self.assertRaises(ValueError):
            DoorKeyConfig(profile=8)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = DoorKeyBenchmark()

        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        validation = tuple(benchmark.episodes("validation", seed=7, count=10))

        self.assertEqual(train, repeated)
        self.assertEqual(len({item.environment_seed for item in train}), 10)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in validation
            )
        )
        self.assertTrue(all(item.scenario is None for item in train))

    def test_environment_is_conformant_and_rejects_invalid_actions(
        self,
    ) -> None:
        benchmark = DoorKeyBenchmark(DoorKeyConfig(profile="5x5"))
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 6),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(
                set(observation),
                {"image", "direction", "mission"},
            )
            image = observation["image"]
            self.assertIsInstance(image, TensorValue)
            assert isinstance(image, TensorValue)
            self.assertEqual(image.dtype, "uint8")
            self.assertEqual(image.shape, (7, 7, 3))
            self.assertEqual(len(image.data), 147)
            self.assertIn(observation["direction"], {0, 1, 2, 3})
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            7,
            True,
            2.0,
            [2],
        )
        for invalid in invalid_actions:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_step_feedback_exposes_task_funnel_and_exploration(self) -> None:
        benchmark = DoorKeyBenchmark(DoorKeyConfig(profile="5x5"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step(6)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["step_count"], 1)
            self.assertEqual(step.metrics["remaining_steps"], 249)
            self.assertEqual(step.metrics["done_count"], 1)
            self.assertEqual(step.metrics["ineffective_action"], True)
            self.assertEqual(step.metrics["unique_observation_count"], 1)
            self.assertIn(step.metrics["key_first_seen_step"], {-1, 0})
            self.assertIn(step.metrics["door_first_seen_step"], {-1, 0})
            self.assertEqual(step.metrics["key_pickup_step"], -1)
            self.assertEqual(step.metrics["door_open_step"], -1)
            self.assertEqual(step.metrics["terminal_reason"], "none")
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        benchmark = DoorKeyBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "5x5"},
                )
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = DoorKeyBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
        self.assertEqual(len(feedback.artifacts), 1)
        trace = feedback.artifacts[0]
        self.assertEqual(trace.name, "trace.jsonl")
        self.assertNotIn(b"environment_seed", trace.read_bytes())
        self.assertNotIn(b"policy_seed", trace.read_bytes())
        self.assertNotIn(b'"profile"', trace.read_bytes())
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        self.assertEqual(feedback.content["key_pickup_episodes"], 0)
        self.assertEqual(feedback.content["door_open_episodes"], 0)

    def test_baseline_solves_every_public_profile_and_publishes_trace(
        self,
    ) -> None:
        for profile in ("5x5", "6x6", "8x8", "16x16"):
            with self.subTest(profile=profile):
                benchmark = DoorKeyBenchmark(DoorKeyConfig(profile=profile))
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

                self.assertEqual(
                    result.benchmark_id,
                    "minigrid/DoorKey-v0/mean-return-v1",
                )
                self.assertEqual(
                    result.environment_digest,
                    benchmark.spec.environment_digest,
                )
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(
                    result.feedback.content["key_pickup_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["door_open_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["key_found_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["goal_found_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["episodes_stalled_before_key_pickup"],
                    0,
                )
                self.assertEqual(
                    result.feedback.content["episodes_stalled_before_door_open"],
                    0,
                )
                trace = result.feedback.artifacts[0]
                documents = tuple(json.loads(line) for line in trace.read_bytes().splitlines())
                transitions = tuple(
                    document for document in documents if document["type"] == "transition"
                )
                self.assertEqual(trace.name, "trace.jsonl")
                self.assertEqual(
                    trace.media_type,
                    "application/x-ndjson",
                )
                self.assertTrue(transitions)
                self.assertTrue(
                    all("grid_rows" in item["next_observation"] for item in transitions)
                )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "use the key to open the door and then get to the goal",
    }


if __name__ == "__main__":
    unittest.main()
