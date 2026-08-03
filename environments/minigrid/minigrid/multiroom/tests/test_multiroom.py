from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_multiroom import (
    MultiRoomBenchmark,
    MultiRoomConfig,
    baseline_program,
)


class MultiRoomBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = MultiRoomBenchmark()
        small = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
        fixed = MultiRoomBenchmark(MultiRoomConfig(profile="N4-S5"))
        legacy = MultiRoomBenchmark(MultiRoomConfig(profile="N4-S5-v0-legacy-N6"))

        self.assertEqual(default.spec.id, "minigrid/MultiRoom-v0/success-rate-v1")
        self.assertEqual(default.spec.max_episode_steps, 120)
        self.assertEqual(small.spec.max_episode_steps, 40)
        self.assertEqual(fixed.spec.max_episode_steps, 80)
        self.assertEqual(legacy.spec.max_episode_steps, 120)
        self.assertNotEqual(default.spec.environment_digest, small.spec.environment_digest)
        self.assertNotEqual(fixed.spec.environment_digest, legacy.spec.environment_digest)
        self.assertEqual(default.spec.environment_parameters["profile"], "N6-S10")
        self.assertEqual(legacy.spec.environment_parameters["maximum_rooms"], 6)
        self.assertEqual(small.spec.environment_parameters["required_connecting_doors"], 1)

    def test_spec_explains_view_actions_reward_and_termination(self) -> None:
        spec = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4")).spec
        image = spec.observation_space["fields"]["image"]

        self.assertEqual(image["axis_order"], ["view_x", "view_y", "channel"])
        self.assertEqual(
            spec.environment_parameters["view_forward_direction"],
            "decreasing view_y",
        )
        self.assertEqual(spec.environment_parameters["unused_actions"], [3, 4, 6])
        self.assertEqual(
            spec.environment_parameters["success_reward_formula"],
            "1 - 0.9*step_count/max_episode_steps",
        )
        self.assertIn(
            "moving forward onto the final green goal",
            spec.environment_parameters["natural_termination"],
        )
        self.assertIn(
            "do not identify physical doors",
            spec.environment_parameters["door_color_rule"],
        )

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            MultiRoomConfig(profile="N4-S5-v0")
        with self.assertRaises(ValueError):
            MultiRoomConfig(profile=6)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = MultiRoomBenchmark()

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

    def test_environment_is_conformant_and_rejects_invalid_actions(self) -> None:
        benchmark = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
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

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), {"image", "direction", "mission"})
            image = observation["image"]
            self.assertIsInstance(image, TensorValue)
            assert isinstance(image, TensorValue)
            self.assertEqual(image.dtype, "uint8")
            self.assertEqual(image.shape, (7, 7, 3))
            self.assertEqual(len(image.data), 147)
            self.assertEqual(
                observation["mission"],
                "traverse the rooms to get to the goal",
            )
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (-1, 7, True, 2.0, [2])
        for invalid in invalid_actions:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_real_trajectory_distinguishes_door_goal_and_success(self) -> None:
        benchmark = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, (0, 2, 1, 5, 2, 2, 2))

        opened = steps[3]
        self.assertFalse(opened.terminated)
        self.assertEqual(opened.reward, 0.0)
        self.assertEqual(opened.metrics["door_opened_this_step"], True)
        self.assertEqual(opened.metrics["door_open_event_count"], 1)
        self.assertEqual(opened.metrics["first_door_opened_step"], 4)
        self.assertEqual(opened.metrics["goal_found"], True)
        self.assertEqual(opened.metrics["goal_first_seen_step"], 4)

        crossed = steps[4]
        self.assertEqual(crossed.metrics["door_crossed_this_step"], True)
        self.assertEqual(crossed.metrics["door_crossing_event_count"], 1)

        final = steps[-1]
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertAlmostEqual(final.reward, 1.0 - 0.9 * 7 / 40)
        self.assertEqual(final.metrics["goal_in_front_before_action"], True)
        self.assertEqual(final.metrics["success"], True)
        self.assertEqual(final.metrics["terminal_reason"], "success")
        self.assertEqual(final.metrics["task_stage"], "success")
        self.assertEqual(final.metrics["remaining_steps"], 33)

    def test_door_metrics_count_events_without_claiming_unique_doors(self) -> None:
        benchmark = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, (0, 2, 1, 5, 5, 5))

        closed = steps[4]
        reopened = steps[5]
        self.assertEqual(closed.metrics["door_closed_this_step"], True)
        self.assertEqual(closed.metrics["door_close_event_count"], 1)
        self.assertEqual(reopened.metrics["door_opened_this_step"], True)
        self.assertEqual(reopened.metrics["door_open_event_count"], 2)
        self.assertEqual(reopened.metrics["door_crossing_event_count"], 0)
        self.assertNotIn("opened_doors", reopened.metrics)

    def test_failed_interactions_and_timeout_are_diagnostic(self) -> None:
        benchmark = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        environment = benchmark.make_environment(episode)
        try:
            environment.reset()
            first = environment.step(5)
            self.assertEqual(first.metrics["failed_toggle"], True)
            self.assertEqual(first.metrics["failed_toggle_count"], 1)
            self.assertEqual(first.metrics["ineffective_action"], True)
            final = first
            for _ in range(39):
                final = environment.step(6)
        finally:
            environment.close()

        self.assertFalse(final.terminated)
        self.assertTrue(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final.metrics["step_count"], 40)
        self.assertEqual(final.metrics["remaining_steps"], 0)
        self.assertEqual(final.metrics["unused_action_count"], 39)
        self.assertEqual(final.metrics["done_count"], 39)
        self.assertEqual(final.metrics["terminal_reason"], "time_limit")
        self.assertEqual(final.metrics["task_stage"], "time_limit")

    def test_episode_scenario_cannot_override_benchmark_configuration(self) -> None:
        benchmark = MultiRoomBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=1, scenario={"profile": "N2-S4"})
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = MultiRoomBenchmark()
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
        self.assertEqual(feedback.content["goal_found_episodes"], 0)
        self.assertEqual(feedback.content["door_opened_episodes"], 0)
        self.assertIsNone(feedback.content["mean_door_open_event_count"])

    def test_baseline_solves_every_public_profile_and_publishes_trace(self) -> None:
        profiles = ("N2-S4", "N4-S5", "N4-S5-v0-legacy-N6", "N6-S10")
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = MultiRoomBenchmark(MultiRoomConfig(profile=profile))
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

                self.assertEqual(
                    result.benchmark_id,
                    "minigrid/MultiRoom-v0/success-rate-v1",
                )
                self.assertEqual(result.environment_digest, benchmark.spec.environment_digest)
                self.assertEqual(result.feedback.score, 1.0)
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(result.feedback.content["goal_found_rate"], 1.0)
                self.assertEqual(result.feedback.content["door_opened_rate"], 1.0)
                trace = result.feedback.artifacts[0]
                documents = tuple(json.loads(line) for line in trace.read_bytes().splitlines())
                transitions = tuple(
                    document for document in documents if document["type"] == "transition"
                )
                self.assertEqual(trace.name, "trace.jsonl")
                self.assertEqual(trace.media_type, "application/x-ndjson")
                self.assertTrue(transitions)
                self.assertTrue(
                    all("visible_objects" in item["next_observation"] for item in transitions)
                )
                self.assertTrue(all("terminal_reason" in item["metrics"] for item in transitions))


def _run_actions(
    benchmark: MultiRoomBenchmark,
    episode: EpisodeSpec,
    actions: tuple[int, ...],
) -> tuple[Step, ...]:
    environment = benchmark.make_environment(episode)
    steps: list[Step] = []
    try:
        environment.reset()
        for action in actions:
            steps.append(environment.step(action))
    finally:
        environment.close()
    return tuple(steps)


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "traverse the rooms to get to the goal",
    }


if __name__ == "__main__":
    unittest.main()
