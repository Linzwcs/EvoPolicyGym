from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_keycorridor import (
    KeyCorridorBenchmark,
    KeyCorridorConfig,
    baseline_program,
)

_SUCCESS_ACTIONS = (
    1,
    1,
    1,
    5,
    2,
    3,
    1,
    1,
    2,
    5,
    1,
    1,
    2,
    4,
    1,
    1,
    2,
    2,
    3,
)


class KeyCorridorBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = KeyCorridorBenchmark()
        small = KeyCorridorBenchmark(KeyCorridorConfig(profile="S3R1"))
        large = KeyCorridorBenchmark(KeyCorridorConfig(profile="S6R3"))

        self.assertEqual(
            default.spec.id,
            "minigrid/KeyCorridor-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 480)
        self.assertEqual(small.spec.max_episode_steps, 270)
        self.assertEqual(large.spec.max_episode_steps, 1_080)
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
            "S4R3",
        )
        self.assertEqual(
            default.spec.environment_parameters["grid_width"],
            10,
        )
        self.assertEqual(
            default.spec.environment_parameters["grid_height"],
            10,
        )
        self.assertEqual(
            default.spec.environment_parameters["image_axis_order"],
            ["view_x", "view_y", "channel"],
        )
        self.assertEqual(
            default.spec.environment_parameters["direction_encoding"],
            {"east": 0, "south": 1, "west": 2, "north": 3},
        )
        self.assertEqual(
            default.spec.environment_parameters["success_reward_formula"],
            "1 - 0.9*step_count/max_episode_steps",
        )
        relationship = default.spec.environment_parameters["key_relationship"]
        self.assertIsInstance(relationship, str)
        self.assertIn("independent", str(relationship))

        exposed = default.spec.environment_parameters["color_encoding"]
        self.assertIsInstance(exposed, dict)
        assert isinstance(exposed, dict)
        exposed["purple"] = 100
        fresh = default.spec.environment_parameters["color_encoding"]
        self.assertIsInstance(fresh, dict)
        assert isinstance(fresh, dict)
        self.assertEqual(fresh["purple"], 3)

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            KeyCorridorConfig(profile="S4R2")
        with self.assertRaises(ValueError):
            KeyCorridorConfig(profile=4)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = KeyCorridorBenchmark()

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
        benchmark = KeyCorridorBenchmark(KeyCorridorConfig(profile="S3R1"))
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
            mission = observation["mission"]
            self.assertIsInstance(mission, str)
            assert isinstance(mission, str)
            self.assertTrue(mission.startswith("pick up the "))
            self.assertTrue(mission.endswith(" ball"))
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

    def test_step_feedback_exposes_search_and_action_usage(self) -> None:
        benchmark = KeyCorridorBenchmark(KeyCorridorConfig(profile="S3R1"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            step = environment.step(6)
        finally:
            environment.close()
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["step_count"], 1)
        self.assertEqual(step.metrics["remaining_steps"], 269)
        self.assertEqual(step.metrics["target_type"], "ball")
        self.assertTrue(str(step.metrics["target_label"]).endswith("_ball"))
        self.assertIsInstance(step.metrics["visible_door_count"], int)
        self.assertEqual(step.metrics["done_count"], 1)
        self.assertIn(
            step.metrics["task_stage"],
            {"explore_rooms", "acquire_key"},
        )
        self.assertEqual(step.metrics["terminal_reason"], "none")

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        benchmark = KeyCorridorBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "S3R1"},
                )
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = KeyCorridorBenchmark()
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
        self.assertEqual(feedback.content["key_found_episodes"], 0)
        self.assertEqual(feedback.content["key_pickup_episodes"], 0)
        self.assertEqual(
            feedback.content["target_door_open_episodes"],
            0,
        )

    def test_real_chain_distinguishes_target_key_and_exploration_door(
        self,
    ) -> None:
        benchmark = KeyCorridorBenchmark(KeyCorridorConfig(profile="S3R1"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        record = _run_episode(benchmark, episode, _SUCCESS_ACTIONS)
        exploration_door = record.transitions[3].step
        key_pickup = record.transitions[5].step
        target_door = record.transitions[9].step
        key_drop = record.transitions[13].step
        final = record.transitions[-1].step
        for step in (
            exploration_door,
            key_pickup,
            target_door,
            key_drop,
            final,
        ):
            self.assertIsInstance(step.metrics, dict)
        assert isinstance(exploration_door.metrics, dict)
        assert isinstance(key_pickup.metrics, dict)
        assert isinstance(target_door.metrics, dict)
        assert isinstance(key_drop.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(
            exploration_door.metrics["front_object_before_action"],
            "red_closed_door",
        )
        self.assertEqual(
            exploration_door.metrics["exploration_door_toggled_this_step"],
            True,
        )
        self.assertEqual(key_pickup.metrics["picked_up_label"], "yellow_key")
        self.assertEqual(
            target_door.metrics["front_object_before_action"],
            "yellow_locked_door",
        )
        self.assertEqual(
            target_door.metrics["target_door_opened_this_step"],
            True,
        )
        self.assertEqual(key_drop.metrics["dropped_label"], "yellow_key")
        self.assertEqual(final.metrics["target_color"], "red")
        self.assertEqual(final.metrics["target_label"], "red_ball")
        self.assertEqual(final.metrics["key_color_found"], "yellow")
        self.assertEqual(
            final.metrics["target_door_color_found"],
            "yellow",
        )
        self.assertEqual(final.metrics["picked_up_label"], "red_ball")
        self.assertEqual(final.metrics["terminal_reason"], "success")

    def test_target_pickup_with_occupied_hands_is_diagnosed(self) -> None:
        benchmark = KeyCorridorBenchmark(KeyCorridorConfig(profile="S3R1"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        actions = (*_SUCCESS_ACTIONS[:13], 6, *_SUCCESS_ACTIONS[14:])
        record = _run_episode(benchmark, episode, actions)
        final = record.transitions[-1].step
        self.assertFalse(final.terminated)
        self.assertFalse(final.truncated)
        self.assertIsInstance(final.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(final.metrics["front_object_before_action"], "red_ball")
        self.assertEqual(final.metrics["carried_object"], "yellow_key")
        self.assertEqual(
            final.metrics["target_pickup_blocked_by_carried_object"],
            True,
        )
        self.assertEqual(final.metrics["target_pickup_blocked_count"], 1)
        self.assertEqual(final.metrics["failed_pickup"], True)
        self.assertEqual(final.metrics["task_stage"], "free_hands_for_target")
        feedback = benchmark.feedback((record,))
        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(
            feedback.content["target_pickup_blocked_episode_rate"],
            1.0,
        )
        document = json.loads(feedback.artifacts[0].read_bytes().splitlines()[0])
        self.assertEqual(document["target_pickup_blocked"], True)
        self.assertEqual(document["outcome"], "incomplete")

    def test_time_limit_is_distinct_from_task_milestones(self) -> None:
        benchmark = KeyCorridorBenchmark(KeyCorridorConfig(profile="S3R1"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step(6)
            for _ in range(benchmark.spec.max_episode_steps - 1):
                step = environment.step(6)
        finally:
            environment.close()
        self.assertFalse(step.terminated)
        self.assertTrue(step.truncated)
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["success"], False)
        self.assertEqual(step.metrics["remaining_steps"], 0)
        self.assertEqual(step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_every_public_profile_and_publishes_trace(
        self,
    ) -> None:
        for profile in (
            "S3R1",
            "S3R2",
            "S3R3",
            "S4R3",
            "S5R3",
            "S6R3",
        ):
            with self.subTest(profile=profile):
                benchmark = KeyCorridorBenchmark(KeyCorridorConfig(profile=profile))
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
                    "minigrid/KeyCorridor-v0/mean-return-v1",
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
                    result.feedback.content["target_door_open_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["target_object_found_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["key_drop_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["target_door_found_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["target_pickup_blocked_episode_rate"],
                    0.0,
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
                    all("target_color" in item["next_observation"] for item in transitions)
                )
                episodes = tuple(
                    document for document in documents if document["type"] == "episode"
                )
                self.assertTrue(
                    all(
                        document["outcome"] == "success"
                        and document["key_color_found"] == document["target_door_color_found"]
                        and document["key_picked_up_step"]
                        < document["target_door_opened_step"]
                        <= document["target_first_seen_step"]
                        for document in episodes
                    )
                )


def _run_episode(
    benchmark: KeyCorridorBenchmark,
    episode: EpisodeSpec,
    actions: tuple[int, ...],
) -> EpisodeRecord:
    environment = benchmark.make_environment(episode)
    transitions: list[Transition] = []
    try:
        initial_observation = environment.reset()
        for action in actions:
            step = environment.step(action)
            transitions.append(Transition(action=action, step=step))
    finally:
        environment.close()
    return EpisodeRecord(
        episode=episode,
        policy_seed=0,
        initial_observation=initial_observation,
        transitions=tuple(transitions),
    )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "pick up the purple ball",
    }


if __name__ == "__main__":
    unittest.main()
