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

from minigrid_locked_room import LockedRoomBenchmark, baseline_program

_SUCCESS_ACTIONS = (
    2,
    1,
    2,
    2,
    0,
    5,
    1,
    1,
    2,
    2,
    5,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    1,
    3,
    1,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    5,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
)


class LockedRoomTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = LockedRoomBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/LockedRoom-v0/mean-return-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 190)
        self.assertEqual(
            benchmark.spec.environment_parameters["image_axis_order"],
            ["view_x", "view_y", "channel"],
        )
        self.assertEqual(
            benchmark.spec.environment_parameters["direction_encoding"],
            {"east": 0, "south": 1, "west": 2, "north": 3},
        )
        self.assertEqual(
            benchmark.spec.environment_parameters["success_reward_formula"],
            "1 - 0.9*step_count/max_episode_steps",
        )
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )

    def test_environment_contract_and_invalid_action(self) -> None:
        benchmark = LockedRoomBenchmark()
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
            self.assertIsInstance(observation["image"], TensorValue)
            self.assertIn(" unlock the ", str(observation["mission"]))
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()

    def test_step_feedback_exposes_mission_and_action_usage(self) -> None:
        benchmark = LockedRoomBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            step = environment.step(6)
        finally:
            environment.close()
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["step_count"], 1)
        self.assertEqual(step.metrics["remaining_steps"], 189)
        self.assertIn(
            step.metrics["target_color"], {"red", "green", "blue", "purple", "yellow", "grey"}
        )
        self.assertIn(
            step.metrics["key_room_color"], {"red", "green", "blue", "purple", "yellow", "grey"}
        )
        self.assertNotEqual(
            step.metrics["target_color"],
            step.metrics["key_room_color"],
        )
        self.assertEqual(step.metrics["done_count"], 1)
        self.assertIn(
            step.metrics["task_stage"],
            {"find_key_room", "open_key_room"},
        )
        self.assertEqual(step.metrics["terminal_reason"], "none")

    def test_scenario_and_feedback_privacy(self) -> None:
        with self.assertRaises(ValueError):
            LockedRoomBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"size": 8})
            )
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = LockedRoomBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_real_mission_chain_separates_room_door_key_and_goal(self) -> None:
        benchmark = LockedRoomBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        record = _run_episode(benchmark, episode, _SUCCESS_ACTIONS)
        other_door = record.transitions[5].step
        key_room_door = record.transitions[10].step
        key_pickup = record.transitions[19].step
        target_door = record.transitions[38].step
        final = record.transitions[-1].step
        for step in (
            other_door,
            key_room_door,
            key_pickup,
            target_door,
            final,
        ):
            self.assertIsInstance(step.metrics, dict)
        assert isinstance(other_door.metrics, dict)
        assert isinstance(key_room_door.metrics, dict)
        assert isinstance(key_pickup.metrics, dict)
        assert isinstance(target_door.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(other_door.metrics["front_object_before_action"], "red_closed_door")
        self.assertEqual(other_door.metrics["key_room_door_opened"], False)
        self.assertEqual(key_room_door.metrics["front_object_before_action"], "green_closed_door")
        self.assertEqual(
            key_room_door.metrics["key_room_door_opened_this_step"],
            True,
        )
        self.assertEqual(key_pickup.metrics["front_object_before_action"], "grey_key")
        self.assertEqual(key_pickup.metrics["key_picked_up_this_step"], True)
        self.assertEqual(target_door.metrics["front_object_before_action"], "grey_locked_door")
        self.assertEqual(
            target_door.metrics["target_door_opened_this_step"],
            True,
        )
        self.assertFalse(target_door.terminated)
        self.assertEqual(final.metrics["target_color"], "grey")
        self.assertEqual(final.metrics["key_room_color"], "green")
        self.assertEqual(final.metrics["goal_in_front_before_action"], True)
        self.assertEqual(final.metrics["front_object_before_action"], "green_goal")
        self.assertEqual(final.metrics["unique_door_color_count_opened"], 3)
        self.assertEqual(final.metrics["door_open_event_count"], 3)
        self.assertEqual(final.metrics["terminal_reason"], "success")

        feedback = benchmark.feedback((record,))
        self.assertEqual(feedback.content["success_rate"], 1.0)
        self.assertEqual(feedback.score, feedback.content["mean_return"])
        document = json.loads(feedback.artifacts[0].read_bytes().splitlines()[0])
        self.assertEqual(document["outcome"], "success")
        self.assertEqual(document["target_color"], "grey")
        self.assertEqual(document["key_room_color"], "green")

    def test_time_limit_is_distinct_from_goal_entry(self) -> None:
        benchmark = LockedRoomBenchmark()
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

    def test_baseline_solves_task(self) -> None:
        result = evaluate(
            baseline_program(),
            LockedRoomBenchmark(),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=12,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )
        self.assertEqual(result.feedback.content["success_rate"], 1.0)
        self.assertEqual(result.feedback.score, result.feedback.content["mean_return"])
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        for name in (
            "key_room_door_opened_rate",
            "key_picked_up_rate",
            "target_door_found_rate",
            "target_door_opened_rate",
            "goal_found_rate",
        ):
            self.assertEqual(result.feedback.content[name], 1.0)
        for name in (
            "key_dropped_rate",
            "failed_pickup_rate",
            "failed_drop_rate",
            "failed_toggle_rate",
        ):
            self.assertEqual(result.feedback.content[name], 0.0)
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episodes = tuple(document for document in documents if document["type"] == "episode")
        self.assertTrue(all(document["outcome"] == "success" for document in episodes))
        self.assertTrue(
            all(
                document["key_room_door_opened_step"]
                <= document["key_picked_up_step"]
                < document["target_door_opened_step"]
                <= document["goal_first_seen_step"]
                for document in episodes
            )
        )


def _run_episode(
    benchmark: LockedRoomBenchmark,
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
        "mission": ("get the red key from the green room, unlock the red door and go to the goal"),
    }


if __name__ == "__main__":
    unittest.main()
