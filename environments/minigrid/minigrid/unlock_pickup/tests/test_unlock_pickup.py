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

from minigrid_unlock_pickup import UnlockPickupBenchmark, baseline_program

_SUCCESS_ACTIONS = (
    1,
    3,
    0,
    2,
    0,
    2,
    1,
    5,
    1,
    1,
    4,
    1,
    1,
    2,
    2,
    1,
    2,
    2,
    3,
)


class UnlockPickupTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = UnlockPickupBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/UnlockPickup-v0/mean-return-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 288)
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
        benchmark = UnlockPickupBenchmark()
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
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()

    def test_step_feedback_exposes_target_and_failed_interaction(self) -> None:
        benchmark = UnlockPickupBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            step = environment.step(4)
        finally:
            environment.close()
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["step_count"], 1)
        self.assertEqual(step.metrics["remaining_steps"], 287)
        self.assertIsInstance(step.metrics["target_label"], str)
        self.assertTrue(str(step.metrics["target_label"]).endswith("_box"))
        self.assertEqual(step.metrics["drop_attempt"], True)
        self.assertEqual(step.metrics["drop_succeeded"], False)
        self.assertEqual(step.metrics["failed_drop"], True)
        self.assertEqual(step.metrics["failed_drop_count"], 1)
        self.assertEqual(step.metrics["drop_count"], 1)
        self.assertEqual(step.metrics["terminal_reason"], "none")

    def test_scenario_and_feedback_privacy(self) -> None:
        with self.assertRaises(ValueError):
            UnlockPickupBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"size": 8})
            )
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = UnlockPickupBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_real_unlock_then_pickup_chain_reports_exact_objects(self) -> None:
        benchmark = UnlockPickupBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        record = _run_episode(benchmark, episode, _SUCCESS_ACTIONS)
        pickup_key = record.transitions[1].step
        open_door = record.transitions[7].step
        drop_key = record.transitions[10].step
        final = record.transitions[-1].step
        for step in (pickup_key, open_door, drop_key, final):
            self.assertIsInstance(step.metrics, dict)
        assert isinstance(pickup_key.metrics, dict)
        assert isinstance(open_door.metrics, dict)
        assert isinstance(drop_key.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(pickup_key.metrics["picked_up_label"], "red_key")
        self.assertEqual(open_door.metrics["door_opened_this_step"], True)
        self.assertEqual(
            open_door.metrics["front_object_before_action"],
            "red_locked_door",
        )
        self.assertFalse(open_door.terminated)
        self.assertEqual(drop_key.metrics["key_dropped_this_step"], True)
        self.assertEqual(drop_key.metrics["dropped_label"], "red_key")
        self.assertEqual(final.metrics["picked_up_label"], "yellow_box")
        self.assertEqual(final.metrics["target_label"], "yellow_box")
        self.assertEqual(final.metrics["key_color_found"], "red")
        self.assertEqual(final.metrics["door_color_found"], "red")
        self.assertEqual(final.metrics["terminal_reason"], "success")

    def test_real_target_box_destruction_is_reported(self) -> None:
        benchmark = UnlockPickupBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        record = _run_episode(
            benchmark,
            episode,
            (*_SUCCESS_ACTIONS[:-1], 5),
        )
        final = record.transitions[-1].step
        self.assertFalse(final.terminated)
        self.assertFalse(final.truncated)
        self.assertIsInstance(final.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(final.metrics["target_in_front_before_action"], True)
        self.assertEqual(
            final.metrics["front_object_before_action"],
            "yellow_box",
        )
        self.assertEqual(final.metrics["target_destroyed_this_step"], True)
        self.assertEqual(final.metrics["target_destroyed_step"], 19)
        self.assertEqual(final.metrics["task_stage"], "target_destroyed")
        feedback = benchmark.feedback((record,))
        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["target_destroyed_rate"], 1.0)
        document = json.loads(feedback.artifacts[0].read_bytes().splitlines()[0])
        self.assertEqual(document["outcome"], "target_destroyed")

    def test_time_limit_is_distinct_from_target_outcomes(self) -> None:
        benchmark = UnlockPickupBenchmark()
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
        self.assertEqual(step.metrics["target_destroyed"], False)
        self.assertEqual(step.metrics["remaining_steps"], 0)
        self.assertEqual(step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_task(self) -> None:
        result = evaluate(
            baseline_program(),
            UnlockPickupBenchmark(),
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
            "key_picked_up_rate",
            "key_dropped_rate",
            "door_found_rate",
            "door_opened_rate",
            "target_found_rate",
        ):
            self.assertEqual(result.feedback.content[name], 1.0)
        for name in (
            "target_destroyed_rate",
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
                document["key_color_found"] == document["door_color_found"]
                and document["key_picked_up_step"]
                < document["door_opened_step"]
                <= document["target_first_seen_step"]
                for document in episodes
            )
        )


def _run_episode(
    benchmark: UnlockPickupBenchmark,
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
        "mission": "pick up the purple box",
    }


if __name__ == "__main__":
    unittest.main()
