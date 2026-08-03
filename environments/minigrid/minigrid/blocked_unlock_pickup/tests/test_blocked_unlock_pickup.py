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

from minigrid_blocked_unlock_pickup import (
    BlockedUnlockPickupBenchmark,
    baseline_program,
)

_SUCCESS_ACTIONS = (
    1,
    1,
    1,
    2,
    2,
    3,
    1,
    1,
    4,
    1,
    2,
    0,
    2,
    1,
    3,
    1,
    2,
    2,
    1,
    2,
    0,
    5,
    1,
    1,
    4,
    1,
    1,
    2,
    2,
    2,
    0,
    2,
    2,
    1,
    3,
)


class BlockedUnlockPickupTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/BlockedUnlockPickup-v0/success-rate-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 576)
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
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertEqual(train, repeated)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )

    def test_environment_contract_and_invalid_actions(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
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
            self.assertTrue(str(observation["mission"]).startswith("pick up the "))
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_step_feedback_exposes_chain_state_and_failed_action(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            step = environment.step(3)
        finally:
            environment.close()
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["step_count"], 1)
        self.assertEqual(step.metrics["remaining_steps"], 575)
        self.assertEqual(step.metrics["target_label"], "purple_box")
        self.assertEqual(step.metrics["front_object_before_action"], "empty")
        self.assertEqual(step.metrics["pickup_attempt"], True)
        self.assertEqual(step.metrics["pickup_succeeded"], False)
        self.assertEqual(step.metrics["failed_pickup"], True)
        self.assertEqual(step.metrics["failed_pickup_count"], 1)
        self.assertEqual(step.metrics["pick_up_count"], 1)
        self.assertIn(
            step.metrics["task_stage"],
            {"find_blocker", "move_blocker"},
        )
        self.assertEqual(step.metrics["terminal_reason"], "none")

    def test_scenario_override_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BlockedUnlockPickupBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"room_size": 8})
            )

    def test_feedback_keeps_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = BlockedUnlockPickupBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"scenario"', trace)

    def test_real_target_box_destruction_is_reported(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        destroyed = _run_episode(
            benchmark,
            episode,
            (*_SUCCESS_ACTIONS[:-1], 5),
        )
        final = destroyed.transitions[-1].step
        self.assertFalse(final.terminated)
        self.assertFalse(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertIsInstance(final.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(final.metrics["target_in_front_before_action"], True)
        self.assertEqual(
            final.metrics["front_object_before_action"],
            "green_box",
        )
        self.assertEqual(final.metrics["target_destroyed_this_step"], True)
        self.assertEqual(final.metrics["target_destroyed"], True)
        self.assertEqual(final.metrics["target_destroyed_step"], 35)
        self.assertEqual(final.metrics["task_stage"], "target_destroyed")
        self.assertEqual(final.metrics["terminal_reason"], "none")

        feedback = benchmark.feedback((destroyed,))
        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["target_destroyed_rate"], 1.0)
        document = json.loads(feedback.artifacts[0].read_bytes().splitlines()[0])
        self.assertEqual(document["outcome"], "target_destroyed")
        self.assertEqual(document["target_destroyed_step"], 35)

    def test_time_limit_is_not_reported_as_success(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
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

    def test_baseline_completes_public_progress_ladder(self) -> None:
        result = evaluate(
            baseline_program(),
            BlockedUnlockPickupBenchmark(),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=12,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )
        self.assertEqual(result.feedback.score, 1.0)
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        for field in (
            "blocker_moved_rate",
            "blocker_dropped_rate",
            "key_picked_up_rate",
            "key_dropped_rate",
            "door_found_rate",
            "door_opened_rate",
            "target_found_rate",
        ):
            self.assertEqual(result.feedback.content[field], 1.0)
        for field in (
            "target_destroyed_rate",
            "failed_pickup_rate",
            "failed_drop_rate",
            "failed_toggle_rate",
        ):
            self.assertEqual(result.feedback.content[field], 0.0)
        self.assertEqual(
            result.feedback.artifacts[0].name,
            "trace.jsonl",
        )
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episodes = tuple(document for document in documents if document["type"] == "episode")
        self.assertTrue(all(document["outcome"] == "success" for document in episodes))
        self.assertTrue(
            all(
                document["blocker_moved_step"]
                < document["blocker_dropped_step"]
                < document["key_picked_up_step"]
                < document["door_opened_step"]
                <= document["target_first_seen_step"]
                for document in episodes
            )
        )


def _run_episode(
    benchmark: BlockedUnlockPickupBenchmark,
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
