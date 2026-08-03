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

from minigrid_unlock import UnlockBenchmark, baseline_program

_SUCCESS_ACTIONS = (1, 1, 2, 2, 3, 1, 2, 2, 2, 0, 2, 1, 5)


class UnlockTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = UnlockBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/Unlock-v0/success-rate-v1",
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
        benchmark = UnlockBenchmark()
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

    def test_step_feedback_exposes_failed_interactions_and_progress(self) -> None:
        benchmark = UnlockBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            toggle = environment.step(5)
            drop = environment.step(4)
        finally:
            environment.close()
        self.assertIsInstance(toggle.metrics, dict)
        self.assertIsInstance(drop.metrics, dict)
        assert isinstance(toggle.metrics, dict)
        assert isinstance(drop.metrics, dict)
        self.assertEqual(toggle.metrics["step_count"], 1)
        self.assertEqual(toggle.metrics["remaining_steps"], 287)
        self.assertEqual(toggle.metrics["toggle_attempt"], True)
        self.assertEqual(toggle.metrics["failed_toggle"], True)
        self.assertEqual(toggle.metrics["failed_toggle_count"], 1)
        self.assertEqual(toggle.metrics["toggle_count"], 1)
        self.assertEqual(toggle.metrics["terminal_reason"], "none")
        self.assertEqual(drop.metrics["drop_attempt"], True)
        self.assertEqual(drop.metrics["failed_drop"], True)
        self.assertEqual(drop.metrics["failed_drop_count"], 1)
        self.assertIn(
            drop.metrics["task_stage"],
            {"find_key", "acquire_key"},
        )

    def test_scenario_and_feedback_privacy(self) -> None:
        with self.assertRaises(ValueError):
            UnlockBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"size": 8})
            )
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = UnlockBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_real_key_and_door_chain_reports_exact_colors(self) -> None:
        benchmark = UnlockBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        record = _run_episode(benchmark, episode, _SUCCESS_ACTIONS)
        pickup = record.transitions[4].step
        final = record.transitions[-1].step
        self.assertIsInstance(pickup.metrics, dict)
        self.assertIsInstance(final.metrics, dict)
        assert isinstance(pickup.metrics, dict)
        assert isinstance(final.metrics, dict)
        self.assertEqual(pickup.metrics["key_picked_up_this_step"], True)
        self.assertEqual(pickup.metrics["front_object_before_action"], "green_key")
        self.assertEqual(pickup.metrics["carried_key_color"], "green")
        self.assertEqual(final.metrics["door_opened_this_step"], True)
        self.assertEqual(
            final.metrics["front_object_before_action"],
            "green_locked_door",
        )
        self.assertEqual(final.metrics["key_color_found"], "green")
        self.assertEqual(final.metrics["door_color_found"], "green")
        self.assertEqual(final.metrics["matching_key_carried"], True)
        self.assertEqual(final.metrics["door_opened_step"], 13)
        self.assertEqual(final.metrics["terminal_reason"], "success")

        feedback = benchmark.feedback((record,))
        self.assertEqual(feedback.score, 1.0)
        document = json.loads(feedback.artifacts[0].read_bytes().splitlines()[0])
        self.assertEqual(document["outcome"], "success")
        self.assertEqual(document["key_color_found"], "green")
        self.assertEqual(document["door_color_found"], "green")

    def test_time_limit_remains_distinct_from_unlocking(self) -> None:
        benchmark = UnlockBenchmark()
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
        self.assertEqual(step.metrics["door_opened"], False)
        self.assertEqual(step.metrics["remaining_steps"], 0)
        self.assertEqual(step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_task(self) -> None:
        result = evaluate(
            baseline_program(),
            UnlockBenchmark(),
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
        self.assertEqual(
            result.feedback.content["key_picked_up_rate"],
            1.0,
        )
        self.assertEqual(
            result.feedback.content["door_opened_rate"],
            1.0,
        )
        self.assertEqual(result.feedback.content["key_dropped_rate"], 0.0)
        self.assertEqual(result.feedback.content["failed_pickup_rate"], 0.0)
        self.assertEqual(result.feedback.content["failed_drop_rate"], 0.0)
        self.assertEqual(result.feedback.content["failed_toggle_rate"], 0.0)
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episodes = tuple(document for document in documents if document["type"] == "episode")
        self.assertTrue(all(document["outcome"] == "success" for document in episodes))
        self.assertTrue(
            all(
                document["key_color_found"] == document["door_color_found"]
                and document["key_picked_up_step"] < document["door_opened_step"]
                for document in episodes
            )
        )


def _run_episode(
    benchmark: UnlockBenchmark,
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
        "mission": "open the door",
    }


if __name__ == "__main__":
    unittest.main()
