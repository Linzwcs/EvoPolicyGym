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

from minigrid_red_blue_doors import (
    RedBlueDoorsBenchmark,
    RedBlueDoorsConfig,
    baseline_program,
)


class RedBlueDoorsBenchmarkTests(unittest.TestCase):
    def test_profiles_and_identity(self) -> None:
        default = RedBlueDoorsBenchmark()
        small = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="6x6"))
        self.assertEqual(
            default.spec.id,
            "minigrid/RedBlueDoors-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1_280)
        self.assertEqual(small.spec.max_episode_steps, 720)
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
            RedBlueDoorsConfig(profile="7x7")

    def test_episode_planning_is_split_scoped(self) -> None:
        benchmark = RedBlueDoorsBenchmark()
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
        benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="6x6"))
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
            image = observation["image"]
            self.assertIsInstance(image, TensorValue)
            assert isinstance(image, TensorValue)
            self.assertEqual(image.shape, (7, 7, 3))
            self.assertEqual(
                observation["mission"],
                "open the red door then the blue door",
            )
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_step_feedback_exposes_ordered_task_stage(self) -> None:
        benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="6x6"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            step = environment.step(6)
        finally:
            environment.close()
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["step_count"], 1)
        self.assertEqual(step.metrics["remaining_steps"], 719)
        self.assertEqual(step.metrics["done_count"], 1)
        self.assertEqual(step.metrics["ineffective_action"], True)
        self.assertIn(step.metrics["task_stage"], {"open_red", "find_red"})
        self.assertEqual(step.metrics["red_door_opened"], False)
        self.assertEqual(step.metrics["blue_door_opened"], False)
        self.assertEqual(step.metrics["terminal_reason"], "none")

    def test_scenario_override_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RedBlueDoorsBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "6x6"},
                )
            )

    def test_feedback_keeps_episode_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = RedBlueDoorsBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"profile"', trace)

    def test_real_order_errors_are_distinguished(self) -> None:
        benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="6x6"))
        blue_first = _run_episode(benchmark, (0, 2, 5))
        red_reclosed = _run_episode(
            benchmark,
            (1, 2, 2, 5, 5, 1, 1, 2, 2, 2, 5),
        )
        blue_metrics = blue_first.transitions[-1].step.metrics
        reclosed_metrics = red_reclosed.transitions[-1].step.metrics
        self.assertIsInstance(blue_metrics, dict)
        self.assertIsInstance(reclosed_metrics, dict)
        assert isinstance(blue_metrics, dict)
        assert isinstance(reclosed_metrics, dict)
        self.assertEqual(blue_metrics["terminal_reason"], "blue_before_red")
        self.assertEqual(blue_metrics["blue_opened_before_red"], True)
        self.assertEqual(blue_metrics["red_door_opened"], False)
        self.assertEqual(
            reclosed_metrics["terminal_reason"],
            "red_reclosed_before_blue",
        )
        self.assertEqual(
            reclosed_metrics["blue_opened_after_red_reclosed"],
            True,
        )
        self.assertEqual(reclosed_metrics["red_door_opened"], True)
        self.assertEqual(reclosed_metrics["red_door_open"], False)
        self.assertEqual(reclosed_metrics["red_door_reclosed"], True)

        feedback = benchmark.feedback((blue_first, red_reclosed))
        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["order_error_rate"], 1.0)
        self.assertEqual(feedback.content["blue_before_red_rate"], 0.5)
        self.assertEqual(
            feedback.content["blue_after_red_reclosed_rate"],
            0.5,
        )
        documents = tuple(
            json.loads(line) for line in feedback.artifacts[0].read_bytes().splitlines()
        )
        outcomes = tuple(
            document["outcome"] for document in documents if document["type"] == "episode"
        )
        self.assertEqual(
            outcomes,
            ("blue_before_red", "red_reclosed_before_blue"),
        )

    def test_time_limit_is_not_an_order_error(self) -> None:
        benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="6x6"))
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
        self.assertEqual(step.metrics["order_error"], False)
        self.assertEqual(step.metrics["remaining_steps"], 0)
        self.assertEqual(step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_every_profile(self) -> None:
        for profile in ("6x6", "8x8"):
            with self.subTest(profile=profile):
                benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile=profile))
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
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertEqual(
                    result.feedback.content["red_door_opened_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["order_error_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["blue_door_opened_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["red_door_reclosed_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.artifacts[0].name,
                    "trace.jsonl",
                )
                documents = tuple(
                    json.loads(line)
                    for line in result.feedback.artifacts[0].read_bytes().splitlines()
                )
                episode_documents = tuple(
                    document for document in documents if document["type"] == "episode"
                )
                self.assertTrue(
                    all(document["outcome"] == "success" for document in episode_documents)
                )


def _run_episode(
    benchmark: RedBlueDoorsBenchmark,
    actions: tuple[int, ...],
) -> EpisodeRecord:
    episode = EpisodeSpec(environment_seed=0)
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
        "mission": "open the red door then the blue door",
    }


if __name__ == "__main__":
    unittest.main()
