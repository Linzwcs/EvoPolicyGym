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

from minigrid_dynamic_obstacles import (
    DynamicObstaclesBenchmark,
    DynamicObstaclesConfig,
    baseline_program,
)


class DynamicObstaclesBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = DynamicObstaclesBenchmark()
        small = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="5x5-N2"))
        random = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="5x5-N2-random"))
        large = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="16x16-N8"))

        self.assertEqual(
            default.spec.id,
            "minigrid/DynamicObstacles-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 256)
        self.assertEqual(small.spec.max_episode_steps, 100)
        self.assertEqual(large.spec.max_episode_steps, 1_024)
        self.assertNotEqual(
            default.spec.environment_digest,
            small.spec.environment_digest,
        )
        self.assertNotEqual(
            small.spec.environment_digest,
            random.spec.environment_digest,
        )
        self.assertEqual(
            default.spec.environment_parameters["profile"],
            "8x8-N4",
        )
        self.assertEqual(
            large.spec.environment_parameters["obstacle_count"],
            8,
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
            default.spec.environment_parameters["collision_reward"],
            -1.0,
        )

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            DynamicObstaclesConfig(profile="8x8")
        with self.assertRaises(ValueError):
            DynamicObstaclesConfig(profile=8)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = DynamicObstaclesBenchmark()

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
        benchmark = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="5x5-N2"))
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2),
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
            self.assertEqual(
                observation["mission"],
                "get to the green goal square",
            )
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            3,
            6,
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

    def test_step_feedback_exposes_obstacles_and_action_usage(self) -> None:
        benchmark = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="5x5-N2"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            step = environment.step(0)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["step_count"], 1)
            self.assertEqual(step.metrics["remaining_steps"], 99)
            self.assertEqual(step.metrics["turn_left_count"], 1)
            self.assertEqual(step.metrics["move_forward_count"], 0)
            self.assertIsInstance(step.metrics["obstacle_visible"], bool)
            self.assertIsInstance(step.metrics["obstacle_found"], bool)
            self.assertIsInstance(
                step.metrics["front_object_before_action"],
                str,
            )
            self.assertEqual(step.metrics["collision"], False)
            self.assertEqual(step.metrics["terminal_reason"], "none")
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        benchmark = DynamicObstaclesBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "5x5-N2"},
                )
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = DynamicObstaclesBenchmark()
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
        self.assertEqual(feedback.content["collision_episodes"], 0)

    def test_real_ball_and_wall_collisions_are_distinguished(self) -> None:
        benchmark = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="5x5-N2"))
        episode = EpisodeSpec(environment_seed=0)

        ball_environment = benchmark.make_environment(episode)
        try:
            ball_initial = ball_environment.reset()
            ball_step = ball_environment.step(2)
        finally:
            ball_environment.close()
        self.assertEqual(ball_step.reward, -1.0)
        self.assertTrue(ball_step.terminated)
        self.assertIsInstance(ball_step.metrics, dict)
        assert isinstance(ball_step.metrics, dict)
        self.assertEqual(ball_step.metrics["obstacle_collision"], True)
        self.assertEqual(ball_step.metrics["wall_collision"], False)
        self.assertEqual(
            ball_step.metrics["front_object_before_action"],
            "ball",
        )
        self.assertEqual(
            ball_step.metrics["terminal_reason"],
            "obstacle_collision",
        )

        wall_environment = benchmark.make_environment(episode)
        try:
            wall_initial = wall_environment.reset()
            turn_step = wall_environment.step(0)
            wall_step = wall_environment.step(2)
        finally:
            wall_environment.close()
        self.assertEqual(wall_step.reward, -1.0)
        self.assertTrue(wall_step.terminated)
        self.assertIsInstance(wall_step.metrics, dict)
        assert isinstance(wall_step.metrics, dict)
        self.assertEqual(wall_step.metrics["obstacle_collision"], False)
        self.assertEqual(wall_step.metrics["wall_collision"], True)
        self.assertEqual(
            wall_step.metrics["front_object_before_action"],
            "wall",
        )
        self.assertEqual(
            wall_step.metrics["terminal_reason"],
            "wall_collision",
        )

        feedback = benchmark.feedback(
            (
                EpisodeRecord(
                    episode=episode,
                    policy_seed=0,
                    initial_observation=ball_initial,
                    transitions=(Transition(action=2, step=ball_step),),
                ),
                EpisodeRecord(
                    episode=episode,
                    policy_seed=1,
                    initial_observation=wall_initial,
                    transitions=(
                        Transition(action=0, step=turn_step),
                        Transition(action=2, step=wall_step),
                    ),
                ),
            )
        )
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["success_rate"], 0.0)
        self.assertEqual(feedback.score, feedback.content["mean_return"])
        self.assertEqual(feedback.score, -1.0)
        self.assertEqual(feedback.content["collision_rate"], 1.0)
        self.assertEqual(feedback.content["obstacle_collision_rate"], 0.5)
        self.assertEqual(feedback.content["wall_collision_rate"], 0.5)
        documents = tuple(
            json.loads(line) for line in feedback.artifacts[0].read_bytes().splitlines()
        )
        outcomes = tuple(
            document["outcome"] for document in documents if document["type"] == "episode"
        )
        self.assertEqual(
            outcomes,
            ("obstacle_collision", "wall_collision"),
        )

    def test_time_limit_is_not_reported_as_collision(self) -> None:
        benchmark = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile="5x5-N2"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step(0)
            for _ in range(benchmark.spec.max_episode_steps - 1):
                step = environment.step(0)
        finally:
            environment.close()
        self.assertFalse(step.terminated)
        self.assertTrue(step.truncated)
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["collision"], False)
        self.assertEqual(step.metrics["remaining_steps"], 0)
        self.assertEqual(step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_every_public_profile_and_publishes_trace(
        self,
    ) -> None:
        profiles = (
            "5x5-N2",
            "5x5-N2-random",
            "6x6-N3",
            "6x6-N3-random",
            "8x8-N4",
            "16x16-N8",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = DynamicObstaclesBenchmark(DynamicObstaclesConfig(profile=profile))
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
                    "minigrid/DynamicObstacles-v0/mean-return-v1",
                )
                self.assertEqual(
                    result.environment_digest,
                    benchmark.spec.environment_digest,
                )
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertEqual(
                    result.feedback.content["collision_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["obstacle_collision_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["wall_collision_rate"],
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
                    all("visible_objects" in item["next_observation"] for item in transitions)
                )
                episode_documents = tuple(
                    document for document in documents if document["type"] == "episode"
                )
                self.assertTrue(
                    all(document["outcome"] == "success" for document in episode_documents)
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
