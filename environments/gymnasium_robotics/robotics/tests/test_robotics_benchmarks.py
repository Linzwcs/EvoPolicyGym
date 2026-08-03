from __future__ import annotations

import json
import struct
import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue, TensorValue

from robotics_benchmarks import (
    ROBOTICS_PROFILES,
    RoboticsBenchmark,
    RoboticsConfig,
    baseline_program,
)


class RoboticsBenchmarkTests(unittest.TestCase):
    def test_all_profiles_reset_and_take_one_strict_action(self) -> None:
        self.assertEqual(len(ROBOTICS_PROFILES), 21)
        for profile in ROBOTICS_PROFILES:
            with self.subTest(profile=profile):
                config = RoboticsConfig(profile=profile)
                benchmark = RoboticsBenchmark(config)
                environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
                try:
                    observation = environment.reset()
                    self.assertTrue(
                        type(observation) in {dict}
                        or hasattr(
                            observation,
                            "shape",
                        )
                    )
                    step = environment.step([0.0] * config.action_size)
                    self.assertIsInstance(step.reward, float)
                    self.assertIsInstance(step.metrics, dict)
                    self.assertIn("goal_distance", step.metrics)
                    self.assertIn("action_l2_norm", step.metrics)
                    self.assertIn("state_motion_l2", step.metrics)
                    self.assertIn("task_stage", step.metrics)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_public_identity(self) -> None:
        fetch = RoboticsBenchmark()
        kitchen = RoboticsBenchmark(RoboticsConfig(profile="franka-kitchen"))
        self.assertNotEqual(
            fetch.spec.environment_digest,
            kitchen.spec.environment_digest,
        )
        self.assertEqual(
            kitchen.spec.environment_parameters["profile"],
            "franka-kitchen",
        )
        self.assertEqual(kitchen.spec.max_episode_steps, 280)

    def test_specs_publish_exact_profile_reward_and_success_semantics(self) -> None:
        fetch = RoboticsBenchmark().spec.environment_parameters
        maze = RoboticsBenchmark(RoboticsConfig(profile="point-maze")).spec.environment_parameters
        adroit = RoboticsBenchmark(
            RoboticsConfig(profile="adroit-hand-door")
        ).spec.environment_parameters
        pen = RoboticsBenchmark(
            RoboticsConfig(profile="hand-manipulate-pen")
        ).spec.environment_parameters

        self.assertIn("-1 until", fetch["reward_semantics"])
        self.assertEqual(
            fetch["success_condition"]["euclidean_goal_distance_strictly_below"],
            0.05,
        )
        self.assertEqual(
            maze["success_condition"]["euclidean_goal_distance_at_most"],
            0.45,
        )
        self.assertIn("dense", adroit["reward_semantics"])
        self.assertEqual(pen["success_condition"]["z_rotation_ignored"], True)
        self.assertIn("rejected", fetch["action_handling"])

    def test_invalid_profile_and_action_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RoboticsConfig(profile="unknown")
        with self.assertRaises(TypeError):
            RoboticsConfig(profile=1)  # type: ignore[arg-type]

        environment = RoboticsBenchmark().make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([0, 0, 0, 0])
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_profile(self) -> None:
        with self.assertRaises(ValueError):
            RoboticsBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "ant-maze"},
                )
            )

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_real_fetch_reach_success_and_regression_are_distinct(self) -> None:
        environment = RoboticsBenchmark().make_environment(EpisodeSpec(environment_seed=5))
        try:
            observation = environment.reset()
            assert isinstance(observation, dict)
            steps = []
            for _ in range(3):
                achieved = _tensor_values(observation["achieved_goal"])
                desired = _tensor_values(observation["desired_goal"])
                action = [
                    max(-1.0, min(1.0, 10.0 * (desired[index] - achieved[index])))
                    for index in range(3)
                ] + [0.0]
                step = environment.step(action)
                steps.append(step)
                observation = step.observation
                assert isinstance(observation, dict)

            self.assertEqual(steps[0].metrics["new_best_goal_distance"], True)
            self.assertGreater(
                steps[0].metrics["goal_distance_improvement_this_step"],
                0.0,
            )
            self.assertEqual(steps[2].reward, 0.0)
            self.assertFalse(steps[2].terminated)
            self.assertEqual(steps[2].metrics["success"], True)
            self.assertEqual(steps[2].metrics["success_ever"], True)
            self.assertEqual(steps[2].metrics["first_success_step"], 3)

            lost = environment.step([-1.0, -1.0, -1.0, 0.0])
            self.assertEqual(lost.metrics["success"], False)
            self.assertEqual(lost.metrics["success_ever"], True)
            self.assertEqual(lost.metrics["success_lost_this_step"], True)
            self.assertEqual(lost.metrics["success_lost_count"], 1)
            self.assertEqual(lost.metrics["task_stage"], "goal_lost_after_achievement")
        finally:
            environment.close()

    def test_control_saturation_and_kitchen_subtasks_are_explicit(self) -> None:
        fetch = RoboticsBenchmark().make_environment(EpisodeSpec(environment_seed=5))
        try:
            fetch.reset()
            saturated = fetch.step([1.0, -1.0, 0.0, 0.5])
            self.assertEqual(saturated.metrics["action_l2_norm"], 1.5)
            self.assertEqual(
                saturated.metrics["saturated_action_component_count"],
                2,
            )
            self.assertEqual(
                saturated.metrics["saturated_action_component_fraction"],
                0.5,
            )
            self.assertEqual(saturated.metrics["zero_action"], False)
        finally:
            fetch.close()

        kitchen_config = RoboticsConfig(profile="franka-kitchen")
        kitchen = RoboticsBenchmark(kitchen_config).make_environment(
            EpisodeSpec(environment_seed=5)
        )
        try:
            kitchen.reset()
            step = kitchen.step([0.0] * kitchen_config.action_size)
            self.assertEqual(step.metrics["completed_task_names"], [])
            self.assertEqual(len(step.metrics["remaining_task_names"]), 7)
            self.assertEqual(step.metrics["completed_tasks"], 0)
            self.assertEqual(step.metrics["task_completion_fraction"], 0.0)
            self.assertEqual(step.metrics["task_progress"], 0.0)
            self.assertEqual(step.reward, 0.0)
        finally:
            kitchen.close()

    def test_feedback_traces_public_state_with_explicit_sampling(self) -> None:
        benchmark = RoboticsBenchmark(RoboticsConfig(profile="point-maze"))
        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        transitions: list[Transition] = []
        try:
            initial = environment.reset()
            for _ in range(benchmark.spec.max_episode_steps):
                action: PolicyValue = [0.0, 0.0]
                step = environment.step(action)
                transitions.append(Transition(action=action, step=step))
                if step.done:
                    break
        finally:
            environment.close()
        record = EpisodeRecord(
            episode=episode,
            policy_seed=456,
            initial_observation=initial,
            transitions=tuple(transitions),
        )

        feedback = benchmark.feedback((record,))

        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["traced_transitions"], 160)
        self.assertEqual(feedback.content["trace_transitions_omitted"], 140)
        documents = [json.loads(line) for line in feedback.artifacts[0].read_bytes().splitlines()]
        self.assertEqual(documents[0]["traced_steps"], 160)
        self.assertEqual(documents[0]["omitted_steps"], 140)
        transitions_json = [document for document in documents if document["type"] == "transition"]
        self.assertEqual(transitions_json[0]["step_index"], 0)
        self.assertEqual(transitions_json[127]["step_index"], 127)
        self.assertEqual(transitions_json[128]["step_index"], 268)
        self.assertEqual(transitions_json[-1]["step_index"], 299)
        self.assertIn("observation", transitions_json[0])
        self.assertIn("next_observation", transitions_json[0])
        self.assertGreater(feedback.content["mean_initial_goal_distance"], 0.0)
        self.assertEqual(feedback.content["mean_zero_action_fraction"], 1.0)
        self.assertEqual(
            feedback.content["mean_saturated_action_component_fraction"],
            0.0,
        )
        self.assertGreaterEqual(feedback.content["mean_state_motion_l2"], 0.0)
        self.assertEqual(transitions[-1].step.metrics["terminal_reason"], "time_limit")

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            RoboticsBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    ([0.0, 0.0, 0.0, 0.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)


def _tensor_values(value: PolicyValue) -> tuple[float, ...]:
    if type(value) is not TensorValue or value.dtype != "float64":
        raise AssertionError("expected a float64 tensor")
    return tuple(item[0] for item in struct.iter_unpack("<d", value.data))


if __name__ == "__main__":
    unittest.main()
