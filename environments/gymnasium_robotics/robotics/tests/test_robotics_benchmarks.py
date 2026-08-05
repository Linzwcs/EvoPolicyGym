from __future__ import annotations

import io
import json
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
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
                    metrics = _step_metrics(step)
                    self.assertIn("goal_distance", metrics)
                    self.assertIn("action_l2_norm", metrics)
                    self.assertIn("state_motion_l2", metrics)
                    self.assertIn("task_stage", metrics)
                    self.assertIs(metrics["feedback_video_capture_failed"], False)
                    self.assertIsInstance(metrics["feedback_video_initial_rgb"], TensorValue)
                    self.assertIsInstance(metrics["feedback_video_rgb"], TensorValue)
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

    def test_camera_feedback_renders_from_episode_worker_thread(self) -> None:
        benchmark = RoboticsBenchmark()

        def run_episode_step() -> dict[str, PolicyValue]:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=17))
            try:
                environment.reset()
                step = environment.step([0.0, 0.0, 0.0, 0.0])
                return _step_metrics(step)
            finally:
                environment.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            metrics = executor.submit(run_episode_step).result(timeout=30)
        initial = metrics["feedback_video_initial_rgb"]
        result = metrics["feedback_video_rgb"]
        self.assertIsInstance(initial, TensorValue)
        self.assertIsInstance(result, TensorValue)
        assert isinstance(initial, TensorValue)
        assert isinstance(result, TensorValue)
        self.assertEqual(initial.shape, (128, 128, 3))
        self.assertEqual(result.shape, (128, 128, 3))
        self.assertIs(metrics["feedback_video_capture_failed"], False)

    def test_specs_publish_exact_profile_reward_and_success_semantics(self) -> None:
        fetch = RoboticsBenchmark().spec.environment_parameters
        maze = RoboticsBenchmark(RoboticsConfig(profile="point-maze")).spec.environment_parameters
        adroit = RoboticsBenchmark(
            RoboticsConfig(profile="adroit-hand-door")
        ).spec.environment_parameters
        pen = RoboticsBenchmark(
            RoboticsConfig(profile="hand-manipulate-pen")
        ).spec.environment_parameters
        fetch_success = fetch["success_condition"]
        maze_success = maze["success_condition"]
        pen_success = pen["success_condition"]
        fetch_reward = fetch["reward_semantics"]
        adroit_reward = adroit["reward_semantics"]
        action_handling = fetch["action_handling"]
        tensor_encoding = fetch["tensor_encoding"]
        assert isinstance(fetch_success, dict)
        assert isinstance(maze_success, dict)
        assert isinstance(pen_success, dict)
        assert isinstance(fetch_reward, str)
        assert isinstance(adroit_reward, str)
        assert isinstance(tensor_encoding, str)
        assert isinstance(action_handling, str)

        self.assertIn("-1 until", fetch_reward)
        self.assertEqual(
            fetch_success["euclidean_goal_distance_strictly_below"],
            0.05,
        )
        self.assertEqual(
            maze_success["euclidean_goal_distance_at_most"],
            0.45,
        )
        self.assertIn("dense", adroit_reward)
        self.assertEqual(pen_success["z_rotation_ignored"], True)
        self.assertIn("rejected", action_handling)
        self.assertIn("not iterable", tensor_encoding)
        self.assertIn("struct.iter_unpack", tensor_encoding)

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
                action: PolicyValue = [
                    *(
                        max(-1.0, min(1.0, 10.0 * (desired[index] - achieved[index])))
                        for index in range(3)
                    ),
                    0.0,
                ]
                step = environment.step(action)
                steps.append(step)
                observation = step.observation
                assert isinstance(observation, dict)

            first_metrics = _step_metrics(steps[0])
            final_metrics = _step_metrics(steps[2])
            self.assertEqual(first_metrics["new_best_goal_distance"], True)
            self.assertGreater(
                _number_metric(first_metrics, "goal_distance_improvement_this_step"),
                0.0,
            )
            self.assertEqual(steps[2].reward, 0.0)
            self.assertFalse(steps[2].terminated)
            self.assertEqual(final_metrics["success"], True)
            self.assertEqual(final_metrics["success_ever"], True)
            self.assertEqual(final_metrics["first_success_step"], 3)

            lost = environment.step([-1.0, -1.0, -1.0, 0.0])
            lost_metrics = _step_metrics(lost)
            self.assertEqual(lost_metrics["success"], False)
            self.assertEqual(lost_metrics["success_ever"], True)
            self.assertEqual(lost_metrics["success_lost_this_step"], True)
            self.assertEqual(lost_metrics["success_lost_count"], 1)
            self.assertEqual(lost_metrics["task_stage"], "goal_lost_after_achievement")
        finally:
            environment.close()

    def test_control_saturation_and_kitchen_subtasks_are_explicit(self) -> None:
        fetch = RoboticsBenchmark().make_environment(EpisodeSpec(environment_seed=5))
        try:
            fetch.reset()
            saturated = fetch.step([1.0, -1.0, 0.0, 0.5])
            saturated_metrics = _step_metrics(saturated)
            self.assertEqual(saturated_metrics["action_l2_norm"], 1.5)
            self.assertEqual(
                saturated_metrics["saturated_action_component_count"],
                2,
            )
            self.assertEqual(
                saturated_metrics["saturated_action_component_fraction"],
                0.5,
            )
            self.assertEqual(saturated_metrics["zero_action"], False)
        finally:
            fetch.close()

        kitchen_config = RoboticsConfig(profile="franka-kitchen")
        kitchen = RoboticsBenchmark(kitchen_config).make_environment(
            EpisodeSpec(environment_seed=5)
        )
        try:
            kitchen.reset()
            step = kitchen.step([0.0] * kitchen_config.action_size)
            metrics = _step_metrics(step)
            remaining_tasks = metrics["remaining_task_names"]
            assert isinstance(remaining_tasks, list)
            self.assertEqual(metrics["completed_task_names"], [])
            self.assertEqual(len(remaining_tasks), 7)
            self.assertEqual(metrics["completed_tasks"], 0)
            self.assertEqual(metrics["task_completion_fraction"], 0.0)
            self.assertEqual(metrics["task_progress"], 0.0)
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
        self.assertGreater(_number_metric(feedback.content, "mean_initial_goal_distance"), 0.0)
        self.assertEqual(feedback.content["mean_zero_action_fraction"], 1.0)
        self.assertEqual(
            feedback.content["mean_saturated_action_component_fraction"],
            0.0,
        )
        self.assertGreaterEqual(_number_metric(feedback.content, "mean_state_motion_l2"), 0.0)
        self.assertEqual(_step_metrics(transitions[-1].step)["terminal_reason"], "time_limit")
        trace = feedback.artifacts[0].read_bytes()
        self.assertNotIn(b"feedback_video_initial_rgb", trace)
        self.assertNotIn(b"feedback_video_rgb", trace)
        self.assertEqual(feedback.content["video_episodes"], 1)
        self.assertEqual(feedback.content["rendered_frame_evidence_episodes"], 1)
        self.assertEqual(len(feedback.artifacts), 3)
        evidence = feedback.artifacts[1]
        self.assertEqual(evidence.name, "episode-000/rendered-frames.npz")
        self.assertEqual(evidence.media_type, "application/x-npz")
        with numpy.load(io.BytesIO(evidence.read_bytes()), allow_pickle=False) as archive:
            self.assertEqual(archive["frames"].shape[1:], (128, 128, 3))
            self.assertEqual(int(archive["step_indices"][0]), -1)
            self.assertFalse(bool(archive["reward_present"][0]))
            self.assertTrue(archive["reward_present"][1:].all())
            initial_frame = _step_metrics(transitions[0].step)[
                "feedback_video_initial_rgb"
            ]
            assert isinstance(initial_frame, TensorValue)
            self.assertEqual(archive["frames"][0].tobytes(), initial_frame.data)
        self.assertEqual(feedback.artifacts[2].name, "episode-000/robot-camera.gif")
        self.assertTrue(feedback.artifacts[2].read_bytes().startswith(b"GIF89a"))

    def test_feedback_saves_frame_evidence_and_preview_per_episode(self) -> None:
        benchmark = RoboticsBenchmark()
        records: list[EpisodeRecord] = []
        for episode_index in range(3):
            episode = EpisodeSpec(environment_seed=100 + episode_index)
            environment = benchmark.make_environment(episode)
            try:
                initial = environment.reset()
                action: PolicyValue = [0.0, 0.0, 0.0, 0.0]
                step = environment.step(action)
            finally:
                environment.close()
            records.append(
                EpisodeRecord(
                    episode=episode,
                    policy_seed=200 + episode_index,
                    initial_observation=initial,
                    transitions=(Transition(action=action, step=step),),
                )
            )

        feedback = benchmark.feedback(tuple(records))

        self.assertEqual(
            [artifact.name for artifact in feedback.artifacts],
            [
                "trace.jsonl",
                "episode-000/rendered-frames.npz",
                "episode-000/robot-camera.gif",
                "episode-001/rendered-frames.npz",
                "episode-001/robot-camera.gif",
                "episode-002/rendered-frames.npz",
                "episode-002/robot-camera.gif",
            ],
        )
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["video_episode_results"], 3)
        self.assertEqual(feedback.content["video_episodes_without_gif"], 0)
        self.assertEqual(feedback.content["rendered_frame_evidence_episodes"], 3)
        manifests = feedback.content["rendered_frame_evidence"]
        assert isinstance(manifests, list)
        first_manifest = manifests[0]
        assert isinstance(first_manifest, dict)
        self.assertEqual(
            first_manifest["evidence_artifact"],
            "episode-000/rendered-frames.npz",
        )
        self.assertEqual(
            first_manifest["preview_artifact"],
            "episode-000/robot-camera.gif",
        )

    def test_zero_step_failure_has_explicit_unavailable_video_result(self) -> None:
        benchmark = RoboticsBenchmark()
        feedback = benchmark.feedback(
            (
                EpisodeRecord(
                    episode=EpisodeSpec(environment_seed=301),
                    policy_seed=401,
                    initial_observation={},
                    transitions=(),
                    policy_failure="invalid_action",
                ),
            )
        )
        self.assertEqual([artifact.name for artifact in feedback.artifacts], ["trace.jsonl"])
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["video_episode_results"], 1)
        self.assertEqual(feedback.content["video_episodes_without_gif"], 1)
        manifests = feedback.content["video_artifacts"]
        assert isinstance(manifests, list)
        manifest = manifests[0]
        assert isinstance(manifest, dict)
        self.assertEqual(manifest["status"], "unavailable")
        self.assertEqual(manifest["reason"], "no_recorded_frames")
        self.assertIsNone(manifest["frames_artifact"])
        self.assertIsNone(manifest["evidence_artifact"])
        self.assertIsNone(manifest["preview_artifact"])

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


def _step_metrics(step: Step) -> dict[str, PolicyValue]:
    metrics = step.metrics
    assert isinstance(metrics, dict)
    return metrics


def _number_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


if __name__ == "__main__":
    unittest.main()
