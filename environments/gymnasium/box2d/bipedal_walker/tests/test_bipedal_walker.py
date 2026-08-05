from __future__ import annotations

import io
import json
import math
import statistics
import unittest
from unittest.mock import patch

import numpy
from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue
from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

from bipedal_walker import (
    BipedalWalkerBenchmark,
    BipedalWalkerConfig,
    baseline_program,
)

_OBSERVATION_FIELDS = {
    "hull_angle",
    "hull_angular_velocity",
    "horizontal_velocity",
    "vertical_velocity",
    "left_hip_angle",
    "left_hip_angular_velocity",
    "left_knee_angle",
    "left_knee_angular_velocity",
    "left_foot_contact",
    "right_hip_angle",
    "right_hip_angular_velocity",
    "right_knee_angle",
    "right_knee_angular_velocity",
    "right_foot_contact",
    "lidar_ranges",
}


class BipedalWalkerBenchmarkTests(unittest.TestCase):
    def test_feedback_publishes_lossless_frames_and_mp4_video(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            action: PolicyValue = [0.0] * 4
            step = environment.step(action)
        finally:
            environment.close()
        record = EpisodeRecord(
            episode=episode,
            policy_seed=7,
            initial_observation=initial,
            transitions=(Transition(action=action, step=step),),
        )
        feedback = benchmark.feedback((record,))

        self.assertEqual(
            [artifact.name for artifact in feedback.artifacts],
            [
                "trace.jsonl",
                "episode-000/rendered-frames.npz",
                "episode-000/behavior.mp4",
            ],
        )
        evidence = feedback.artifacts[1]
        with numpy.load(io.BytesIO(evidence.read_bytes()), allow_pickle=False) as archive:
            self.assertEqual(archive["frames"].shape, (2, 400, 600, 3))
            self.assertEqual(archive["step_indices"].tolist(), [-1, 1])
            metrics = step.metrics
            assert isinstance(metrics, dict)
            initial_frame = metrics["feedback_visual_initial_rgb"]
            assert isinstance(initial_frame, TensorValue)
            self.assertEqual(archive["frames"][0].tobytes(), initial_frame.data)
        video = feedback.artifacts[2]
        self.assertEqual(video.media_type, "video/mp4")
        self.assertEqual(video.retention, "bulk")
        self.assertEqual(video.read_bytes()[4:8], b"ftyp")
        self.assertNotIn(b"feedback_visual_initial_rgb", feedback.artifacts[0].read_bytes())
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["rendered_frame_evidence_episodes"], 1)
        self.assertEqual(feedback.content["video_episodes"], 1)
        with patch(
            "bipedal_walker.visual._video_artifact",
            side_effect=RuntimeError("encoder unavailable"),
        ):
            fallback = benchmark.feedback((record,))
        self.assertEqual(
            [artifact.name for artifact in fallback.artifacts],
            ["trace.jsonl", "episode-000/rendered-frames.npz"],
        )
        assert isinstance(fallback.content, dict)
        self.assertEqual(fallback.content["rendered_frame_evidence_episodes"], 1)
        self.assertEqual(fallback.content["video_episodes"], 0)

    def test_config_controls_environment_identity(self) -> None:
        normal = BipedalWalkerBenchmark()
        hardcore = BipedalWalkerBenchmark(
            BipedalWalkerConfig(hardcore=True)
        )

        self.assertEqual(
            normal.spec.id,
            "gymnasium/BipedalWalker-v3/mean-return-v1",
        )
        self.assertEqual(normal.spec.max_episode_steps, 1600)
        self.assertEqual(normal.spec.primary_metric, "mean_return")
        self.assertFalse(normal.spec.environment_parameters["hardcore"])
        self.assertTrue(hardcore.spec.environment_parameters["hardcore"])
        self.assertEqual(
            normal.spec.environment_parameters["maximum_motor_torque"],
            80.0,
        )
        self.assertEqual(
            normal.spec.environment_parameters[
                "motor_energy_penalty_per_absolute_action_component"
            ],
            0.028,
        )
        self.assertIn("reward", normal.spec.description)
        self.assertIsInstance(normal.spec.observation_space, dict)
        assert isinstance(normal.spec.observation_space, dict)
        fields = normal.spec.observation_space["fields"]
        self.assertIsInstance(fields, dict)
        assert isinstance(fields, dict)
        hull_angle = fields["hull_angle"]
        self.assertIsInstance(hull_angle, dict)
        assert isinstance(hull_angle, dict)
        self.assertEqual(hull_angle["unit"], "radians")
        knee = fields["left_knee_angle"]
        self.assertIsInstance(knee, dict)
        assert isinstance(knee, dict)
        knee_meaning = knee["meaning"]
        self.assertIsInstance(knee_meaning, str)
        assert isinstance(knee_meaning, str)
        self.assertIn("plus 1 radian", knee_meaning)
        self.assertNotEqual(
            normal.spec.environment_digest,
            hardcore.spec.environment_digest,
        )

    def test_config_rejects_non_boolean_hardcore(self) -> None:
        with self.assertRaises(TypeError):
            BipedalWalkerConfig(hardcore=1)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = BipedalWalkerBenchmark()

        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        validation = tuple(
            benchmark.episodes("validation", seed=7, count=10)
        )

        self.assertEqual(train, repeated)
        self.assertEqual(len({item.environment_seed for item in train}), 10)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in validation
            )
        )
        self.assertTrue(all(item.scenario is None for item in train))

    def test_environment_is_deterministic_and_semantic(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, -0.5, 0.25, -0.25],
                    ),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), _OBSERVATION_FIELDS)
            self.assertIsInstance(observation["left_foot_contact"], bool)
            self.assertIsInstance(observation["right_foot_contact"], bool)
            lidar = observation["lidar_ranges"]
            self.assertIsInstance(lidar, list)
            assert isinstance(lidar, list)
            self.assertEqual(len(lidar), 10)
            self.assertTrue(all(type(value) is float for value in lidar))
        finally:
            environment.close()
            environment.close()

    def test_environment_requires_four_exact_bounded_floats(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0, 0.0, 0.0),
            [0.0, 0.0, 0.0],
            [0, 0, 0, 0],
            [1.1, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0],
            True,
        )
        for invalid in invalid_actions:
            environment = benchmark.make_environment(
                EpisodeSpec(environment_seed=123)
            )
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_hardcore_environment_conforms(self) -> None:
        benchmark = BipedalWalkerBenchmark(
            BipedalWalkerConfig(hardcore=True)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.0, 0.0, 0.0, 0.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_real_motor_command_reports_reward_and_actuator_semantics(self) -> None:
        environment = BipedalWalkerBenchmark().make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            environment.reset()
            result = environment.step([0.5, -0.5, 0.25, -0.25])
        finally:
            environment.close()

        self.assertFalse(result.done)
        metrics = _metrics(result)
        self.assertAlmostEqual(
            _float_metric(metrics, "requested_motor_energy_penalty"),
            0.028 * 1.5,
        )
        self.assertAlmostEqual(
            _float_metric(metrics, "charged_motor_energy_penalty"),
            0.028 * 1.5,
        )
        self.assertAlmostEqual(
            _float_metric(metrics, "reward_from_public_terms"),
            result.reward,
        )
        target_speeds = _object_metric(
            metrics,
            "target_motor_speeds_radians_per_second",
        )
        self.assertEqual(target_speeds["left_hip"], 4.0)
        self.assertEqual(target_speeds["left_knee"], -6.0)
        maximum_torques = _object_metric(metrics, "maximum_motor_torques")
        self.assertEqual(maximum_torques["left_hip"], 40.0)
        self.assertEqual(maximum_torques["right_knee"], 20.0)
        self.assertFalse(
            _bool_metric(metrics, "reward_was_terminal_override")
        )

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            BipedalWalkerBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"hardcore": True},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = BipedalWalkerBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_sample_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -1000.0)
        self.assertEqual(len(feedback.artifacts), 1)
        self.assertEqual(feedback.artifacts[0].name, "trace.jsonl")
        self.assertNotIn(
            b"environment_seed",
            feedback.artifacts[0].read_bytes(),
        )
        self.assertNotIn(
            b"policy_seed",
            feedback.artifacts[0].read_bytes(),
        )
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        self.assertEqual(feedback.content["failure_return"], -1000.0)

    def test_zero_torque_baseline_publishes_complete_trace(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/BipedalWalker-v3/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, 0.0)
        trace = result.feedback.artifacts[0]
        documents = tuple(
            json.loads(line)
            for line in trace.read_bytes().splitlines()
        )
        transitions = tuple(
            document
            for document in documents
            if document["type"] == "transition"
        )
        self.assertTrue(transitions)
        self.assertEqual(
            set(transitions[0]["observation"]),
            _OBSERVATION_FIELDS,
        )
        self.assertEqual(
            transitions[0]["action"],
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            set(transitions[0]["action_components"]),
            {"left_hip", "left_knee", "right_hip", "right_knee"},
        )
        self.assertEqual(
            transitions[-1]["metrics"]["terminal_reason"],
            "fall_or_behind_start",
        )
        self.assertTrue(
            transitions[-1]["metrics"]["reward_was_terminal_override"]
        )
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(
            result.feedback.content["fall_or_behind_start_episodes"],
            1,
        )
        self.assertEqual(result.feedback.content["completed_courses"], 0)
        self.assertEqual(
            result.feedback.content[
                "mean_episode_requested_motor_energy_penalty"
            ],
            0.0,
        )

    def test_reference_heuristic_completes_real_course_with_outcome(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        completed_step: Step | None = None
        outcomes: list[tuple[int, str]] = []
        for environment_seed in (789, 20, 31, 32, 34, 44, 45, 68, 71, 75, 77, 79, 80, 82):
            environment = benchmark.make_environment(
                EpisodeSpec(environment_seed=environment_seed)
            )
            controller = BipedalWalkerHeuristics()
            final_step: Step | None = None
            try:
                observation = environment.reset()
                for _ in range(1600):
                    assert isinstance(observation, dict)
                    action: PolicyValue = [
                        float(value)
                        for value in controller.step_heuristic(  # type: ignore[no-untyped-call]
                            _observation_vector(observation)
                        )
                    ]
                    final_step = environment.step(action)
                    observation = final_step.observation
                    if final_step.done:
                        break
            finally:
                environment.close()

            self.assertIsNotNone(final_step)
            assert final_step is not None
            terminal_reason = _string_metric(_metrics(final_step), "terminal_reason")
            outcomes.append((environment_seed, terminal_reason))
            if terminal_reason == "course_complete":
                completed_step = final_step
                break

        self.assertIsNotNone(completed_step, outcomes)
        assert completed_step is not None
        self.assertTrue(completed_step.terminated)
        self.assertFalse(completed_step.truncated)
        self.assertGreater(completed_step.reward, -100.0)
        metrics = _metrics(completed_step)
        self.assertEqual(
            _string_metric(metrics, "terminal_reason"),
            "course_complete",
        )
        self.assertGreater(
            _float_metric(metrics, "maximum_relative_progress_coordinate"),
            80.0,
        )

    def test_reference_heuristic_improves_on_zero_torque(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=4,
        )
        zero_torque: list[float] = []
        heuristic: list[float] = []

        for episode in episodes:
            zero_torque.append(
                _rollout(benchmark, episode, use_heuristic=False)
            )
            heuristic.append(
                _rollout(benchmark, episode, use_heuristic=True)
            )

        self.assertGreater(
            statistics.fmean(heuristic),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {
        "hull_angle": 0.0,
        "hull_angular_velocity": 0.0,
        "horizontal_velocity": 0.0,
        "vertical_velocity": 0.0,
        "left_hip_angle": 0.0,
        "left_hip_angular_velocity": 0.0,
        "left_knee_angle": 0.0,
        "left_knee_angular_velocity": 0.0,
        "left_foot_contact": True,
        "right_hip_angle": 0.0,
        "right_hip_angular_velocity": 0.0,
        "right_knee_angle": 0.0,
        "right_knee_angular_velocity": 0.0,
        "right_foot_contact": True,
        "lidar_ranges": [1.0] * 10,
    }


def _metrics(step: Step) -> dict[str, PolicyValue]:
    if type(step.metrics) is not dict:
        raise AssertionError("expected object metrics")
    return step.metrics


def _object_metric(
    metrics: dict[str, PolicyValue],
    name: str,
) -> dict[str, PolicyValue]:
    value = metrics.get(name)
    if type(value) is not dict:
        raise AssertionError(f"expected object metric {name}")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise AssertionError(f"expected string metric {name}")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise AssertionError(f"expected float metric {name}")
    return value


def _bool_metric(metrics: dict[str, PolicyValue], name: str) -> bool:
    value = metrics.get(name)
    if type(value) is not bool:
        raise AssertionError(f"expected bool metric {name}")
    return value


def _rollout(
    benchmark: BipedalWalkerBenchmark,
    episode: EpisodeSpec,
    *,
    use_heuristic: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    controller = BipedalWalkerHeuristics()
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(1600):
            assert isinstance(observation, dict)
            action: PolicyValue = (
                [
                    float(value)
                    for value in controller.step_heuristic(  # type: ignore[no-untyped-call]
                        _observation_vector(observation)
                    )
                ]
                if use_heuristic
                else [0.0, 0.0, 0.0, 0.0]
            )
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    return total


def _observation_vector(
    observation: dict[str, PolicyValue],
) -> list[float]:
    vector: list[float] = []
    for key in (
        "hull_angle",
        "hull_angular_velocity",
        "horizontal_velocity",
        "vertical_velocity",
        "left_hip_angle",
        "left_hip_angular_velocity",
        "left_knee_angle",
        "left_knee_angular_velocity",
        "left_foot_contact",
        "right_hip_angle",
        "right_hip_angular_velocity",
        "right_knee_angle",
        "right_knee_angular_velocity",
        "right_foot_contact",
    ):
        value = observation[key]
        if type(value) is bool:
            vector.append(float(value))
        else:
            assert type(value) is float
            vector.append(value)
    lidar = observation["lidar_ranges"]
    assert isinstance(lidar, list)
    for value in lidar:
        assert type(value) is float
        vector.append(value)
    return vector


if __name__ == "__main__":
    unittest.main()
