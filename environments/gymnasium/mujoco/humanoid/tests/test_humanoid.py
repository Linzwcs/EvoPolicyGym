from __future__ import annotations

import json
import math
import statistics
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.artifacts import ARTIFACT_MAX_BYTES
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from humanoid import HumanoidBenchmark, HumanoidConfig, baseline_program

_JOINTS = (
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder_1",
    "right_shoulder_2",
    "right_elbow",
    "left_shoulder_1",
    "left_shoulder_2",
    "left_elbow",
)
_ACTION_JOINTS = (
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    *_JOINTS[3:],
)
_STATE_FIELDS = {
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    *(f"{joint}_angle" for joint in _JOINTS),
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    *(f"{joint}_angular_velocity" for joint in _JOINTS),
}
_BODIES = {
    "torso",
    "lower_waist",
    "pelvis",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_upper_arm",
    "right_lower_arm",
    "left_upper_arm",
    "left_lower_arm",
}
_INERTIA_COMPONENTS = {
    "inertia_upper_0",
    "inertia_upper_1",
    "inertia_upper_2",
    "inertia_upper_3",
    "inertia_upper_4",
    "inertia_upper_5",
    "mass_times_com_offset_x",
    "mass_times_com_offset_y",
    "mass_times_com_offset_z",
    "mass",
}
_BODY_VELOCITY_COMPONENTS = {
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
}
_EXTERNAL_FORCE_COMPONENTS = {
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
}
_METRIC_FIELDS = {
    "step_count",
    "remaining_steps",
    "seconds_per_step",
    "simulated_seconds",
    "requested_action_by_joint",
    "actuator_gear_scaled_controls",
    "sum_squared_action",
    "sum_absolute_action",
    "cumulative_absolute_action",
    "initial_x_position",
    "initial_y_position",
    "x_position",
    "y_position",
    "net_x_displacement",
    "net_y_displacement",
    "distance_from_origin",
    "minimum_x_position",
    "maximum_x_position",
    "center_of_mass_x_velocity",
    "center_of_mass_y_velocity",
    "horizontal_center_of_mass_speed",
    "minimum_center_of_mass_x_velocity",
    "maximum_center_of_mass_x_velocity",
    "maximum_horizontal_center_of_mass_speed",
    "forward_step_fraction",
    "mean_root_x_velocity_from_displacement",
    "torso_z_position",
    "minimum_torso_z_position",
    "healthy",
    "healthy_z_lower_bound",
    "healthy_z_upper_bound",
    "healthy_z_margin",
    "minimum_healthy_z_margin",
    "healthy_step_fraction",
    "torso_tilt_radians",
    "torso_tilt_degrees",
    "maximum_torso_tilt_radians",
    "quaternion_norm_error",
    "external_forces_in_observation",
    "sum_squared_external_force_components",
    "raw_contact_cost_before_clamp",
    "maximum_external_force_body_norm_this_step",
    "maximum_external_force_body_this_step",
    "maximum_external_force_body_norm",
    "maximum_external_force_body",
    "actuator_forces_in_observation",
    "maximum_absolute_actuator_force_this_step",
    "maximum_actuator_force_joint_this_step",
    "maximum_absolute_actuator_force",
    "maximum_actuator_force_joint",
    "reward_survive",
    "reward_forward",
    "reward_control",
    "reward_contact",
    "reward_from_public_terms",
    "cumulative_reward_survive",
    "cumulative_reward_forward",
    "cumulative_reward_control",
    "cumulative_reward_contact",
    "cumulative_return",
    "tendon_lengths",
    "tendon_velocities",
    "maximum_absolute_tendon_velocity",
    "terminal_reason",
}
_OPTIONAL_FIELDS = {
    "body_inertias",
    "body_velocities",
    "actuator_forces",
    "external_forces",
}


class HumanoidBenchmarkTests(unittest.TestCase):
    def test_config_controls_nested_schema_and_identity(self) -> None:
        default = HumanoidBenchmark()
        minimal = HumanoidBenchmark(
            HumanoidConfig(
                frame_skip=6,
                forward_reward_weight=2.0,
                ctrl_cost_weight=0.2,
                contact_cost_weight=0.000001,
                contact_cost_range=(-1.0, 8.0),
                healthy_reward=4.0,
                terminate_when_unhealthy=False,
                healthy_z_range=(0.8, 2.2),
                reset_noise_scale=0.02,
                exclude_current_positions_from_observation=False,
                include_cinert_in_observation=False,
                include_cvel_in_observation=False,
                include_qfrc_actuator_in_observation=False,
                include_cfrc_ext_in_observation=False,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/Humanoid-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertIsInstance(default.spec.action_space, dict)
        assert isinstance(default.spec.action_space, dict)
        self.assertEqual(
            default.spec.action_space["components"],
            list(_ACTION_JOINTS),
        )
        self.assertEqual(
            default.spec.action_space["actuator_gears"],
            [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                300.0,
                200.0,
                100.0,
                100.0,
                300.0,
                200.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
            ],
        )
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 5,
                "model_timestep_seconds": 0.003,
                "seconds_per_step": 0.015,
                "action_components": list(_ACTION_JOINTS),
                "actuator_gears": [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    300.0,
                    200.0,
                    100.0,
                    100.0,
                    300.0,
                    200.0,
                    25.0,
                    25.0,
                    25.0,
                    25.0,
                    25.0,
                    25.0,
                ],
                "forward_reward_weight": 1.25,
                "ctrl_cost_weight": 0.1,
                "contact_cost_weight": 0.0000005,
                "contact_cost_range": [None, 10.0],
                "healthy_reward": 5.0,
                "terminate_when_unhealthy": True,
                "healthy_z_range": [1.0, 2.0],
                "health_bounds": "strict_open_interval",
                "forward_velocity_source": ("whole_body_center_of_mass_displacement"),
                "position_metric_source": "root_qpos_xy",
                "reward_formula": (
                    "healthy_reward_if_healthy+forward_reward_weight*"
                    "com_x_velocity-ctrl_cost_weight*sum(action^2)-"
                    "clip(contact_cost_weight*sum(cfrc_ext^2),"
                    "contact_cost_range)"
                ),
                "reset_noise_scale": 0.01,
                "exclude_current_positions_from_observation": True,
                "include_cinert_in_observation": True,
                "include_cvel_in_observation": True,
                "include_qfrc_actuator_in_observation": True,
                "include_cfrc_ext_in_observation": True,
                "time_limit": 1000,
            },
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            minimal.spec.environment_digest,
        )
        self.assertIsInstance(default.spec.observation_space, dict)
        self.assertIsInstance(minimal.spec.observation_space, dict)
        assert isinstance(default.spec.observation_space, dict)
        assert isinstance(minimal.spec.observation_space, dict)
        default_fields = default.spec.observation_space["fields"]
        minimal_fields = minimal.spec.observation_space["fields"]
        self.assertIsInstance(default_fields, dict)
        self.assertIsInstance(minimal_fields, dict)
        assert isinstance(default_fields, dict)
        assert isinstance(minimal_fields, dict)
        self.assertEqual(
            set(default_fields),
            _STATE_FIELDS | _OPTIONAL_FIELDS,
        )
        self.assertEqual(
            set(minimal_fields),
            {"torso_x_position", "torso_y_position", *_STATE_FIELDS},
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            HumanoidConfig(frame_skip=5.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HumanoidConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            HumanoidConfig(healthy_reward=5)
        with self.assertRaises(TypeError):
            HumanoidConfig(contact_cost_range=[None, 10.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HumanoidConfig(contact_cost_range=(10.0, 10.0))
        with self.assertRaises(ValueError):
            HumanoidConfig(contact_cost_range=(None, math.inf))
        with self.assertRaises(ValueError):
            HumanoidConfig(healthy_z_range=(2.0, 1.0))
        with self.assertRaises(TypeError):
            HumanoidConfig(
                include_cinert_in_observation=1  # type: ignore[arg-type]
            )
        for invalid in (-0.1, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    HumanoidConfig(ctrl_cost_weight=invalid)
        with self.assertRaises(ValueError):
            HumanoidConfig(reset_noise_scale=1.1)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = HumanoidBenchmark()

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

    def test_default_environment_is_nested_and_conformant(self) -> None:
        benchmark = HumanoidBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        _policy_action([0.0] * 17),
                        _policy_action([0.2, -0.2] * 8 + [0.1]),
                    ),
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
                _STATE_FIELDS | _OPTIONAL_FIELDS,
            )
            inertias = observation["body_inertias"]
            self.assertIsInstance(inertias, dict)
            assert isinstance(inertias, dict)
            self.assertEqual(set(inertias), _BODIES)
            torso_inertia = inertias["torso"]
            self.assertIsInstance(torso_inertia, dict)
            assert isinstance(torso_inertia, dict)
            self.assertEqual(
                set(torso_inertia),
                _INERTIA_COMPONENTS,
            )
            body_velocities = observation["body_velocities"]
            self.assertIsInstance(body_velocities, dict)
            assert isinstance(body_velocities, dict)
            torso_velocity = body_velocities["torso"]
            self.assertIsInstance(torso_velocity, dict)
            assert isinstance(torso_velocity, dict)
            self.assertEqual(
                set(torso_velocity),
                _BODY_VELOCITY_COMPONENTS,
            )
            external = observation["external_forces"]
            self.assertIsInstance(external, dict)
            assert isinstance(external, dict)
            torso_external = external["torso"]
            self.assertIsInstance(torso_external, dict)
            assert isinstance(torso_external, dict)
            self.assertEqual(
                set(torso_external),
                _EXTERNAL_FORCE_COMPONENTS,
            )
            step = environment.step([0.0] * 17)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(set(step.metrics), _METRIC_FIELDS)
            forward = _float(step.metrics["reward_forward"])
            control = _float(step.metrics["reward_control"])
            contact = _float(step.metrics["reward_contact"])
            survive = _float(step.metrics["reward_survive"])
            self.assertAlmostEqual(
                step.reward,
                forward + control + contact + survive,
            )
            self.assertEqual(step.metrics["seconds_per_step"], 0.015)
            self.assertEqual(step.metrics["terminal_reason"], "none")
            self.assertTrue(step.metrics["healthy"])
            action_by_joint = step.metrics["requested_action_by_joint"]
            self.assertIsInstance(action_by_joint, dict)
            assert isinstance(action_by_joint, dict)
            self.assertEqual(set(action_by_joint), set(_ACTION_JOINTS))
            self.assertEqual(
                step.metrics["actuator_gear_scaled_controls"],
                {joint: 0.0 for joint in _ACTION_JOINTS},
            )
            self.assertTrue(step.metrics["external_forces_in_observation"])
            self.assertIsInstance(
                step.metrics["raw_contact_cost_before_clamp"],
                float,
            )
            tendons = step.metrics["tendon_lengths"]
            self.assertIsInstance(tendons, dict)
            assert isinstance(tendons, dict)
            self.assertEqual(
                set(tendons),
                {"left_hip_to_knee", "right_hip_to_knee"},
            )
        finally:
            environment.close()

    def test_real_action_cost_and_nonuniform_gears_are_public(self) -> None:
        benchmark = HumanoidBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=321))
        action = [0.0] * 17
        action[0] = 0.4
        action[5] = -0.2
        action[6] = 0.1
        action[11] = -0.4
        try:
            environment.reset()
            step = environment.step(_policy_action(action))
            assert isinstance(step.metrics, dict)
            self.assertAlmostEqual(
                _float(step.metrics["reward_control"]),
                -0.1 * (0.16 + 0.04 + 0.01 + 0.16),
                places=7,
            )
            scaled = step.metrics["actuator_gear_scaled_controls"]
            self.assertIsInstance(scaled, dict)
            assert isinstance(scaled, dict)
            self.assertAlmostEqual(
                _float(scaled["abdomen_y"]),
                40.0,
                delta=0.00001,
            )
            self.assertAlmostEqual(
                _float(scaled["right_hip_y"]),
                -60.0,
                delta=0.00001,
            )
            self.assertAlmostEqual(
                _float(scaled["right_knee"]),
                20.0,
                delta=0.00001,
            )
            self.assertAlmostEqual(
                _float(scaled["right_shoulder_1"]),
                -10.0,
                delta=0.00001,
            )
        finally:
            environment.close()

    def test_narrow_height_range_reports_real_unhealthy_termination(
        self,
    ) -> None:
        benchmark = HumanoidBenchmark(HumanoidConfig(healthy_z_range=(1.8, 2.0)))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step([0.0] * 17)
            assert isinstance(step.metrics, dict)
            self.assertTrue(step.terminated)
            self.assertFalse(step.truncated)
            self.assertFalse(step.metrics["healthy"])
            self.assertLess(_float(step.metrics["healthy_z_margin"]), 0.0)
            self.assertEqual(step.metrics["reward_survive"], 0.0)
            self.assertEqual(step.metrics["terminal_reason"], "unhealthy")
        finally:
            environment.close()

    def test_minimal_observation_marks_unavailable_force_diagnostics(
        self,
    ) -> None:
        benchmark = HumanoidBenchmark(
            HumanoidConfig(
                include_qfrc_actuator_in_observation=False,
                include_cfrc_ext_in_observation=False,
            )
        )
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=789))
        try:
            environment.reset()
            step = environment.step([0.0] * 17)
            assert isinstance(step.metrics, dict)
            self.assertFalse(step.metrics["external_forces_in_observation"])
            self.assertIsNone(step.metrics["sum_squared_external_force_components"])
            self.assertFalse(step.metrics["actuator_forces_in_observation"])
            self.assertIsNone(step.metrics["maximum_absolute_actuator_force"])
        finally:
            environment.close()
            environment.close()

    def test_minimal_position_including_environment_conforms(self) -> None:
        benchmark = HumanoidBenchmark(
            HumanoidConfig(
                exclude_current_positions_from_observation=False,
                include_cinert_in_observation=False,
                include_cvel_in_observation=False,
                include_qfrc_actuator_in_observation=False,
                include_cfrc_ext_in_observation=False,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.1] * 17,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=456))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(
                set(observation),
                {"torso_x_position", "torso_y_position", *_STATE_FIELDS},
            )
        finally:
            environment.close()

    def test_environment_requires_seventeen_exact_bounded_floats(
        self,
    ) -> None:
        benchmark = HumanoidBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            tuple(0.0 for _ in range(17)),
            [0.0] * 16,
            [0] * 17,
            [0.5, *([0.0] * 16)],
            [math.nan, *([0.0] * 16)],
            True,
        )
        for invalid in invalid_actions:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            HumanoidBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"include_cinert_in_observation": False},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = HumanoidBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_sample_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -10000.0)
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
        self.assertEqual(feedback.content["mean_final_x_position"], None)

    def test_zero_torque_baseline_publishes_complete_nested_trace(
        self,
    ) -> None:
        benchmark = HumanoidBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=30,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/Humanoid-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(
            result.feedback.content["unhealthy_termination_episodes"],
            1,
        )
        self.assertIsInstance(
            result.feedback.content["mean_net_x_displacement"],
            float,
        )
        self.assertIsInstance(
            result.feedback.content["mean_episode_maximum_external_force_body_norm"],
            float,
        )
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episode = documents[0]
        transitions = tuple(document for document in documents if document["type"] == "transition")
        self.assertEqual(len(transitions), episode["steps"])
        self.assertGreater(len(transitions), 0)
        self.assertEqual(episode["outcome"], "unhealthy")
        self.assertEqual(
            set(transitions[0]["observation"]),
            _STATE_FIELDS | _OPTIONAL_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0] * 17)
        self.assertEqual(set(transitions[0]["metrics"]), _METRIC_FIELDS)

    def test_full_horizon_trace_is_uniformly_sampled_within_byte_limit(
        self,
    ) -> None:
        benchmark = HumanoidBenchmark(
            HumanoidConfig(terminate_when_unhealthy=False)
        )
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                episodes=1,
                seed=0,
                episode_timeout_seconds=30,
            ),
        )

        artifact = result.feedback.artifacts[0]
        documents = tuple(
            json.loads(line) for line in artifact.read_bytes().splitlines()
        )
        episode = documents[0]
        transitions = tuple(
            document
            for document in documents
            if document["type"] == "transition"
        )
        self.assertEqual(episode["steps"], 1000)
        self.assertEqual(episode["traced_transitions"], 100)
        self.assertEqual(episode["omitted_transitions"], 900)
        self.assertEqual(
            episode["trace_sampling"],
            "uniform_including_endpoints",
        )
        self.assertEqual(len(transitions), 100)
        self.assertEqual(transitions[0]["step_index"], 0)
        self.assertEqual(transitions[-1]["step_index"], 999)
        self.assertLessEqual(artifact.size, ARTIFACT_MAX_BYTES)

    def test_joint_stabilizer_improves_on_zero_torque(self) -> None:
        benchmark = HumanoidBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[float] = []
        stabilized: list[float] = []

        for episode in episodes:
            zero_torque.append(_rollout(benchmark, episode, stabilize=False))
            stabilized.append(_rollout(benchmark, episode, stabilize=True))

        self.assertGreater(
            statistics.fmean(stabilized),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    observation: dict[str, PolicyValue] = {field: 0.0 for field in _STATE_FIELDS}
    observation["body_inertias"] = {
        body: {component: 0.0 for component in _INERTIA_COMPONENTS} for body in _BODIES
    }
    observation["body_velocities"] = {
        body: {component: 0.0 for component in _BODY_VELOCITY_COMPONENTS} for body in _BODIES
    }
    observation["actuator_forces"] = {joint: 0.0 for joint in _JOINTS}
    observation["external_forces"] = {
        body: {component: 0.0 for component in _EXTERNAL_FORCE_COMPONENTS} for body in _BODIES
    }
    return observation


def _rollout(
    benchmark: HumanoidBenchmark,
    episode: EpisodeSpec,
    *,
    stabilize: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(1000):
            action: PolicyValue = [0.0] * 17
            if stabilize:
                assert isinstance(observation, dict)
                action = [
                    _clip(
                        -1.2 * _float(observation[f"{joint}_angle"])
                        - 0.01 * _float(observation[f"{joint}_angular_velocity"])
                    )
                    for joint in _ACTION_JOINTS
                ]
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    return total


def _float(value: PolicyValue) -> float:
    assert type(value) is float
    return value


def _clip(value: float) -> float:
    return max(-0.4, min(0.4, value))


def _policy_action(values: list[float]) -> PolicyValue:
    return [value for value in values]


if __name__ == "__main__":
    unittest.main()
