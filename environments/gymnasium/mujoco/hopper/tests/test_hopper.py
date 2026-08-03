from __future__ import annotations

import json
import math
import statistics
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from hopper import HopperBenchmark, HopperConfig, baseline_program

_BODY_FIELDS = {
    "torso_z_position",
    "torso_pitch_angle",
    "thigh_angle",
    "leg_angle",
    "foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_pitch_angular_velocity",
    "thigh_angular_velocity",
    "leg_angular_velocity",
    "foot_angular_velocity",
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
    "x_position",
    "net_x_displacement",
    "minimum_x_position",
    "maximum_x_position",
    "z_distance_from_origin",
    "x_velocity",
    "minimum_x_velocity",
    "maximum_x_velocity",
    "mean_x_velocity_from_displacement",
    "forward_step_fraction",
    "torso_z_position",
    "minimum_torso_z_position",
    "torso_pitch_radians",
    "torso_pitch_degrees",
    "maximum_absolute_torso_pitch_radians",
    "healthy",
    "healthy_state",
    "healthy_z",
    "healthy_angle",
    "failed_health_conditions",
    "healthy_state_margin",
    "healthy_z_margin",
    "healthy_angle_margin",
    "minimum_healthy_state_margin",
    "minimum_healthy_z_margin",
    "minimum_healthy_angle_margin",
    "healthy_step_fraction",
    "reward_forward",
    "reward_control",
    "reward_survive",
    "reward_from_public_terms",
    "cumulative_reward_forward",
    "cumulative_reward_control",
    "cumulative_reward_survive",
    "cumulative_return",
    "terminal_reason",
}


class HopperBenchmarkTests(unittest.TestCase):
    def test_config_controls_observation_schema_and_identity(self) -> None:
        default = HopperBenchmark()
        configured = HopperBenchmark(
            HopperConfig(
                frame_skip=5,
                forward_reward_weight=2.0,
                ctrl_cost_weight=0.002,
                healthy_reward=1.5,
                terminate_when_unhealthy=False,
                healthy_state_range=(-50.0, 50.0),
                healthy_z_range=(0.6, 3.0),
                healthy_angle_range=(-0.3, 0.3),
                reset_noise_scale=0.01,
                exclude_current_positions_from_observation=False,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/Hopper-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertIsInstance(default.spec.action_space, dict)
        assert isinstance(default.spec.action_space, dict)
        self.assertEqual(
            default.spec.action_space["components"],
            ["thigh", "leg", "foot"],
        )
        self.assertEqual(
            default.spec.action_space["actuator_gears"],
            [200.0, 200.0, 200.0],
        )
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 4,
                "model_timestep_seconds": 0.002,
                "seconds_per_step": 0.008,
                "action_components": ["thigh", "leg", "foot"],
                "actuator_gears": [200.0, 200.0, 200.0],
                "forward_reward_weight": 1.0,
                "ctrl_cost_weight": 0.001,
                "healthy_reward": 1.0,
                "terminate_when_unhealthy": True,
                "healthy_state_range": [-100.0, 100.0],
                "healthy_z_range": [0.7, None],
                "healthy_angle_range": [-0.2, 0.2],
                "health_bounds": "strict_open_intervals",
                "health_state_source": "unclipped_qpos[2:]+qvel",
                "observation_velocity_clipping": [-10.0, 10.0],
                "reset_noise_scale": 0.005,
                "exclude_current_positions_from_observation": True,
                "reward_formula": (
                    "forward_reward_weight*x_velocity+"
                    "healthy_reward_if_healthy-"
                    "ctrl_cost_weight*sum(action^2)"
                ),
                "natural_termination": ("unhealthy when terminate_when_unhealthy is true"),
                "time_limit": 1000,
            },
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            configured.spec.environment_digest,
        )
        self.assertIsInstance(default.spec.observation_space, dict)
        self.assertIsInstance(configured.spec.observation_space, dict)
        assert isinstance(default.spec.observation_space, dict)
        assert isinstance(configured.spec.observation_space, dict)
        default_fields = default.spec.observation_space["fields"]
        configured_fields = configured.spec.observation_space["fields"]
        self.assertIsInstance(default_fields, dict)
        self.assertIsInstance(configured_fields, dict)
        assert isinstance(default_fields, dict)
        assert isinstance(configured_fields, dict)
        self.assertEqual(set(default_fields), _BODY_FIELDS)
        self.assertEqual(
            set(configured_fields),
            {"torso_x_position", *_BODY_FIELDS},
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            HopperConfig(frame_skip=4.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HopperConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            HopperConfig(healthy_reward=1)
        with self.assertRaises(TypeError):
            HopperConfig(terminate_when_unhealthy=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            HopperConfig(healthy_state_range=[-1.0, 1.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HopperConfig(healthy_state_range=(1.0, 1.0))
        with self.assertRaises(ValueError):
            HopperConfig(healthy_z_range=(0.7, math.inf))
        with self.assertRaises(ValueError):
            HopperConfig(healthy_angle_range=(math.nan, 0.2))
        for invalid in (-0.1, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    HopperConfig(ctrl_cost_weight=invalid)
        with self.assertRaises(ValueError):
            HopperConfig(reset_noise_scale=1.1)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = HopperBenchmark()

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

    def test_default_environment_is_semantic_and_conformant(self) -> None:
        benchmark = HopperBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=([0.0, 0.0, 0.0], [0.5, -0.5, 0.25]),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), _BODY_FIELDS)
            step = environment.step([0.0, 0.0, 0.0])
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(set(step.metrics), _METRIC_FIELDS)
            forward = step.metrics["reward_forward"]
            control = step.metrics["reward_control"]
            survive = step.metrics["reward_survive"]
            assert type(forward) is float
            assert type(control) is float
            assert type(survive) is float
            self.assertAlmostEqual(
                step.reward,
                forward + control + survive,
            )
            self.assertEqual(
                step.metrics["requested_action_by_joint"],
                {"thigh": 0.0, "leg": 0.0, "foot": 0.0},
            )
            self.assertEqual(step.metrics["seconds_per_step"], 0.008)
            self.assertEqual(step.metrics["terminal_reason"], "none")
        finally:
            environment.close()
            environment.close()

    def test_position_including_environment_conforms(self) -> None:
        benchmark = HopperBenchmark(
            HopperConfig(
                healthy_z_range=(0.6, 3.0),
                exclude_current_positions_from_observation=False,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25, -0.25, 0.5],),
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
                {"torso_x_position", *_BODY_FIELDS},
            )
        finally:
            environment.close()

    def test_real_action_cost_and_actuator_gears_are_public(self) -> None:
        benchmark = HopperBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=321))
        try:
            environment.reset()
            step = environment.step([1.0, -0.5, 0.25])
            assert isinstance(step.metrics, dict)
            self.assertAlmostEqual(
                _float(step.metrics["reward_control"]),
                -0.001 * (1.0 + 0.25 + 0.0625),
                places=7,
            )
            self.assertEqual(
                step.metrics["actuator_gear_scaled_controls"],
                {"thigh": 200.0, "leg": -100.0, "foot": 50.0},
            )
            self.assertTrue(step.metrics["healthy"])
            self.assertEqual(step.metrics["failed_health_conditions"], [])
        finally:
            environment.close()

    def test_narrow_height_range_reports_real_unhealthy_termination(
        self,
    ) -> None:
        benchmark = HopperBenchmark(HopperConfig(healthy_z_range=(2.0, 3.0)))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step([0.0, 0.0, 0.0])
            assert isinstance(step.metrics, dict)
            self.assertTrue(step.terminated)
            self.assertFalse(step.truncated)
            self.assertFalse(step.metrics["healthy"])
            self.assertFalse(step.metrics["healthy_z"])
            failed_conditions = step.metrics["failed_health_conditions"]
            self.assertIsInstance(failed_conditions, list)
            assert isinstance(failed_conditions, list)
            self.assertIn(
                "torso_height",
                failed_conditions,
            )
            self.assertLess(_float(step.metrics["healthy_z_margin"]), 0.0)
            self.assertEqual(step.metrics["reward_survive"], 0.0)
            self.assertEqual(step.metrics["terminal_reason"], "unhealthy")
        finally:
            environment.close()

    def test_zero_torque_reaches_an_explained_real_outcome(self) -> None:
        benchmark = HopperBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            final = None
            for _ in range(1000):
                final = environment.step([0.0, 0.0, 0.0])
                if final.done:
                    break
            assert final is not None
            assert isinstance(final.metrics, dict)
            self.assertTrue(final.done)
            self.assertIn(
                final.metrics["terminal_reason"],
                {"unhealthy", "time_limit", "unhealthy_and_time_limit"},
            )
            if final.terminated:
                self.assertFalse(final.metrics["healthy"])
                self.assertTrue(final.metrics["failed_health_conditions"])
        finally:
            environment.close()

    def test_environment_requires_three_exact_bounded_floats(self) -> None:
        benchmark = HopperBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0, 0.0),
            [0.0, 0.0],
            [0, 0, 0],
            [1.1, 0.0, 0.0],
            [math.nan, 0.0, 0.0],
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
            HopperBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"terminate_when_unhealthy": False},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = HopperBenchmark()
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
        self.assertEqual(feedback.content["mean_final_x_position"], None)

    def test_zero_torque_baseline_publishes_complete_trace(self) -> None:
        benchmark = HopperBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=15,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/Hopper-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, 3800.0)
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(
            result.feedback.content["unhealthy_termination_episodes"],
            1,
        )
        self.assertIsInstance(
            result.feedback.content["mean_healthy_step_fraction"],
            float,
        )
        self.assertIsInstance(
            result.feedback.content["mean_net_x_displacement"],
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
            _BODY_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0, 0.0, 0.0])
        self.assertEqual(set(transitions[0]["metrics"]), _METRIC_FIELDS)

    def test_balance_controller_improves_on_zero_torque(self) -> None:
        benchmark = HopperBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[float] = []
        balance: list[float] = []

        for episode in episodes:
            zero_torque.append(_rollout(benchmark, episode, balance=False))
            balance.append(_rollout(benchmark, episode, balance=True))

        self.assertGreater(
            statistics.fmean(balance),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {field: 0.0 for field in _BODY_FIELDS}


def _rollout(
    benchmark: HopperBenchmark,
    episode: EpisodeSpec,
    *,
    balance: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(1000):
            action: PolicyValue = [0.0, 0.0, 0.0]
            if balance:
                assert isinstance(observation, dict)
                action = [
                    _clip(
                        -2.0 * _float(observation["thigh_angle"])
                        - 0.2 * _float(observation["thigh_angular_velocity"])
                    ),
                    _clip(
                        -2.0 * _float(observation["leg_angle"])
                        - 0.2 * _float(observation["leg_angular_velocity"])
                    ),
                    _clip(
                        -2.0 * _float(observation["foot_angle"])
                        - 0.2 * _float(observation["foot_angular_velocity"])
                    ),
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
    return max(-1.0, min(1.0, value))


if __name__ == "__main__":
    unittest.main()
