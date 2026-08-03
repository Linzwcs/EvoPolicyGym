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

from walker2d import Walker2dBenchmark, Walker2dConfig, baseline_program

_BODY_FIELDS = {
    "torso_z_position",
    "torso_angle",
    "right_thigh_angle",
    "right_leg_angle",
    "right_foot_angle",
    "left_thigh_angle",
    "left_leg_angle",
    "left_foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_angular_velocity",
    "right_thigh_angular_velocity",
    "right_leg_angular_velocity",
    "right_foot_angular_velocity",
    "left_thigh_angular_velocity",
    "left_leg_angular_velocity",
    "left_foot_angular_velocity",
}
_METRIC_FIELDS = {
    "step_count",
    "remaining_steps",
    "seconds_per_step",
    "requested_right_thigh_control",
    "gear_scaled_right_thigh_torque",
    "x_position",
    "forward_displacement",
    "step_average_x_velocity",
    "observation_torso_x_velocity",
    "torso_z_position",
    "height_health_margin",
    "torso_angle_radians",
    "angle_health_margin",
    "healthy",
    "velocity_observation_at_clip_limit",
    "reward_forward",
    "reward_control",
    "reward_survive",
    "cumulative_return",
    "terminal_reason",
}
_ANGLE_FIELDS = (
    "right_thigh_angle",
    "right_leg_angle",
    "right_foot_angle",
    "left_thigh_angle",
    "left_leg_angle",
    "left_foot_angle",
)
_VELOCITY_FIELDS = (
    "right_thigh_angular_velocity",
    "right_leg_angular_velocity",
    "right_foot_angular_velocity",
    "left_thigh_angular_velocity",
    "left_leg_angular_velocity",
    "left_foot_angular_velocity",
)


class Walker2dBenchmarkTests(unittest.TestCase):
    def test_config_controls_observation_schema_and_identity(self) -> None:
        default = Walker2dBenchmark()
        configured = Walker2dBenchmark(
            Walker2dConfig(
                frame_skip=5,
                forward_reward_weight=2.0,
                ctrl_cost_weight=0.002,
                healthy_reward=1.5,
                terminate_when_unhealthy=False,
                healthy_z_range=(0.7, 2.2),
                healthy_angle_range=(-1.2, 1.2),
                reset_noise_scale=0.01,
                exclude_current_positions_from_observation=False,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/Walker2d-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 4,
                "model_timestep_seconds": 0.002,
                "seconds_per_step": 0.008,
                "actuator_gears": [100.0] * 6,
                "forward_reward_weight": 1.0,
                "ctrl_cost_weight": 0.001,
                "healthy_reward": 1.0,
                "reward_formula": (
                    "forward_reward_weight*((x_after-x_before)/seconds_per_step)+"
                    "healthy_reward_if_strictly_healthy-ctrl_cost_weight*sum(action^2)"
                ),
                "terminate_when_unhealthy": True,
                "healthy_z_range": [0.8, 2.0],
                "healthy_angle_range": [-1.0, 1.0],
                "healthy_bounds_inclusive": False,
                "observation_velocity_clipping": [-10.0, 10.0],
                "reset_noise_scale": 0.005,
                "exclude_current_positions_from_observation": True,
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
            Walker2dConfig(frame_skip=4.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Walker2dConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            Walker2dConfig(healthy_reward=1)
        with self.assertRaises(TypeError):
            Walker2dConfig(terminate_when_unhealthy=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Walker2dConfig(healthy_z_range=[0.8, 2.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Walker2dConfig(healthy_z_range=(2.0, 0.8))
        with self.assertRaises(ValueError):
            Walker2dConfig(healthy_angle_range=(math.nan, 1.0))
        for invalid in (-0.1, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    Walker2dConfig(ctrl_cost_weight=invalid)
        with self.assertRaises(ValueError):
            Walker2dConfig(reset_noise_scale=1.1)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = Walker2dBenchmark()

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
        benchmark = Walker2dBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.5, -0.5, 0.25, -0.25, 0.75, -0.75],
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
            self.assertEqual(set(observation), _BODY_FIELDS)
            step = environment.step([0.0] * 6)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertTrue(_METRIC_FIELDS.issubset(step.metrics))
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
            self.assertEqual(step.metrics["step_count"], 1)
            self.assertEqual(step.metrics["remaining_steps"], 999)
            self.assertEqual(step.metrics["seconds_per_step"], 0.008)
            self.assertEqual(step.metrics["healthy"], True)
            self.assertEqual(step.metrics["terminal_reason"], "none")
            controlled_step = environment.step([0.5, -0.5, 0.25, -0.25, 0.75, -0.75])
            self.assertIsInstance(controlled_step.metrics, dict)
            assert isinstance(controlled_step.metrics, dict)
            self.assertEqual(
                controlled_step.metrics["gear_scaled_right_thigh_torque"],
                50.0,
            )
            self.assertEqual(
                controlled_step.metrics["gear_scaled_left_foot_torque"],
                -75.0,
            )
        finally:
            environment.close()
            environment.close()

    def test_position_including_environment_conforms(self) -> None:
        benchmark = Walker2dBenchmark(
            Walker2dConfig(exclude_current_positions_from_observation=False)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25, -0.25, 0.5, -0.5, 0.75, -0.75],),
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

    def test_unhealthy_walker_can_continue_without_survival_reward(
        self,
    ) -> None:
        benchmark = Walker2dBenchmark(Walker2dConfig(terminate_when_unhealthy=False))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            unhealthy = None
            for _ in range(300):
                step = environment.step([0.0] * 6)
                assert isinstance(step.metrics, dict)
                if step.metrics["healthy"] is False:
                    unhealthy = step
                    break
            self.assertIsNotNone(unhealthy)
            assert unhealthy is not None
            self.assertFalse(unhealthy.terminated)
            self.assertFalse(unhealthy.truncated)
            assert isinstance(unhealthy.metrics, dict)
            self.assertEqual(unhealthy.metrics["reward_survive"], 0.0)
            self.assertEqual(unhealthy.metrics["terminal_reason"], "none")
        finally:
            environment.close()

    def test_environment_requires_six_exact_bounded_floats(self) -> None:
        benchmark = Walker2dBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            [0.0] * 5,
            [0] * 6,
            [1.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            Walker2dBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"terminate_when_unhealthy": False},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = Walker2dBenchmark()
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
        benchmark = Walker2dBenchmark()
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
            "gymnasium/Walker2d-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episode = documents[0]
        transitions = tuple(document for document in documents if document["type"] == "transition")
        self.assertEqual(len(transitions), episode["steps"])
        self.assertGreater(len(transitions), 0)
        self.assertEqual(
            set(transitions[0]["observation"]),
            _BODY_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0] * 6)
        self.assertEqual(
            set(transitions[0]["action_components"]),
            {f"{name.removesuffix('_angle')}_control" for name in _ANGLE_FIELDS},
        )
        self.assertTrue(_METRIC_FIELDS.issubset(transitions[0]["metrics"]))

    def test_balance_controller_improves_on_zero_torque(self) -> None:
        benchmark = Walker2dBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[tuple[float, int, float]] = []
        balance: list[tuple[float, int, float]] = []

        for episode in episodes:
            zero_torque.append(_rollout(benchmark, episode, balance=False))
            balance.append(_rollout(benchmark, episode, balance=True))

        self.assertGreater(
            statistics.fmean(item[0] for item in balance),
            statistics.fmean(item[0] for item in zero_torque),
        )
        self.assertGreater(
            statistics.fmean(item[1] for item in balance),
            statistics.fmean(item[1] for item in zero_torque),
        )
        self.assertGreater(
            statistics.fmean(item[2] for item in balance),
            statistics.fmean(item[2] for item in zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {field: 0.0 for field in _BODY_FIELDS}


def _rollout(
    benchmark: Walker2dBenchmark,
    episode: EpisodeSpec,
    *,
    balance: bool,
) -> tuple[float, int, float]:
    environment = benchmark.make_environment(episode)
    total = 0.0
    final_metrics: dict[str, PolicyValue] | None = None
    try:
        observation = environment.reset()
        for _ in range(1000):
            action: PolicyValue = [0.0] * 6
            if balance:
                assert isinstance(observation, dict)
                action_values = [
                    _clip(-3.0 * _float(observation[angle]) - _float(observation[velocity]))
                    for angle, velocity in zip(
                        _ANGLE_FIELDS,
                        _VELOCITY_FIELDS,
                        strict=True,
                    )
                ]
                torso_correction = _clip_to(
                    -2.0 * _float(observation["torso_angle"])
                    - 0.3 * _float(observation["torso_angular_velocity"]),
                    bound=0.5,
                )
                action_values[0] = _clip(action_values[0] + torso_correction)
                action_values[3] = _clip(action_values[3] + torso_correction)
                action = _policy_action(action_values)
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            assert isinstance(result.metrics, dict)
            final_metrics = result.metrics
            if result.done:
                break
    finally:
        environment.close()
    assert final_metrics is not None
    steps = final_metrics["step_count"]
    forward_displacement = final_metrics["forward_displacement"]
    assert type(steps) is int
    assert type(forward_displacement) is float
    return total, steps, forward_displacement


def _float(value: PolicyValue) -> float:
    assert type(value) is float
    return value


def _clip(value: float) -> float:
    return _clip_to(value, bound=1.0)


def _clip_to(value: float, *, bound: float) -> float:
    return max(-bound, min(bound, value))


def _policy_action(values: list[float]) -> PolicyValue:
    return [value for value in values]


if __name__ == "__main__":
    unittest.main()
