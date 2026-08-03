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

from swimmer import SwimmerBenchmark, SwimmerConfig, baseline_program

_BODY_FIELDS = {
    "front_angle",
    "rotor1_angle",
    "rotor2_angle",
    "tip_x_velocity",
    "tip_y_velocity",
    "front_angular_velocity",
    "rotor1_angular_velocity",
    "rotor2_angular_velocity",
}
_METRIC_FIELDS = {
    "step_count",
    "remaining_steps",
    "seconds_per_step",
    "requested_rotor1_control",
    "requested_rotor2_control",
    "gear_scaled_rotor1_torque",
    "gear_scaled_rotor2_torque",
    "x_position",
    "y_position",
    "distance_from_origin",
    "forward_displacement",
    "lateral_displacement",
    "path_length",
    "step_average_x_velocity",
    "step_average_y_velocity",
    "observation_tip_x_velocity",
    "observation_tip_y_velocity",
    "backward_step_fraction",
    "reward_forward",
    "reward_control",
    "cumulative_return",
    "terminal_reason",
}


class SwimmerBenchmarkTests(unittest.TestCase):
    def test_config_controls_observation_schema_and_identity(self) -> None:
        excluded = SwimmerBenchmark()
        included = SwimmerBenchmark(
            SwimmerConfig(
                frame_skip=5,
                forward_reward_weight=2.0,
                ctrl_cost_weight=0.001,
                reset_noise_scale=0.05,
                exclude_current_positions_from_observation=False,
            )
        )

        self.assertEqual(
            excluded.spec.id,
            "gymnasium/Swimmer-v5/mean-return-v1",
        )
        self.assertEqual(excluded.spec.max_episode_steps, 1000)
        self.assertEqual(excluded.spec.primary_metric, "mean_return")
        self.assertEqual(
            excluded.spec.environment_parameters,
            {
                "frame_skip": 4,
                "model_timestep_seconds": 0.01,
                "seconds_per_step": 0.04,
                "actuator_gears": [150.0, 150.0],
                "forward_reward_weight": 1.0,
                "ctrl_cost_weight": 0.0001,
                "reward_formula": (
                    "forward_reward_weight*((x_after-x_before)/seconds_per_step)-"
                    "ctrl_cost_weight*sum(action^2)"
                ),
                "reset_noise_scale": 0.1,
                "exclude_current_positions_from_observation": True,
                "natural_termination": "none",
                "time_limit": 1000,
            },
        )
        self.assertNotEqual(
            excluded.spec.environment_digest,
            included.spec.environment_digest,
        )
        self.assertIsInstance(excluded.spec.observation_space, dict)
        self.assertIsInstance(included.spec.observation_space, dict)
        assert isinstance(excluded.spec.observation_space, dict)
        assert isinstance(included.spec.observation_space, dict)
        excluded_fields = excluded.spec.observation_space["fields"]
        included_fields = included.spec.observation_space["fields"]
        self.assertIsInstance(excluded_fields, dict)
        self.assertIsInstance(included_fields, dict)
        assert isinstance(excluded_fields, dict)
        assert isinstance(included_fields, dict)
        self.assertEqual(set(excluded_fields), _BODY_FIELDS)
        self.assertEqual(
            set(included_fields),
            {"tip_x_position", "tip_y_position", *_BODY_FIELDS},
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            SwimmerConfig(frame_skip=4.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SwimmerConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            SwimmerConfig(forward_reward_weight=1)
        with self.assertRaises(TypeError):
            SwimmerConfig(
                exclude_current_positions_from_observation=1  # type: ignore[arg-type]
            )
        for invalid in (-0.1, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    SwimmerConfig(ctrl_cost_weight=invalid)
        with self.assertRaises(ValueError):
            SwimmerConfig(reset_noise_scale=1.1)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = SwimmerBenchmark()

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
        benchmark = SwimmerBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=([0.0, 0.0], [0.5, -0.5]),
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
            step = environment.step([0.0, 0.0])
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertTrue(_METRIC_FIELDS.issubset(step.metrics))
            forward = step.metrics["reward_forward"]
            control = step.metrics["reward_control"]
            assert type(forward) is float
            assert type(control) is float
            self.assertAlmostEqual(step.reward, forward + control)
            self.assertEqual(step.metrics["step_count"], 1)
            self.assertEqual(step.metrics["remaining_steps"], 999)
            self.assertEqual(step.metrics["seconds_per_step"], 0.04)
            self.assertEqual(step.metrics["terminal_reason"], "none")
            controlled_step = environment.step([0.5, -0.25])
            self.assertIsInstance(controlled_step.metrics, dict)
            assert isinstance(controlled_step.metrics, dict)
            self.assertEqual(
                controlled_step.metrics["gear_scaled_rotor1_torque"],
                75.0,
            )
            self.assertEqual(
                controlled_step.metrics["gear_scaled_rotor2_torque"],
                -37.5,
            )
        finally:
            environment.close()
            environment.close()

    def test_position_including_environment_conforms(self) -> None:
        benchmark = SwimmerBenchmark(
            SwimmerConfig(exclude_current_positions_from_observation=False)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25, -0.25],),
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
                {"tip_x_position", "tip_y_position", *_BODY_FIELDS},
            )
        finally:
            environment.close()

    def test_environment_requires_two_exact_bounded_floats(self) -> None:
        benchmark = SwimmerBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0),
            [0.0],
            [0, 0],
            [1.1, 0.0],
            [math.nan, 0.0],
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
            SwimmerBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"exclude_current_positions_from_observation": False},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = SwimmerBenchmark()
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
        benchmark = SwimmerBenchmark()
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
            "gymnasium/Swimmer-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, 360.0)
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        transitions = tuple(document for document in documents if document["type"] == "transition")
        self.assertEqual(len(transitions), 1000)
        self.assertEqual(
            set(transitions[0]["observation"]),
            _BODY_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0, 0.0])
        self.assertEqual(
            transitions[0]["action_components"],
            {"rotor1_control": 0.0, "rotor2_control": 0.0},
        )
        self.assertTrue(_METRIC_FIELDS.issubset(transitions[0]["metrics"]))

    def test_sinusoidal_gait_improves_on_zero_torque(self) -> None:
        benchmark = SwimmerBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[tuple[float, float]] = []
        gait: list[tuple[float, float]] = []

        for episode in episodes:
            zero_torque.append(_rollout(benchmark, episode, gait=False))
            gait.append(_rollout(benchmark, episode, gait=True))

        self.assertGreater(
            statistics.fmean(item[0] for item in gait),
            statistics.fmean(item[0] for item in zero_torque),
        )
        self.assertGreater(
            statistics.fmean(item[1] for item in gait),
            statistics.fmean(item[1] for item in zero_torque),
        )
        self.assertTrue(all(item[1] > 0.0 for item in gait))


def _sample_observation() -> dict[str, PolicyValue]:
    return {field: 0.0 for field in _BODY_FIELDS}


def _rollout(
    benchmark: SwimmerBenchmark,
    episode: EpisodeSpec,
    *,
    gait: bool,
) -> tuple[float, float]:
    environment = benchmark.make_environment(episode)
    total = 0.0
    final_metrics: dict[str, PolicyValue] | None = None
    try:
        environment.reset()
        for step_index in range(1000):
            action: PolicyValue
            if gait:
                phase = 0.15 * step_index
                action = [
                    math.sin(phase),
                    math.sin(phase + 1.5 * math.pi),
                ]
            else:
                action = [0.0, 0.0]
            result = environment.step(action)
            total += result.reward
            assert isinstance(result.metrics, dict)
            final_metrics = result.metrics
            if result.done:
                break
    finally:
        environment.close()
    assert final_metrics is not None
    forward_displacement = final_metrics["forward_displacement"]
    assert type(forward_displacement) is float
    return total, forward_displacement


if __name__ == "__main__":
    unittest.main()
