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

from inverted_double_pendulum import (
    InvertedDoublePendulumBenchmark,
    InvertedDoublePendulumConfig,
    baseline_program,
)

_OBSERVATION_FIELDS = {
    "cart_position",
    "pole1_sin",
    "pole2_relative_sin",
    "pole1_cos",
    "pole2_relative_cos",
    "cart_velocity",
    "pole1_angular_velocity",
    "pole2_relative_angular_velocity",
    "cart_constraint_force",
}
_METRIC_FIELDS = {
    "step_count",
    "remaining_steps",
    "seconds_per_step",
    "simulated_seconds",
    "requested_cart_control",
    "actuator_gear_scaled_cart_force",
    "cumulative_absolute_action",
    "cart_position",
    "minimum_cart_position",
    "maximum_cart_position",
    "cart_velocity",
    "pole1_angle_radians",
    "pole2_relative_angle_radians",
    "pole2_absolute_angle_radians",
    "maximum_absolute_pole1_angle_radians",
    "maximum_absolute_pole2_absolute_angle_radians",
    "pole1_angular_velocity_observed",
    "pole2_relative_angular_velocity_observed",
    "maximum_observed_absolute_pole_angular_velocity",
    "velocity_observation_at_clip_limit",
    "velocity_clip_limit_step_fraction",
    "tip_x_position",
    "tip_y_position",
    "tip_x_position_from_observation",
    "tip_y_position_from_observation",
    "tip_position_reconstruction_error",
    "minimum_tip_y_position",
    "maximum_tip_y_position",
    "tip_height_termination_threshold",
    "tip_height_margin",
    "minimum_tip_height_margin",
    "maximum_absolute_tip_x_position",
    "reward_target_tip_height",
    "maximum_physical_tip_height",
    "unavoidable_upright_distance_penalty",
    "pole1_unit_circle_error",
    "pole2_relative_unit_circle_error",
    "reward_survive",
    "reward_distance_penalty",
    "reward_velocity_penalty",
    "reward_from_public_terms",
    "cumulative_reward_survive",
    "cumulative_reward_distance_penalty",
    "cumulative_reward_velocity_penalty",
    "cumulative_return",
    "terminal_reason",
}


class InvertedDoublePendulumBenchmarkTests(unittest.TestCase):
    def test_config_controls_environment_identity_and_failure_scale(
        self,
    ) -> None:
        default = InvertedDoublePendulumBenchmark()
        configured = InvertedDoublePendulumBenchmark(
            InvertedDoublePendulumConfig(
                frame_skip=4,
                healthy_reward=20.0,
                reset_noise_scale=0.05,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/InvertedDoublePendulum-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertIsInstance(default.spec.action_space, dict)
        assert isinstance(default.spec.action_space, dict)
        self.assertEqual(default.spec.action_space["components"], ["cart_control"])
        self.assertEqual(default.spec.action_space["actuator_gears"], [500.0])
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 5,
                "model_timestep_seconds": 0.01,
                "seconds_per_step": 0.05,
                "actuator_gear": 500.0,
                "pole_length_meters": 0.6,
                "maximum_physical_tip_height": 1.2,
                "reward_target_tip_height": 2.0,
                "termination_tip_height": 1.0,
                "healthy_reward": 10.0,
                "reward_formula": (
                    "healthy_reward_if_tip_y>1-(0.01*tip_x^2+"
                    "(tip_y-2)^2)-(0.001*pole1_qvel^2+"
                    "0.005*pole2_relative_qvel^2)"
                ),
                "observation_velocity_clipping": [-10.0, 10.0],
                "reward_tip_position_source": "site_xpos",
                "observation_tip_reconstruction": (
                    "qpos trigonometry; may differ slightly from site_xpos "
                    "due to MuJoCo derived-geometry update timing"
                ),
                "reset_noise_scale": 0.1,
                "time_limit": 1000,
            },
        )
        self.assertEqual(
            configured.spec.environment_parameters,
            {
                "frame_skip": 4,
                "model_timestep_seconds": 0.01,
                "seconds_per_step": 0.04,
                "actuator_gear": 500.0,
                "pole_length_meters": 0.6,
                "maximum_physical_tip_height": 1.2,
                "reward_target_tip_height": 2.0,
                "termination_tip_height": 1.0,
                "healthy_reward": 20.0,
                "reward_formula": (
                    "healthy_reward_if_tip_y>1-(0.01*tip_x^2+"
                    "(tip_y-2)^2)-(0.001*pole1_qvel^2+"
                    "0.005*pole2_relative_qvel^2)"
                ),
                "observation_velocity_clipping": [-10.0, 10.0],
                "reward_tip_position_source": "site_xpos",
                "observation_tip_reconstruction": (
                    "qpos trigonometry; may differ slightly from site_xpos "
                    "due to MuJoCo derived-geometry update timing"
                ),
                "reset_noise_scale": 0.05,
                "time_limit": 1000,
            },
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            configured.spec.environment_digest,
        )
        self.assertEqual(
            configured.spec.metadata["failure_return"],
            -2000.0,
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            InvertedDoublePendulumConfig(
                frame_skip=5.0  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            InvertedDoublePendulumConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            InvertedDoublePendulumConfig(healthy_reward=10)
        for invalid in (-0.1, math.nan, math.inf, 1_000_001.0):
            with self.subTest(healthy_reward=invalid):
                with self.assertRaises(ValueError):
                    InvertedDoublePendulumConfig(healthy_reward=invalid)
        with self.assertRaises(ValueError):
            InvertedDoublePendulumConfig(reset_noise_scale=1.1)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = InvertedDoublePendulumBenchmark()

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

    def test_environment_is_deterministic_semantic_and_conformant(
        self,
    ) -> None:
        benchmark = InvertedDoublePendulumBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=([0.0], [0.25], [-0.25]),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), _OBSERVATION_FIELDS)
            self.assertTrue(all(type(value) is float for value in observation.values()))
            step = environment.step([0.0])
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(set(step.metrics), _METRIC_FIELDS)
            self.assertAlmostEqual(
                step.reward,
                _float(step.metrics["reward_survive"])
                + _float(step.metrics["reward_distance_penalty"])
                + _float(step.metrics["reward_velocity_penalty"]),
            )
            self.assertEqual(step.metrics["seconds_per_step"], 0.05)
            self.assertEqual(step.metrics["requested_cart_control"], 0.0)
            self.assertEqual(
                step.metrics["actuator_gear_scaled_cart_force"],
                0.0,
            )
            self.assertIsInstance(step.observation, dict)
            assert isinstance(step.observation, dict)
            self.assertAlmostEqual(
                _float(step.metrics["tip_y_position_from_observation"]),
                0.6
                * (
                    _float(step.observation["pole1_cos"])
                    + _float(step.observation["pole1_cos"])
                    * _float(step.observation["pole2_relative_cos"])
                    - _float(step.observation["pole1_sin"])
                    * _float(step.observation["pole2_relative_sin"])
                ),
            )
            self.assertEqual(step.metrics["reward_target_tip_height"], 2.0)
            self.assertEqual(step.metrics["maximum_physical_tip_height"], 1.2)
            self.assertAlmostEqual(
                _float(step.metrics["unavoidable_upright_distance_penalty"]),
                -0.64,
            )
        finally:
            environment.close()
            environment.close()

    def test_parameterized_environment_conforms(self) -> None:
        benchmark = InvertedDoublePendulumBenchmark(
            InvertedDoublePendulumConfig(
                frame_skip=4,
                healthy_reward=5.0,
                reset_noise_scale=0.05,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.1],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_real_control_is_published_with_gear_scaling(self) -> None:
        environment = InvertedDoublePendulumBenchmark().make_environment(
            EpisodeSpec(environment_seed=456)
        )
        try:
            environment.reset()
            step = environment.step([0.25])
            assert isinstance(step.metrics, dict)
            self.assertAlmostEqual(
                _float(step.metrics["requested_cart_control"]),
                0.25,
            )
            self.assertAlmostEqual(
                _float(step.metrics["actuator_gear_scaled_cart_force"]),
                125.0,
            )
            self.assertGreater(
                _float(step.metrics["tip_height_margin"]),
                0.0,
            )
            self.assertEqual(step.metrics["terminal_reason"], "none")
        finally:
            environment.close()

    def test_environment_requires_one_exact_bounded_float(self) -> None:
        benchmark = InvertedDoublePendulumBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            0.0,
            (0.0,),
            [],
            [0],
            [1.1],
            [math.nan],
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
            InvertedDoublePendulumBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"frame_skip": 4},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = InvertedDoublePendulumBenchmark()
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
        self.assertEqual(feedback.content["full_horizon_balances"], 0)

    def test_zero_force_baseline_publishes_reward_breakdown(self) -> None:
        benchmark = InvertedDoublePendulumBenchmark()
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
            "gymnasium/InvertedDoublePendulum-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertGreater(result.feedback.score, 0.0)
        self.assertLess(result.feedback.score, 9100.0)
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(result.feedback.content["fallen_episodes"], 1)
        self.assertIsInstance(
            result.feedback.content["mean_episode_minimum_tip_height_margin"],
            float,
        )
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episode = documents[0]
        transitions = tuple(document for document in documents if document["type"] == "transition")
        self.assertTrue(transitions)
        self.assertEqual(episode["outcome"], "fallen")
        self.assertEqual(
            set(transitions[0]["observation"]),
            _OBSERVATION_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0])
        self.assertEqual(
            set(transitions[0]["metrics"]),
            _METRIC_FIELDS,
        )

    def test_linear_feedback_improves_on_zero_force(self) -> None:
        benchmark = InvertedDoublePendulumBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_force: list[float] = []
        controlled: list[float] = []
        zero_steps: list[int] = []
        controlled_steps: list[int] = []

        for episode in episodes:
            zero_return, zero_episode_steps = _rollout(
                benchmark,
                episode,
                controlled=False,
            )
            controlled_return, controlled_episode_steps = _rollout(
                benchmark,
                episode,
                controlled=True,
            )
            zero_force.append(zero_return)
            controlled.append(controlled_return)
            zero_steps.append(zero_episode_steps)
            controlled_steps.append(controlled_episode_steps)

        self.assertGreater(
            statistics.fmean(controlled),
            statistics.fmean(zero_force),
        )
        self.assertGreater(
            statistics.fmean(controlled_steps),
            statistics.fmean(zero_steps),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {
        "cart_position": 0.0,
        "pole1_sin": 0.0,
        "pole2_relative_sin": 0.0,
        "pole1_cos": 1.0,
        "pole2_relative_cos": 1.0,
        "cart_velocity": 0.0,
        "pole1_angular_velocity": 0.0,
        "pole2_relative_angular_velocity": 0.0,
        "cart_constraint_force": 0.0,
    }


def _rollout(
    benchmark: InvertedDoublePendulumBenchmark,
    episode: EpisodeSpec,
    *,
    controlled: bool,
) -> tuple[float, int]:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        observation = environment.reset()
        result = None
        for _ in range(1000):
            assert isinstance(observation, dict)
            action: PolicyValue = _feedback_action(observation) if controlled else [0.0]
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    assert result is not None
    assert isinstance(result.metrics, dict)
    step_count = result.metrics["step_count"]
    assert type(step_count) is int
    return total, step_count


def _feedback_action(
    observation: dict[str, PolicyValue],
) -> list[PolicyValue]:
    values: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        assert type(value) is float
        values[key] = value
    force = (
        1.19583931 * values["cart_position"]
        - 5.99471097 * values["pole1_sin"]
        - 12.43115997 * values["pole2_relative_sin"]
        + 1.34462683 * values["cart_velocity"]
        + 1.44989414 * values["pole1_angular_velocity"]
        + 0.35466646 * values["pole2_relative_angular_velocity"]
    )
    return [max(-1.0, min(1.0, force))]


def _float(value: PolicyValue) -> float:
    assert type(value) is float
    return value


if __name__ == "__main__":
    unittest.main()
