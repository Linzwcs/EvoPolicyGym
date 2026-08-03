from __future__ import annotations

import json
import math
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from pendulum import PendulumBenchmark, baseline_program


class PendulumBenchmarkTests(unittest.TestCase):
    def test_spec_describes_the_supported_profile(self) -> None:
        spec = PendulumBenchmark().spec

        self.assertEqual(
            spec.id,
            "gymnasium/Pendulum-v1/mean-return-v1",
        )
        self.assertEqual(spec.max_episode_steps, 200)
        self.assertEqual(spec.primary_metric, "mean_return")
        self.assertEqual(spec.score_direction, "maximize")
        self.assertIsInstance(spec.observation_space, dict)
        self.assertIsInstance(spec.action_space, dict)
        assert isinstance(spec.observation_space, dict)
        assert isinstance(spec.action_space, dict)
        self.assertEqual(spec.observation_space["type"], "object")
        fields = spec.observation_space["fields"]
        self.assertIsInstance(fields, dict)
        assert isinstance(fields, dict)
        self.assertEqual(
            set(fields),
            {
                "cos_theta",
                "sin_theta",
                "theta_angular_velocity",
            },
        )
        self.assertEqual(spec.action_space["type"], "float")
        self.assertEqual(spec.action_space["minimum"], -2.0)
        self.assertEqual(spec.action_space["maximum"], 2.0)
        self.assertEqual(spec.metadata["failure_return"], -3300.0)
        self.assertIn("state before the action", spec.description)
        self.assertEqual(spec.observation_space["source_dtype"], "float32")
        self.assertEqual(spec.environment_parameters["seconds_per_step"], 0.05)
        self.assertEqual(
            spec.environment_parameters["natural_termination"],
            "none",
        )
        velocity = fields["theta_angular_velocity"]
        self.assertIsInstance(velocity, dict)
        assert isinstance(velocity, dict)
        self.assertEqual(velocity["unit"], "radians_per_second")

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = PendulumBenchmark()

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

    def test_environment_is_deterministic_and_rejects_invalid_actions(
        self,
    ) -> None:
        benchmark = PendulumBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(-2.0, 0.0, 2.0, 0.0),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        invalid_actions: tuple[PolicyValue, ...] = (
            -2.0001,
            2.0001,
            True,
            0,
            [0.0],
            float("inf"),
            float("nan"),
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
                environment.close()

        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=123, scenario={"gravity": 9.81})
            )

    def test_real_torque_reports_dynamics_and_reward_decomposition(self) -> None:
        environment = PendulumBenchmark().make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            assert isinstance(observation, dict)
            cosine = observation["cos_theta"]
            sine = observation["sin_theta"]
            velocity = observation["theta_angular_velocity"]
            assert type(cosine) is float
            assert type(sine) is float
            assert type(velocity) is float
            angle = math.atan2(sine, cosine)
            expected_cost = angle**2 + 0.1 * velocity**2 + 0.001 * 0.5**2
            result = environment.step(0.5)
        finally:
            environment.close()

        self.assertAlmostEqual(result.reward, -expected_cost, places=5)
        metrics = _metrics(result)
        self.assertEqual(
            _float_metric(metrics, "requested_torque_newton_meters"),
            0.5,
        )
        self.assertAlmostEqual(_float_metric(metrics, "angle_cost"), angle**2)
        self.assertAlmostEqual(
            _float_metric(metrics, "angular_velocity_cost"),
            0.1 * velocity**2,
        )
        self.assertAlmostEqual(_float_metric(metrics, "torque_cost"), 0.00025)
        self.assertAlmostEqual(
            _float_metric(metrics, "reward_from_public_terms"),
            result.reward,
            places=5,
        )

    def test_feedback_uses_explicit_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = PendulumBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "cos_theta": -1.0,
                "sin_theta": 0.0,
                "theta_angular_velocity": 0.0,
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -3300.0)
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
        self.assertEqual(feedback.content["failure_return"], -3300.0)

    def test_zero_torque_baseline_publishes_transition_trace(self) -> None:
        result = evaluate(
            baseline_program(),
            PendulumBenchmark(),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=2,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/Pendulum-v1/mean-return-v1",
        )
        self.assertLess(result.feedback.score, 0.0)
        self.assertGreater(result.feedback.score, -3300.0)
        self.assertEqual(
            tuple(episode.steps for episode in result.episodes),
            (200, 200),
        )
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
        self.assertEqual(trace.name, "trace.jsonl")
        self.assertEqual(trace.media_type, "application/x-ndjson")
        self.assertTrue(transitions)
        self.assertEqual(
            set(transitions[0]["observation"]),
            {
                "cos_theta",
                "sin_theta",
                "theta_angular_velocity",
            },
        )
        self.assertEqual(transitions[0]["action"], 0.0)
        self.assertEqual(
            transitions[0]["action_meaning"],
            "signed_torque_newton_meters",
        )
        self.assertLessEqual(transitions[0]["reward"], 0.0)
        self.assertEqual(
            set(transitions[0]["next_observation"]),
            {
                "cos_theta",
                "sin_theta",
                "theta_angular_velocity",
            },
        )
        self.assertEqual(transitions[-1]["metrics"]["terminal_reason"], "time_limit")
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(result.feedback.content["truncated_episodes"], 2)
        self.assertEqual(result.feedback.content["incomplete_episodes"], 0)
        self.assertEqual(result.feedback.content["mean_episode_torque_cost"], 0.0)
        angle_cost = result.feedback.content["mean_episode_angle_cost"]
        velocity_cost = result.feedback.content[
            "mean_episode_angular_velocity_cost"
        ]
        self.assertIsInstance(angle_cost, float)
        self.assertIsInstance(velocity_cost, float)
        assert isinstance(angle_cost, float)
        assert isinstance(velocity_cost, float)
        self.assertAlmostEqual(angle_cost + velocity_cost, -result.feedback.score, places=3)

    def test_energy_shaping_strategy_improves_on_the_baseline(self) -> None:
        benchmark = PendulumBenchmark()
        returns: list[float] = []

        for episode in benchmark.episodes("validation", seed=11, count=10):
            environment = benchmark.make_environment(episode)
            total_reward = 0.0
            try:
                observation = environment.reset()
                for _ in range(200):
                    assert isinstance(observation, dict)
                    cosine = observation["cos_theta"]
                    sine = observation["sin_theta"]
                    velocity = observation["theta_angular_velocity"]
                    assert type(cosine) is float
                    assert type(sine) is float
                    assert type(velocity) is float
                    angle = math.atan2(sine, cosine)
                    if cosine >= 0.8:
                        unbounded = -4.0 * angle - velocity
                    else:
                        energy = 0.5 * velocity * velocity + 15.0 * cosine
                        direction = 1.0 if velocity >= 0.0 else -1.0
                        unbounded = (15.0 - energy) * direction
                    action = max(-2.0, min(2.0, unbounded))
                    result = environment.step(action)
                    total_reward += result.reward
                    observation = result.observation
                    if result.terminated or result.truncated:
                        self.assertFalse(result.terminated)
                        self.assertTrue(result.truncated)
                        metrics = _metrics(result)
                        self.assertEqual(
                            _string_metric(metrics, "terminal_reason"),
                            "time_limit",
                        )
                        self.assertAlmostEqual(
                            _float_metric(metrics, "cumulative_return"),
                            total_reward,
                            places=8,
                        )
                        break
            finally:
                environment.close()
            returns.append(total_reward)

        self.assertEqual(len(returns), 10)
        self.assertGreater(sum(returns) / len(returns), -650.0)


def _metrics(step: Step) -> dict[str, PolicyValue]:
    if type(step.metrics) is not dict:
        raise AssertionError("expected object metrics")
    return step.metrics


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


if __name__ == "__main__":
    unittest.main()
