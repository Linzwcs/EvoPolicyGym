from __future__ import annotations

import json
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

from acrobot import AcrobotBenchmark, baseline_program


class AcrobotBenchmarkTests(unittest.TestCase):
    def test_spec_describes_the_supported_profile(self) -> None:
        spec = AcrobotBenchmark().spec

        self.assertEqual(
            spec.id,
            "gymnasium/Acrobot-v1/mean-return-v1",
        )
        self.assertEqual(spec.max_episode_steps, 500)
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
                "cos_theta_1",
                "sin_theta_1",
                "cos_theta_2",
                "sin_theta_2",
                "theta_1_angular_velocity",
                "theta_2_angular_velocity",
            },
        )
        self.assertEqual(spec.action_space["values"], [0, 1, 2])
        self.assertEqual(spec.metadata["failure_return"], -500.0)
        self.assertIn("-cos(theta1)-cos(theta1+theta2)", spec.description)
        self.assertEqual(spec.environment_parameters["seconds_per_step"], 0.2)
        self.assertEqual(
            spec.environment_parameters["available_torques_newton_meters"],
            [-1.0, 0.0, 1.0],
        )
        self.assertEqual(
            spec.environment_parameters[
                "target_height_strictly_greater_than_meters"
            ],
            1.0,
        )
        theta_2 = fields["cos_theta_2"]
        self.assertIsInstance(theta_2, dict)
        assert isinstance(theta_2, dict)
        theta_2_meaning = theta_2["meaning"]
        self.assertIsInstance(theta_2_meaning, str)
        assert isinstance(theta_2_meaning, str)
        self.assertIn("relative to link 1", theta_2_meaning)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = AcrobotBenchmark()

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
        benchmark = AcrobotBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 1),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            3,
            True,
            1.0,
            [1],
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
                EpisodeSpec(environment_seed=123, scenario={"dynamics": "nips"})
            )

    def test_real_energy_pumping_reaches_target_with_explicit_progress(self) -> None:
        environment = AcrobotBenchmark().make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            for _ in range(500):
                self.assertIsInstance(observation, dict)
                assert isinstance(observation, dict)
                angular_velocity = observation["theta_2_angular_velocity"]
                if type(angular_velocity) is not float:
                    raise AssertionError("expected angular velocity")
                result = environment.step(2 if angular_velocity >= 0.0 else 0)
                observation = result.observation
                if result.done:
                    break
        finally:
            environment.close()
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.reward, 0.0)
        metrics = _metrics(result)
        self.assertEqual(
            _string_metric(metrics, "terminal_reason"),
            "target_height_reached",
        )
        self.assertGreater(
            _float_metric(metrics, "free_end_vertical_height_meters"),
            1.0,
        )
        self.assertGreater(
            _float_metric(metrics, "target_height_margin_meters"),
            0.0,
        )
        self.assertEqual(
            _float_metric(metrics, "height_remaining_to_target_meters"),
            0.0,
        )
        self.assertTrue(
            _bool_metric(metrics, "target_reached_from_public_observation")
        )
        self.assertLess(_int_metric(metrics, "step_count"), 500)

    def test_feedback_uses_explicit_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = AcrobotBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "cos_theta_1": 1.0,
                "sin_theta_1": 0.0,
                "cos_theta_2": 1.0,
                "sin_theta_2": 0.0,
                "theta_1_angular_velocity": 0.0,
                "theta_2_angular_velocity": 0.0,
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -500.0)
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
        self.assertEqual(feedback.content["failure_return"], -500.0)

    def test_zero_torque_baseline_publishes_transition_trace(self) -> None:
        result = evaluate(
            baseline_program(),
            AcrobotBenchmark(),
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
            "gymnasium/Acrobot-v1/mean-return-v1",
        )
        self.assertEqual(result.feedback.score, -500.0)
        self.assertEqual(tuple(episode.steps for episode in result.episodes), (500, 500))
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
                "cos_theta_1",
                "sin_theta_1",
                "cos_theta_2",
                "sin_theta_2",
                "theta_1_angular_velocity",
                "theta_2_angular_velocity",
            },
        )
        self.assertEqual(transitions[0]["action"], 1)
        self.assertEqual(transitions[0]["action_meaning"], "zero_torque")
        self.assertEqual(transitions[0]["reward"], -1.0)
        self.assertEqual(len(transitions[0]["next_observation"]), 6)
        self.assertEqual(transitions[-1]["metrics"]["terminal_reason"], "time_limit")
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(result.feedback.content["time_limit_episodes"], 2)
        maximum_height = result.feedback.content["maximum_tip_height_meters"]
        self.assertIsInstance(maximum_height, float)
        assert isinstance(maximum_height, float)
        self.assertLess(maximum_height, 1.0)


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


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise AssertionError(f"expected integer metric {name}")
    return value


def _bool_metric(metrics: dict[str, PolicyValue], name: str) -> bool:
    value = metrics.get(name)
    if type(value) is not bool:
        raise AssertionError(f"expected bool metric {name}")
    return value


if __name__ == "__main__":
    unittest.main()
