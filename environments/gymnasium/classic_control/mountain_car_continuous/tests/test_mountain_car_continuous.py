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

from mountain_car_continuous import (
    MountainCarContinuousBenchmark,
    baseline_program,
)


class MountainCarContinuousBenchmarkTests(unittest.TestCase):
    def test_spec_describes_the_supported_profile(self) -> None:
        spec = MountainCarContinuousBenchmark().spec

        self.assertEqual(
            spec.id,
            "gymnasium/MountainCarContinuous-v0/mean-return-v1",
        )
        self.assertEqual(spec.max_episode_steps, 999)
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
        self.assertEqual(set(fields), {"position", "velocity"})
        self.assertEqual(spec.action_space["type"], "float")
        self.assertEqual(spec.action_space["minimum"], -1.0)
        self.assertEqual(spec.action_space["maximum"], 1.0)
        self.assertEqual(spec.metadata["failure_return"], -100.0)
        self.assertIn("force*0.0015", spec.description)
        self.assertEqual(spec.observation_space["source_dtype"], "float32")
        self.assertEqual(
            spec.environment_parameters["action_cost_coefficient"],
            0.1,
        )
        self.assertEqual(
            spec.environment_parameters["goal_position_minimum"],
            0.45,
        )
        velocity = fields["velocity"]
        self.assertIsInstance(velocity, dict)
        assert isinstance(velocity, dict)
        self.assertEqual(velocity["unit"], "track_position_per_step")

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = MountainCarContinuousBenchmark()

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
        benchmark = MountainCarContinuousBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(-1.0, 0.0, 1.0, 0.0),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        invalid_actions: tuple[PolicyValue, ...] = (
            -1.0001,
            1.0001,
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
                EpisodeSpec(environment_seed=123, scenario={"goal_position": 0.4})
            )

    def test_real_force_reports_dynamics_and_reward_decomposition(self) -> None:
        environment = MountainCarContinuousBenchmark().make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            environment.reset()
            result = environment.step(0.5)
        finally:
            environment.close()

        self.assertAlmostEqual(result.reward, -0.025)
        metrics = _metrics(result)
        self.assertEqual(_float_metric(metrics, "requested_force"), 0.5)
        self.assertEqual(_string_metric(metrics, "force_direction"), "right")
        self.assertAlmostEqual(
            _float_metric(metrics, "engine_velocity_increment"),
            0.00075,
        )
        self.assertAlmostEqual(_float_metric(metrics, "control_cost"), 0.025)
        self.assertEqual(_float_metric(metrics, "goal_bonus"), 0.0)
        self.assertAlmostEqual(
            _float_metric(metrics, "reward_from_public_terms"),
            result.reward,
        )

    def test_feedback_uses_explicit_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = MountainCarContinuousBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "position": -0.5,
                "velocity": 0.0,
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -100.0)
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
        self.assertEqual(feedback.content["failure_return"], -100.0)

    def test_zero_force_baseline_publishes_transition_trace(self) -> None:
        result = evaluate(
            baseline_program(),
            MountainCarContinuousBenchmark(),
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
            "gymnasium/MountainCarContinuous-v0/mean-return-v1",
        )
        self.assertEqual(result.feedback.score, 0.0)
        self.assertEqual(result.episodes[0].steps, 999)
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
            {"position", "velocity"},
        )
        self.assertEqual(transitions[0]["action"], 0.0)
        self.assertEqual(
            transitions[0]["action_meaning"],
            "signed_force_left_negative_right_positive",
        )
        self.assertEqual(transitions[0]["reward"], 0.0)
        self.assertEqual(
            set(transitions[0]["next_observation"]),
            {"position", "velocity"},
        )
        self.assertEqual(transitions[-1]["metrics"]["terminal_reason"], "time_limit")
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(result.feedback.content["time_limit_episodes"], 1)
        self.assertEqual(result.feedback.content["mean_episode_control_cost"], 0.0)
        self.assertEqual(result.feedback.content["mean_episode_absolute_force"], 0.0)
        maximum_position = result.feedback.content["maximum_position_reached"]
        self.assertIsInstance(maximum_position, float)
        assert isinstance(maximum_position, float)
        self.assertLess(maximum_position, 0.45)

    def test_velocity_direction_strategy_improves_on_the_baseline(
        self,
    ) -> None:
        benchmark = MountainCarContinuousBenchmark()
        steps_to_goal: list[int] = []
        returns: list[float] = []

        for episode in benchmark.episodes("validation", seed=11, count=10):
            environment = benchmark.make_environment(episode)
            total_reward = 0.0
            try:
                observation = environment.reset()
                for step_index in range(1, 1000):
                    assert isinstance(observation, dict)
                    velocity = observation["velocity"]
                    assert type(velocity) is float
                    action = 1.0 if velocity >= 0.0 else -1.0
                    result = environment.step(action)
                    total_reward += result.reward
                    observation = result.observation
                    if result.terminated or result.truncated:
                        steps_to_goal.append(step_index)
                        returns.append(total_reward)
                        self.assertTrue(result.terminated)
                        metrics = _metrics(result)
                        self.assertTrue(_bool_metric(metrics, "goal_reached"))
                        self.assertEqual(
                            _string_metric(metrics, "terminal_reason"),
                            "goal_reached",
                        )
                        self.assertEqual(_float_metric(metrics, "goal_bonus"), 100.0)
                        self.assertAlmostEqual(
                            _float_metric(metrics, "control_cost"),
                            0.1,
                        )
                        self.assertAlmostEqual(result.reward, 99.9)
                        self.assertEqual(
                            _float_metric(metrics, "distance_to_goal_position"),
                            0.0,
                        )
                        break
            finally:
                environment.close()

        self.assertEqual(len(steps_to_goal), 10)
        self.assertLess(sum(steps_to_goal) / len(steps_to_goal), 120.0)
        self.assertGreater(sum(returns) / len(returns), 88.0)


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


def _bool_metric(metrics: dict[str, PolicyValue], name: str) -> bool:
    value = metrics.get(name)
    if type(value) is not bool:
        raise AssertionError(f"expected bool metric {name}")
    return value


if __name__ == "__main__":
    unittest.main()
