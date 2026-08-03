from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym import EvaluationConfig, Program, evaluate
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

from cartpole import CartPoleBenchmark, baseline_program

_HEURISTIC_POLICY = """\
class HeuristicPolicy:
    def act(self, observation):
        angle = float(observation[2])
        angular_velocity = float(observation[3])
        return 1 if angle + 0.25 * angular_velocity > 0.0 else 0


def make_policy(context):
    del context
    return HeuristicPolicy()
"""


class CartPoleBenchmarkTests(unittest.TestCase):
    def test_spec_publishes_units_limits_dynamics_and_reward(self) -> None:
        spec = CartPoleBenchmark().spec
        self.assertIn("return therefore equals survived steps", spec.description)
        observation_space = _object_value(spec.observation_space)
        self.assertEqual(observation_space["policy_carrier"], "list[float]")
        self.assertEqual(observation_space["source_dtype"], "float32")
        meanings = _object_value(observation_space["component_meanings"])
        self.assertIn("meters per second", _string_value(meanings["cart_velocity"]))
        self.assertIn("radians per second", _string_value(meanings["pole_angular_velocity"]))
        parameters = spec.environment_parameters
        self.assertEqual(parameters["cart_position_limit_meters"], 2.4)
        self.assertAlmostEqual(
            _float_value(parameters["pole_angle_limit_radians"]),
            0.20943951023931953,
        )
        self.assertEqual(parameters["force_magnitude_newtons"], 10.0)
        self.assertEqual(parameters["seconds_per_step"], 0.02)
        self.assertEqual(parameters["reward_per_step_including_termination"], 1.0)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = CartPoleBenchmark()

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
        benchmark = CartPoleBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 0),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(2)
        finally:
            environment.close()
            environment.close()

        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=123, scenario={"low": -0.1})
            )

    def test_real_failure_feedback_reports_crossed_limit_and_margins(self) -> None:
        environment = CartPoleBenchmark().make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            environment.reset()
            for _ in range(500):
                result = environment.step(0)
                if result.done:
                    break
        finally:
            environment.close()
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.reward, 1.0)
        metrics = _metrics(result)
        reason = _string_metric(metrics, "terminal_reason")
        self.assertTrue(
            "cart_position_limit" in reason or "pole_angle_limit" in reason
        )
        self.assertGreater(_int_metric(metrics, "step_count"), 0)
        self.assertLess(_float_metric(metrics, "survival_fraction"), 1.0)
        self.assertFalse(_bool_metric(metrics, "balanced_within_limits"))
        self.assertIn(_string_metric(metrics, "requested_action"), {"push_left", "push_right"})

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = CartPoleBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=[0.0, 0.0, 0.0, 0.0],
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
        self.assertEqual(len(feedback.artifacts), 1)
        self.assertEqual(feedback.artifacts[0].name, "trace.jsonl")
        self.assertNotIn(b"environment_seed", feedback.artifacts[0].read_bytes())
        self.assertNotIn(b"policy_seed", feedback.artifacts[0].read_bytes())
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)

    def test_baseline_evaluation_publishes_transition_trace(self) -> None:
        result = evaluate(
            baseline_program(),
            CartPoleBenchmark(),
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
            "gymnasium/CartPole-v1/mean-return-v1",
        )
        self.assertGreater(result.feedback.score, 0.0)
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
        self.assertEqual(len(transitions[0]["observation"]), 4)
        self.assertIn(transitions[0]["action"], {0, 1})
        self.assertIn(transitions[0]["action_meaning"], {"push_left", "push_right"})
        self.assertEqual(transitions[0]["reward"], 1.0)
        self.assertEqual(len(transitions[0]["next_observation"]), 4)
        self.assertIn(
            transitions[-1]["metrics"]["terminal_reason"],
            {"cart_position_limit", "pole_angle_limit", "cart_position_limit+pole_angle_limit"},
        )
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        cart_limits = _int_value(
            result.feedback.content["cart_position_limit_episodes"]
        )
        pole_limits = _int_value(
            result.feedback.content["pole_angle_limit_episodes"]
        )
        self.assertGreaterEqual(cart_limits + pole_limits, 2)

    def test_simple_heuristic_improves_on_the_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "program"
            source.mkdir()
            (source / "policy.py").write_text(
                _HEURISTIC_POLICY,
                encoding="utf-8",
            )
            program = Program.from_directory(source)

            result = evaluate(
                program,
                CartPoleBenchmark(),
                execution=ProcessExecution.unsafe(),
                config=EvaluationConfig(
                    split="train",
                    episodes=3,
                    seed=0,
                    episode_timeout_seconds=10,
                ),
            )

        self.assertEqual(result.feedback.score, 500.0)
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(result.feedback.content["time_limit_successes"], 3)


def _metrics(step: Step) -> dict[str, PolicyValue]:
    return _object_value(step.metrics)


def _object_value(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise AssertionError("expected object PolicyValue")
    return value


def _string_value(value: PolicyValue) -> str:
    if type(value) is not str:
        raise AssertionError("expected string PolicyValue")
    return value


def _float_value(value: PolicyValue) -> float:
    if type(value) is not float:
        raise AssertionError("expected float PolicyValue")
    return value


def _int_value(value: PolicyValue) -> int:
    if type(value) is not int:
        raise AssertionError("expected integer PolicyValue")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    return _string_value(metrics.get(name))


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    return _float_value(metrics.get(name))


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    return _int_value(metrics.get(name))


def _bool_metric(metrics: dict[str, PolicyValue], name: str) -> bool:
    value = metrics.get(name)
    if type(value) is not bool:
        raise AssertionError(f"expected bool metric {name}")
    return value


if __name__ == "__main__":
    unittest.main()
