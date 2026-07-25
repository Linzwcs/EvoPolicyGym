from __future__ import annotations

import json
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
        self.assertEqual(transitions[0]["reward"], -1.0)
        self.assertEqual(len(transitions[0]["next_observation"]), 6)


if __name__ == "__main__":
    unittest.main()
