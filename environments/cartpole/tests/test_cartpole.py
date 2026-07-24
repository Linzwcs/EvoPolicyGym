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
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution

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
        self.assertEqual(transitions[0]["reward"], 1.0)
        self.assertEqual(len(transitions[0]["next_observation"]), 4)

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

        self.assertGreater(result.feedback.score, 50.0)


if __name__ == "__main__":
    unittest.main()
