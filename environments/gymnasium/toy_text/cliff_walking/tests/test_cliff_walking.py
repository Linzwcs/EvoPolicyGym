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

from cliff_walking import (
    CliffWalkingBenchmark,
    CliffWalkingConfig,
    baseline_program,
)


class CliffWalkingBenchmarkTests(unittest.TestCase):
    def test_config_is_typed_and_changes_environment_identity(self) -> None:
        dry = CliffWalkingBenchmark()
        slippery = CliffWalkingBenchmark(
            CliffWalkingConfig(is_slippery=True)
        )

        self.assertEqual(
            dry.spec.id,
            "gymnasium/CliffWalking-v1/mean-return-v1",
        )
        self.assertEqual(dry.spec.max_episode_steps, 200)
        self.assertEqual(dry.spec.primary_metric, "mean_return")
        self.assertEqual(
            dry.spec.environment_parameters,
            {"is_slippery": False},
        )
        self.assertEqual(
            slippery.spec.environment_parameters,
            {"is_slippery": True},
        )
        self.assertNotEqual(
            dry.spec.environment_digest,
            slippery.spec.environment_digest,
        )

    def test_config_rejects_ambiguous_boolean_values(self) -> None:
        with self.assertRaises(TypeError):
            CliffWalkingConfig(is_slippery=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CliffWalkingBenchmark(config=object())  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = CliffWalkingBenchmark()

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

    def test_safe_path_reaches_goal_and_conformance_is_deterministic(
        self,
    ) -> None:
        benchmark = CliffWalkingBenchmark()
        safe_path = (0,) + (1,) * 11 + (2,)
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=safe_path,
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertEqual(
                observation,
                {
                    "state": 36,
                    "row": 3,
                    "column": 0,
                    "tile": "start",
                },
            )
            total = 0.0
            result = None
            for action in safe_path:
                result = environment.step(action)
                total += result.reward
            assert result is not None
            self.assertTrue(result.terminated)
            self.assertFalse(result.truncated)
            self.assertIsInstance(result.observation, dict)
            assert isinstance(result.observation, dict)
            self.assertEqual(result.observation["tile"], "goal")
            self.assertEqual(total, -13.0)
        finally:
            environment.close()

        slippery = CliffWalkingBenchmark(
            CliffWalkingConfig(is_slippery=True)
        )
        report = check_benchmark(
            slippery,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=(0, 1, 0, 1, 2, 3),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_cliff_resets_to_start_and_actions_are_strict(self) -> None:
        benchmark = CliffWalkingBenchmark()
        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            environment.reset()
            result = environment.step(1)
            self.assertEqual(result.reward, -100.0)
            self.assertFalse(result.done)
            self.assertEqual(
                result.observation,
                {
                    "state": 36,
                    "row": 3,
                    "column": 0,
                    "tile": "start",
                },
            )
        finally:
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            4,
            True,
            0.0,
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

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            CliffWalkingBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"is_slippery": True},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = CliffWalkingBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "state": 36,
                "row": 3,
                "column": 0,
                "tile": "start",
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -20000.0)
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
        self.assertEqual(feedback.content["failure_return"], -20000.0)

    def test_weak_baseline_truncates_and_publishes_complete_trace(
        self,
    ) -> None:
        benchmark = CliffWalkingBenchmark()
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
            "gymnasium/CliffWalking-v1/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertEqual(result.feedback.score, -20000.0)
        self.assertEqual(result.episodes[0].steps, 200)
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
        self.assertEqual(len(transitions), 200)
        self.assertEqual(transitions[0]["action"], 1)
        self.assertEqual(transitions[0]["reward"], -100.0)
        self.assertFalse(transitions[0]["truncated"])
        self.assertTrue(transitions[-1]["truncated"])


if __name__ == "__main__":
    unittest.main()
