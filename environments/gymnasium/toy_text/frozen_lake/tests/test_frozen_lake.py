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
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from frozen_lake import (
    FrozenLakeBenchmark,
    FrozenLakeConfig,
    baseline_program,
)


class FrozenLakeBenchmarkTests(unittest.TestCase):
    def test_config_is_typed_and_changes_environment_identity(self) -> None:
        default = FrozenLakeBenchmark()
        deterministic = FrozenLakeBenchmark(
            FrozenLakeConfig(is_slippery=False)
        )
        large = FrozenLakeBenchmark(
            FrozenLakeConfig(map_name="8x8")
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/FrozenLake-v1/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 100)
        self.assertEqual(large.spec.max_episode_steps, 200)
        self.assertNotEqual(
            default.spec.environment_digest,
            deterministic.spec.environment_digest,
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            large.spec.environment_digest,
        )
        self.assertEqual(
            deterministic.spec.environment_parameters,
            {
                "map_name": "4x4",
                "map": [
                    "SFFF",
                    "FHFH",
                    "FFFH",
                    "HFFG",
                ],
                "is_slippery": False,
                "success_rate": 1.0 / 3.0,
                "reward_schedule": {
                    "goal": 1.0,
                    "hole": 0.0,
                    "frozen": 0.0,
                },
            },
        )

        exposed_map = deterministic.spec.environment_parameters["map"]
        self.assertIsInstance(exposed_map, list)
        assert isinstance(exposed_map, list)
        exposed_map.append("modified")
        fresh_map = deterministic.spec.environment_parameters["map"]
        self.assertIsInstance(fresh_map, list)
        assert isinstance(fresh_map, list)
        self.assertEqual(len(fresh_map), 4)

    def test_config_rejects_unsupported_or_ambiguous_values(self) -> None:
        with self.assertRaises(ValueError):
            FrozenLakeConfig(map_name="random")
        with self.assertRaises(TypeError):
            FrozenLakeConfig(is_slippery=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            FrozenLakeConfig(success_rate=1)
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(success_rate=invalid):
                with self.assertRaises(ValueError):
                    FrozenLakeConfig(success_rate=invalid)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = FrozenLakeBenchmark()

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

    def test_deterministic_profile_reaches_goal_and_rejects_invalid_actions(
        self,
    ) -> None:
        benchmark = FrozenLakeBenchmark(
            FrozenLakeConfig(is_slippery=False)
        )
        safe_path = (1, 1, 2, 1, 2, 2)
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
                    "state": 0,
                    "row": 0,
                    "column": 0,
                    "tile": "S",
                },
            )
            result = None
            for action in safe_path:
                result = environment.step(action)
            assert result is not None
            self.assertTrue(result.terminated)
            self.assertFalse(result.truncated)
            self.assertEqual(result.reward, 1.0)
            self.assertIsInstance(result.observation, dict)
            assert isinstance(result.observation, dict)
            self.assertEqual(result.observation["tile"], "G")
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            4,
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

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        benchmark = FrozenLakeBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"map_name": "8x8"},
                )
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = FrozenLakeBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "state": 0,
                "row": 0,
                "column": 0,
                "tile": "S",
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
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
        self.assertEqual(feedback.content["successful_episodes"], 0)

    def test_baseline_consumes_public_parameters_and_publishes_trace(
        self,
    ) -> None:
        benchmark = FrozenLakeBenchmark(
            FrozenLakeConfig(
                map_name="8x8",
                is_slippery=False,
                success_rate=0.75,
            )
        )
        result = evaluate(
            baseline_program(),
            benchmark,
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
            "gymnasium/FrozenLake-v1/success-rate-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertEqual(result.feedback.score, 1.0)
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
            {"state", "row", "column", "tile"},
        )
        self.assertIn(transitions[0]["action"], {0, 1, 2, 3})
        self.assertIn(transitions[0]["reward"], {0.0, 1.0})
        self.assertEqual(
            set(transitions[0]["next_observation"]),
            {"state", "row", "column", "tile"},
        )


if __name__ == "__main__":
    unittest.main()
