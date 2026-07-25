from __future__ import annotations

import json
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

from blackjack import (
    BlackjackBenchmark,
    BlackjackConfig,
    baseline_program,
)


class BlackjackBenchmarkTests(unittest.TestCase):
    def test_config_matches_standard_registration_and_changes_identity(
        self,
    ) -> None:
        standard = BlackjackBenchmark()
        casino = BlackjackBenchmark(
            BlackjackConfig(natural=True, sab=False)
        )

        self.assertEqual(
            standard.spec.id,
            "gymnasium/Blackjack-v1/mean-reward-v1",
        )
        self.assertEqual(standard.spec.max_episode_steps, 32)
        self.assertEqual(standard.spec.primary_metric, "mean_reward")
        self.assertEqual(
            standard.spec.environment_parameters,
            {
                "natural": False,
                "sab": True,
            },
        )
        self.assertNotEqual(
            standard.spec.environment_digest,
            casino.spec.environment_digest,
        )
        self.assertEqual(
            casino.spec.metadata["reward_schedule"],
            {
                "win": 1.0,
                "natural_win": 1.5,
                "draw": 0.0,
                "loss": -1.0,
            },
        )

    def test_config_rejects_ambiguous_boolean_values(self) -> None:
        with self.assertRaises(TypeError):
            BlackjackConfig(natural=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BlackjackConfig(sab=0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BlackjackBenchmark(config=object())  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = BlackjackBenchmark()

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
        benchmark = BlackjackBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(1, 0),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(
                set(observation),
                {"player_sum", "dealer_showing", "usable_ace"},
            )
            self.assertIsInstance(observation["player_sum"], int)
            self.assertIsInstance(observation["dealer_showing"], int)
            self.assertIsInstance(observation["usable_ace"], bool)
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            2,
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
            BlackjackBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"natural": True},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = BlackjackBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "player_sum": 16,
                "dealer_showing": 10,
                "usable_ace": False,
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -1.0)
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
        self.assertEqual(feedback.content["failure_return"], -1.0)

    def test_always_stick_baseline_publishes_complete_trace(self) -> None:
        benchmark = BlackjackBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=8,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/Blackjack-v1/mean-reward-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertEqual(
            tuple(episode.steps for episode in result.episodes),
            (1,) * 8,
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
        self.assertEqual(len(transitions), 8)
        self.assertEqual(
            set(transitions[0]["observation"]),
            {"player_sum", "dealer_showing", "usable_ace"},
        )
        self.assertEqual(transitions[0]["action"], 0)
        self.assertIn(transitions[0]["reward"], {-1.0, 0.0, 1.0})

    def test_threshold_strategy_improves_on_always_stick(self) -> None:
        benchmark = BlackjackBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=512,
        )
        always_stick: list[float] = []
        threshold: list[float] = []

        for episode in episodes:
            environment = benchmark.make_environment(episode)
            try:
                environment.reset()
                always_stick.append(environment.step(0).reward)
            finally:
                environment.close()

            environment = benchmark.make_environment(episode)
            try:
                observation = environment.reset()
                total = 0.0
                for _ in range(32):
                    assert isinstance(observation, dict)
                    player_sum = observation["player_sum"]
                    assert type(player_sum) is int
                    result = environment.step(
                        1 if player_sum < 17 else 0
                    )
                    total += result.reward
                    observation = result.observation
                    if result.done:
                        break
                threshold.append(total)
            finally:
                environment.close()

        self.assertGreater(
            statistics.fmean(threshold),
            statistics.fmean(always_stick),
        )


if __name__ == "__main__":
    unittest.main()
