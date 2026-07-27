from __future__ import annotations

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
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_blocked_unlock_pickup import (
    BlockedUnlockPickupBenchmark,
    baseline_program,
)


class BlockedUnlockPickupTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/BlockedUnlockPickup-v0/success-rate-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 576)
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertEqual(train, repeated)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )

    def test_environment_contract_and_invalid_actions(self) -> None:
        benchmark = BlockedUnlockPickupBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 5, 6),
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
            self.assertIsInstance(observation["image"], TensorValue)
            self.assertTrue(
                str(observation["mission"]).startswith("pick up the ")
            )
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_scenario_override_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BlockedUnlockPickupBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"room_size": 8})
            )

    def test_feedback_keeps_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = BlockedUnlockPickupBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"scenario"', trace)

    def test_baseline_completes_public_progress_ladder(self) -> None:
        result = evaluate(
            baseline_program(),
            BlockedUnlockPickupBenchmark(),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=12,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )
        self.assertEqual(result.feedback.score, 1.0)
        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        for field in (
            "blocker_moved_rate",
            "key_picked_up_rate",
            "door_opened_rate",
            "target_found_rate",
        ):
            self.assertEqual(result.feedback.content[field], 1.0)
        self.assertEqual(
            result.feedback.artifacts[0].name,
            "trace.jsonl",
        )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "pick up the purple box",
    }


if __name__ == "__main__":
    unittest.main()
