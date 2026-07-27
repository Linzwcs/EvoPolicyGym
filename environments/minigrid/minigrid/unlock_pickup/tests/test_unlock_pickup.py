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

from minigrid_unlock_pickup import UnlockPickupBenchmark, baseline_program


class UnlockPickupTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = UnlockPickupBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/UnlockPickup-v0/success-rate-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 288)
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )

    def test_environment_contract_and_invalid_action(self) -> None:
        benchmark = UnlockPickupBenchmark()
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
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()

    def test_scenario_and_feedback_privacy(self) -> None:
        with self.assertRaises(ValueError):
            UnlockPickupBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"size": 8})
            )
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = (
            UnlockPickupBenchmark()
            .feedback((failed,))
            .artifacts[0]
            .read_bytes()
        )
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_baseline_solves_task(self) -> None:
        result = evaluate(
            baseline_program(),
            UnlockPickupBenchmark(),
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
        for name in (
            "key_picked_up_rate",
            "door_opened_rate",
            "target_found_rate",
        ):
            self.assertEqual(result.feedback.content[name], 1.0)


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
