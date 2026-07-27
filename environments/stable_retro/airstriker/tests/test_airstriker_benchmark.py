from __future__ import annotations

import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.policy import TensorValue

from airstriker import AirstrikerBenchmark, AirstrikerConfig, baseline_program


class AirstrikerBenchmarkTests(unittest.TestCase):
    def test_environment_resets_and_steps(self) -> None:
        environment = AirstrikerBenchmark().make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, TensorValue)
            assert isinstance(observation, TensorValue)
            self.assertEqual(observation.shape, (224, 320, 3))
            step = environment.step(0)
            self.assertIsInstance(step.reward, float)
        finally:
            environment.close()
            environment.close()

    def test_non_portable_game_and_invalid_action_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AirstrikerConfig(game="SonicTheHedgehog-Genesis-v0")
        environment = AirstrikerBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            for invalid in (True, -1, 126):
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
        finally:
            environment.close()

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            AirstrikerBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    (0,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
