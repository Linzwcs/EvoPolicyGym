from __future__ import annotations

import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue, TensorValue

from highway_benchmarks import (
    HIGHWAY_PROFILES,
    HighwayBenchmark,
    HighwayConfig,
    baseline_program,
)


class HighwayBenchmarkTests(unittest.TestCase):
    def test_all_profiles_reset_and_take_one_strict_action(self) -> None:
        self.assertEqual(len(HIGHWAY_PROFILES), 10)
        for profile in HIGHWAY_PROFILES:
            with self.subTest(profile=profile):
                config = HighwayConfig(profile=profile)
                benchmark = HighwayBenchmark(config)
                environment = benchmark.make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    observation = environment.reset()
                    self.assertTrue(
                        type(observation) in {TensorValue, dict}
                    )
                    action: PolicyValue = (
                        [0.0] * config.action_size
                        if config.continuous
                        else 1
                    )
                    step = environment.step(action)
                    self.assertIsInstance(step.reward, float)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_public_identity(self) -> None:
        highway = HighwayBenchmark()
        parking = HighwayBenchmark(HighwayConfig(profile="parking"))
        self.assertNotEqual(
            highway.spec.environment_digest,
            parking.spec.environment_digest,
        )
        self.assertEqual(
            parking.spec.environment_parameters["profile"],
            "parking",
        )
        self.assertEqual(parking.spec.max_episode_steps, 500)

    def test_invalid_profile_and_actions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HighwayConfig(profile="unknown")
        with self.assertRaises(TypeError):
            HighwayConfig(profile=1)  # type: ignore[arg-type]

        environment = HighwayBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

        environment = HighwayBenchmark(
            HighwayConfig(profile="parking")
        ).make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([0, 0])
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_profile(self) -> None:
        with self.assertRaises(ValueError):
            HighwayBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "parking"},
                )
            )

    def test_baseline_is_packaged(self) -> None:
        program = baseline_program()
        self.assertIn("policy.py", program.files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            HighwayBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    (1,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
