from __future__ import annotations

import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue

from vizdoom_benchmarks import (
    VIZDOOM_PROFILES,
    ViZDoomBenchmark,
    ViZDoomConfig,
    baseline_program,
)


class ViZDoomBenchmarkTests(unittest.TestCase):
    def test_all_bundled_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(VIZDOOM_PROFILES), 12)
        for profile in VIZDOOM_PROFILES:
            with self.subTest(profile=profile):
                config = ViZDoomConfig(profile=profile)
                environment = ViZDoomBenchmark(
                    config
                ).make_environment(EpisodeSpec(environment_seed=123))
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    action: PolicyValue = (
                        {
                            "binary": 0,
                            "continuous": [0.0, 0.0, 0.0],
                        }
                        if config.hybrid_action
                        else 0
                    )
                    step = environment.step(action)
                    self.assertIsInstance(step.reward, float)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_environment_identity(self) -> None:
        basic = ViZDoomBenchmark()
        audio = ViZDoomBenchmark(ViZDoomConfig(profile="basic-audio"))
        self.assertNotEqual(
            basic.spec.environment_digest,
            audio.spec.environment_digest,
        )
        self.assertEqual(audio.spec.max_episode_steps, 300)

    def test_invalid_actions_are_rejected(self) -> None:
        environment = ViZDoomBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

        environment = ViZDoomBenchmark(
            ViZDoomConfig(profile="deathmatch")
        ).make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(
                    {"binary": 0, "continuous": [0, 0, 0]}
                )
        finally:
            environment.close()

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            ViZDoomBenchmark(),
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
