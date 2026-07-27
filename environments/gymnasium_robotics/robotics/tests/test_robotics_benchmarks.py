from __future__ import annotations

import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)

from robotics_benchmarks import (
    ROBOTICS_PROFILES,
    RoboticsBenchmark,
    RoboticsConfig,
    baseline_program,
)


class RoboticsBenchmarkTests(unittest.TestCase):
    def test_all_profiles_reset_and_take_one_strict_action(self) -> None:
        self.assertEqual(len(ROBOTICS_PROFILES), 21)
        for profile in ROBOTICS_PROFILES:
            with self.subTest(profile=profile):
                config = RoboticsConfig(profile=profile)
                benchmark = RoboticsBenchmark(config)
                environment = benchmark.make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    observation = environment.reset()
                    self.assertTrue(type(observation) in {dict} or hasattr(
                        observation,
                        "shape",
                    ))
                    step = environment.step([0.0] * config.action_size)
                    self.assertIsInstance(step.reward, float)
                    self.assertIsInstance(step.metrics, dict)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_public_identity(self) -> None:
        fetch = RoboticsBenchmark()
        kitchen = RoboticsBenchmark(
            RoboticsConfig(profile="franka-kitchen")
        )
        self.assertNotEqual(
            fetch.spec.environment_digest,
            kitchen.spec.environment_digest,
        )
        self.assertEqual(
            kitchen.spec.environment_parameters["profile"],
            "franka-kitchen",
        )
        self.assertEqual(kitchen.spec.max_episode_steps, 280)

    def test_invalid_profile_and_action_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RoboticsConfig(profile="unknown")
        with self.assertRaises(TypeError):
            RoboticsConfig(profile=1)  # type: ignore[arg-type]

        environment = RoboticsBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([0, 0, 0, 0])
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_profile(self) -> None:
        with self.assertRaises(ValueError):
            RoboticsBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "ant-maze"},
                )
            )

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            RoboticsBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    ([0.0, 0.0, 0.0, 0.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
