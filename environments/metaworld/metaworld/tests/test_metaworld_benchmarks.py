from __future__ import annotations

import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.policy import TensorValue

from metaworld_benchmarks import (
    METAWORLD_MT1_PROFILES,
    MetaWorldBenchmark,
    MetaWorldConfig,
    baseline_program,
)


class MetaWorldBenchmarkTests(unittest.TestCase):
    def test_all_fifty_mt1_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(METAWORLD_MT1_PROFILES), 50)
        for profile in METAWORLD_MT1_PROFILES:
            with self.subTest(profile=profile):
                benchmark = MetaWorldBenchmark(
                    MetaWorldConfig(profile=profile)
                )
                environment = benchmark.make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, TensorValue)
                    step = environment.step([0.0, 0.0, 0.0, 0.0])
                    self.assertIsInstance(step.reward, float)
                    self.assertIsInstance(step.metrics, dict)
                finally:
                    environment.close()
                    environment.close()

    def test_collection_profiles_have_public_one_hot_tasks(self) -> None:
        configs = (
            MetaWorldConfig(profile="mt10"),
            MetaWorldConfig(profile="mt50"),
            MetaWorldConfig(
                profile="custom",
                custom_tasks=("reach-v3", "push-v3", "door-open-v3"),
            ),
        )
        for config in configs:
            with self.subTest(profile=config.profile):
                benchmark = MetaWorldBenchmark(config)
                episode = benchmark.episodes("train", seed=7, count=1)[0]
                environment = benchmark.make_environment(episode)
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    assert isinstance(observation, dict)
                    self.assertEqual(set(observation), {"state", "task"})
                    task = observation["task"]
                    self.assertIsInstance(task, TensorValue)
                    assert isinstance(task, TensorValue)
                    self.assertEqual(task.dtype, "bool")
                    self.assertEqual(sum(task.data), 1)
                    self.assertEqual(task.shape, (len(config.task_names),))
                finally:
                    environment.close()

    def test_plans_are_reproducible_and_balanced(self) -> None:
        benchmark = MetaWorldBenchmark(MetaWorldConfig(profile="mt10"))
        first = tuple(benchmark.episodes("train", seed=7, count=20))
        repeated = tuple(benchmark.episodes("train", seed=7, count=20))
        self.assertEqual(first, repeated)
        indexes = [
            episode.scenario["task_index"]
            for episode in first
            if type(episode.scenario) is dict
        ]
        self.assertEqual(len(set(indexes)), 10)
        self.assertTrue(all(indexes.count(index) == 2 for index in set(indexes)))

    def test_invalid_configuration_and_action_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetaWorldConfig(profile="unknown")
        with self.assertRaises(ValueError):
            MetaWorldConfig(profile="custom")
        with self.assertRaises(ValueError):
            MetaWorldConfig(
                profile="custom",
                custom_tasks=("reach-v3", "reach-v3"),
            )
        with self.assertRaises(TypeError):
            MetaWorldConfig(custom_tasks=["reach-v3"])  # type: ignore[arg-type]

        environment = MetaWorldBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([0, 0, 0, 0])
        finally:
            environment.close()

    def test_collection_requires_host_task_scenario(self) -> None:
        with self.assertRaises(ValueError):
            MetaWorldBenchmark(
                MetaWorldConfig(profile="mt10")
            ).make_environment(EpisodeSpec(environment_seed=1))

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            MetaWorldBenchmark(),
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
