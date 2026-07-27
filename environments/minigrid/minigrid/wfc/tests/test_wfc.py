from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_wfc import (
    WFC_PROFILES,
    WFCBenchmark,
    WFCConfig,
    baseline_program,
)


class WFCTests(unittest.TestCase):
    def test_spec_defines_upstream_identity(self) -> None:
        default = WFCBenchmark()
        self.assertEqual(
            default.spec.id,
            "minigrid/WFC-v0/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 2500)
        self.assertEqual(
            default.spec.metadata["environment"],
            "MiniGrid-WFC-MazeSimple-v0",
        )
        slow = WFCBenchmark(
            WFCConfig(profile="DungeonSpirals", size=15)
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            slow.spec.environment_digest,
        )

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = WFCBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=1, scenario={"size": 11})
            )

    def test_feedback_privacy(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        trace = (
            WFCBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        )
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_all_requested_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(WFC_PROFILES), 22)
        for profile in WFC_PROFILES:
            with self.subTest(profile=profile):
                benchmark = WFCBenchmark(
                    WFCConfig(profile=profile, size=15)
                )
                environment = benchmark.make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    step = environment.step(0)
                    self.assertIsInstance(step.reward, float)
                finally:
                    environment.close()
                    environment.close()

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            WFCBenchmark(WFCConfig(size=15)),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    (0,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_baseline_solves_seeded_episodes(self) -> None:
        profiles = (
            "MazeSimple",
            "DungeonMazeScaled",
            "RoomsFabric",
            "ObstaclesBlackdots",
            "ObstaclesAngular",
            "ObstaclesHogs3",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = WFCBenchmark(
                    WFCConfig(profile=profile, size=15)
                )
                result = evaluate(
                    baseline_program(),
                    benchmark,
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="validation",
                        episodes=1,
                        seed=5,
                        episode_timeout_seconds=30,
                    ),
                )
                self.assertEqual(result.feedback.score, 1.0)


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "traverse the maze to get to the goal",
    }


if __name__ == "__main__":
    unittest.main()
