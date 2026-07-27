from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import EpisodeRecord, EpisodeSpec
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_playground import (
    PlaygroundBenchmark,
    baseline_program,
)


class PlaygroundTests(unittest.TestCase):
    def test_spec_defines_upstream_identity(self) -> None:
        default = PlaygroundBenchmark()
        self.assertEqual(
            default.spec.id,
            "minigrid/Playground-v0/room-coverage-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(
            default.spec.metadata["environment"],
            "MiniGrid-Playground-v0",
        )

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = PlaygroundBenchmark()
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
            PlaygroundBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        )
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_baseline_solves_seeded_episodes(self) -> None:
        benchmark = PlaygroundBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=6,
                seed=5,
                episode_timeout_seconds=10,
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
        "mission": "",
    }


if __name__ == "__main__":
    unittest.main()
