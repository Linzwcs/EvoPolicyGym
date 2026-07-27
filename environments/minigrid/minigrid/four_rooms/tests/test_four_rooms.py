from __future__ import annotations

import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import EpisodeRecord, EpisodeSpec
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_four_rooms import (
    FourRoomsBenchmark,
    baseline_program,
)


class FourRoomsTests(unittest.TestCase):
    def test_spec_defines_upstream_identity(self) -> None:
        default = FourRoomsBenchmark()
        self.assertEqual(
            default.spec.id,
            "minigrid/FourRooms-v0/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 256)
        self.assertEqual(
            default.spec.metadata["environment"],
            "MiniGrid-FourRooms-v0",
        )

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = FourRoomsBenchmark()
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
            FourRoomsBenchmark().feedback((failed,)).artifacts[0].read_bytes()
        )
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)

    def test_baseline_solves_seeded_episodes(self) -> None:
        benchmark = FourRoomsBenchmark()
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
        "mission": "reach the goal",
    }


if __name__ == "__main__":
    unittest.main()
