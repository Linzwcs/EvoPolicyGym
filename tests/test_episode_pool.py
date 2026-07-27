from __future__ import annotations

import unittest
from collections.abc import Sequence

from evopolicygym.authoring import (
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.errors import EvaluationError
from evopolicygym.run import RunConfig
from evopolicygym.run._episode_pool import build_training_episode_pool


class PoolBenchmark:
    def __init__(self, *, count_delta: int = 0) -> None:
        self.count_delta = count_delta
        self.calls: list[tuple[str, int, int]] = []

    @property
    def spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(
            id="example/pool-v1",
            description="Episode pool fixture.",
            observation_space=None,
            action_space=None,
            metadata={},
            max_episode_steps=1,
            primary_metric="reward",
            score_direction="maximize",
        )

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        self.calls.append((split, seed, count))
        returned = count + self.count_delta
        return tuple(
            EpisodeSpec(
                environment_seed=(seed + index) % 2**64,
                scenario={"pool_offset": index},
            )
            for index in range(max(returned, 0))
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        del episode
        raise AssertionError("pool construction must not make Environments")

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        del episodes
        raise AssertionError("pool construction must not produce Feedback")


class TrainingEpisodePoolTests(unittest.TestCase):
    def test_v1_derivation_is_stable_and_plans_the_pool_once(self) -> None:
        benchmark = PoolBenchmark()
        config = RunConfig(
            split="train",
            seed=42,
            episode_budget=3,
            episode_pool_size=6,
        )

        pool = build_training_episode_pool(benchmark, config)

        self.assertEqual(
            benchmark.calls,
            [("train", 18_379_659_693_720_673_948, 6)],
        )
        self.assertEqual(
            tuple(item.policy_seed for item in pool),
            (
                14_800_640_075_610_612_724,
                18_417_242_732_802_892_877,
                4_899_245_222_503_671_816,
                4_992_441_726_760_788_367,
                15_637_959_876_430_147_121,
                17_254_460_676_015_203_078,
            ),
        )
        self.assertEqual(
            tuple(item.episode.scenario for item in pool),
            tuple({"pool_offset": index} for index in range(6)),
        )

    def test_wrong_benchmark_count_is_a_trusted_evaluation_failure(self) -> None:
        with self.assertRaisesRegex(
            EvaluationError,
            "wrong training Episode count",
        ):
            build_training_episode_pool(
                PoolBenchmark(count_delta=-1),
                RunConfig(episode_budget=3),
            )


if __name__ == "__main__":
    unittest.main()
