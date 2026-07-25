from __future__ import annotations

import base64
import json
import math
import statistics
import unittest
import zlib

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

from car_racing import (
    CarRacingBenchmark,
    CarRacingConfig,
    baseline_program,
)


class CarRacingBenchmarkTests(unittest.TestCase):
    def test_config_controls_action_space_and_environment_identity(
        self,
    ) -> None:
        continuous = CarRacingBenchmark()
        discrete = CarRacingBenchmark(
            CarRacingConfig(
                continuous=False,
                lap_complete_percent=0.8,
                domain_randomize=True,
            )
        )

        self.assertEqual(
            continuous.spec.id,
            "gymnasium/CarRacing-v3/mean-return-v1",
        )
        self.assertEqual(continuous.spec.max_episode_steps, 1000)
        self.assertEqual(continuous.spec.primary_metric, "mean_return")
        self.assertEqual(
            continuous.spec.environment_parameters,
            {
                "continuous": True,
                "lap_complete_percent": 0.95,
                "domain_randomize": False,
            },
        )
        self.assertEqual(
            discrete.spec.environment_parameters,
            {
                "continuous": False,
                "lap_complete_percent": 0.8,
                "domain_randomize": True,
            },
        )
        self.assertNotEqual(
            continuous.spec.environment_digest,
            discrete.spec.environment_digest,
        )
        self.assertIsInstance(continuous.spec.action_space, dict)
        self.assertIsInstance(discrete.spec.action_space, dict)
        assert isinstance(continuous.spec.action_space, dict)
        assert isinstance(discrete.spec.action_space, dict)
        self.assertEqual(continuous.spec.action_space["shape"], [3])
        self.assertEqual(discrete.spec.action_space["type"], "discrete")

    def test_config_rejects_invalid_types_and_percentages(self) -> None:
        with self.assertRaises(TypeError):
            CarRacingConfig(continuous=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CarRacingConfig(domain_randomize=0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CarRacingConfig(lap_complete_percent=1)
        for invalid in (0.0, -0.1, 1.1, math.nan, math.inf):
            with self.subTest(lap_complete_percent=invalid):
                with self.assertRaises(ValueError):
                    CarRacingConfig(lap_complete_percent=invalid)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = CarRacingBenchmark()

        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        validation = tuple(
            benchmark.episodes("validation", seed=7, count=10)
        )

        self.assertEqual(train, repeated)
        self.assertEqual(len({item.environment_seed for item in train}), 10)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in validation
            )
        )
        self.assertTrue(all(item.scenario is None for item in train))

    def test_continuous_environment_is_deterministic_and_lossless(
        self,
    ) -> None:
        benchmark = CarRacingBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0, 0.0, 0.0],
                        [0.25, 0.5, 0.0],
                    ),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, TensorValue)
            assert isinstance(observation, TensorValue)
            self.assertEqual(observation.dtype, "uint8")
            self.assertEqual(observation.shape, (96, 96, 3))
            self.assertEqual(len(observation.data), 96 * 96 * 3)
        finally:
            environment.close()
            environment.close()

    def test_continuous_actions_require_exact_bounded_floats(self) -> None:
        benchmark = CarRacingBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0, 0.0),
            [0.0, 0.0],
            [0, 0, 0],
            [-1.1, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.0, 1.1],
            [math.nan, 0.0, 0.0],
        )
        for invalid in invalid_actions:
            environment = benchmark.make_environment(
                EpisodeSpec(environment_seed=123)
            )
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_discrete_environment_and_actions_conform(self) -> None:
        benchmark = CarRacingBenchmark(
            CarRacingConfig(continuous=False)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=(0, 1, 2, 3, 4),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            5,
            True,
            0.0,
            [0.0, 0.0, 0.0],
        )
        for invalid in invalid_actions:
            environment = benchmark.make_environment(
                EpisodeSpec(environment_seed=456)
            )
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_domain_randomized_environment_conforms(self) -> None:
        benchmark = CarRacingBenchmark(
            CarRacingConfig(domain_randomize=True)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=789),
                    actions=([0.0, 0.0, 0.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            CarRacingBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"continuous": False},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = CarRacingBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_black_frame(),
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -1000.0)
        self.assertEqual(len(feedback.artifacts), 1)
        self.assertEqual(feedback.artifacts[0].name, "trace.jsonl")
        self.assertNotIn(
            b"environment_seed",
            feedback.artifacts[0].read_bytes(),
        )
        self.assertNotIn(
            b"policy_seed",
            feedback.artifacts[0].read_bytes(),
        )
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        self.assertEqual(feedback.content["failure_return"], -1000.0)
        self.assertEqual(
            feedback.content["trace_frame_encoding"],
            "zlib+base64",
        )

    def test_baseline_publishes_losslessly_reconstructable_trace(
        self,
    ) -> None:
        benchmark = CarRacingBenchmark()
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

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/CarRacing-v3/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, 0.0)
        documents = tuple(
            json.loads(line)
            for line in result.feedback.artifacts[0]
            .read_bytes()
            .splitlines()
        )
        self.assertGreater(len(documents), 1)
        episode = documents[0]
        self.assertEqual(episode["type"], "episode")
        frame = episode["initial_observation"]
        self.assertEqual(frame["dtype"], "uint8")
        self.assertEqual(frame["shape"], [96, 96, 3])
        self.assertEqual(frame["encoding"], "zlib+base64")
        reconstructed = zlib.decompress(
            base64.b64decode(frame["data"], validate=True)
        )
        self.assertEqual(len(reconstructed), 96 * 96 * 3)
        transition = documents[1]
        self.assertEqual(transition["type"], "transition")
        self.assertEqual(transition["action"], [0.0, 0.5, 0.0])
        self.assertIn("next_observation", transition)

    def test_full_throttle_improves_on_half_throttle_baseline(self) -> None:
        benchmark = CarRacingBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=2,
        )
        half_throttle: list[float] = []
        full_throttle: list[float] = []

        for episode in episodes:
            half_throttle.append(
                _rollout(benchmark, episode, gas=0.5)
            )
            full_throttle.append(
                _rollout(benchmark, episode, gas=1.0)
            )

        self.assertGreater(
            statistics.fmean(full_throttle),
            statistics.fmean(half_throttle),
        )


def _black_frame() -> TensorValue:
    return TensorValue(
        dtype="uint8",
        shape=(96, 96, 3),
        data=bytes(96 * 96 * 3),
    )


def _rollout(
    benchmark: CarRacingBenchmark,
    episode: EpisodeSpec,
    *,
    gas: float,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        environment.reset()
        action: PolicyValue = [0.0, gas, 0.0]
        for _ in range(1000):
            result = environment.step(action)
            total += result.reward
            if result.done:
                break
    finally:
        environment.close()
    return total


if __name__ == "__main__":
    unittest.main()
