from __future__ import annotations

import json
import math
import statistics
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from inverted_pendulum import (
    InvertedPendulumBenchmark,
    InvertedPendulumConfig,
    baseline_program,
)

_OBSERVATION_FIELDS = {
    "cart_position",
    "pole_angle",
    "cart_velocity",
    "pole_angular_velocity",
}


class InvertedPendulumBenchmarkTests(unittest.TestCase):
    def test_config_controls_environment_identity(self) -> None:
        default = InvertedPendulumBenchmark()
        configured = InvertedPendulumBenchmark(
            InvertedPendulumConfig(
                frame_skip=3,
                reset_noise_scale=0.05,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/InvertedPendulum-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertEqual(
            default.spec.environment_parameters,
            {"frame_skip": 2, "reset_noise_scale": 0.01},
        )
        self.assertEqual(
            configured.spec.environment_parameters,
            {"frame_skip": 3, "reset_noise_scale": 0.05},
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            configured.spec.environment_digest,
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            InvertedPendulumConfig(
                frame_skip=2.0  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            InvertedPendulumConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            InvertedPendulumConfig(reset_noise_scale=0)
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(reset_noise_scale=invalid):
                with self.assertRaises(ValueError):
                    InvertedPendulumConfig(reset_noise_scale=invalid)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = InvertedPendulumBenchmark()

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

    def test_environment_is_deterministic_semantic_and_conformant(
        self,
    ) -> None:
        benchmark = InvertedPendulumBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=([0.0], [0.5], [-0.5]),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), _OBSERVATION_FIELDS)
            self.assertTrue(
                all(type(value) is float for value in observation.values())
            )
            step = environment.step([0.0])
            self.assertEqual(step.metrics, {"reward_survive": 1.0})
            self.assertEqual(step.reward, 1.0)
        finally:
            environment.close()
            environment.close()

    def test_parameterized_environment_conforms(self) -> None:
        benchmark = InvertedPendulumBenchmark(
            InvertedPendulumConfig(
                frame_skip=3,
                reset_noise_scale=0.05,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_environment_requires_one_exact_bounded_float(self) -> None:
        benchmark = InvertedPendulumBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            0.0,
            (0.0,),
            [],
            [0],
            [3.1],
            [math.nan],
            True,
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

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            InvertedPendulumBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"frame_skip": 3},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = InvertedPendulumBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_sample_observation(),
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
        self.assertEqual(feedback.content["full_horizon_balances"], 0)

    def test_zero_force_baseline_publishes_survival_trace(self) -> None:
        benchmark = InvertedPendulumBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/InvertedPendulum-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertGreater(result.feedback.score, 0.0)
        self.assertLess(result.feedback.score, 950.0)
        documents = tuple(
            json.loads(line)
            for line in result.feedback.artifacts[0]
            .read_bytes()
            .splitlines()
        )
        transitions = tuple(
            document
            for document in documents
            if document["type"] == "transition"
        )
        self.assertTrue(transitions)
        self.assertEqual(
            set(transitions[0]["observation"]),
            _OBSERVATION_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0])
        self.assertEqual(
            transitions[0]["reward_terms"],
            {"reward_survive": 1.0},
        )

    def test_linear_feedback_balances_full_horizon(self) -> None:
        benchmark = InvertedPendulumBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_force: list[float] = []
        controlled: list[float] = []

        for episode in episodes:
            zero_force.append(
                _rollout(benchmark, episode, controlled=False)
            )
            controlled.append(
                _rollout(benchmark, episode, controlled=True)
            )

        self.assertEqual(controlled, [1000.0] * len(controlled))
        self.assertGreater(
            statistics.fmean(controlled),
            statistics.fmean(zero_force),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {
        "cart_position": 0.0,
        "pole_angle": 0.0,
        "cart_velocity": 0.0,
        "pole_angular_velocity": 0.0,
    }


def _rollout(
    benchmark: InvertedPendulumBenchmark,
    episode: EpisodeSpec,
    *,
    controlled: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(1000):
            assert isinstance(observation, dict)
            action: PolicyValue = (
                _feedback_action(observation)
                if controlled
                else [0.0]
            )
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    return total


def _feedback_action(
    observation: dict[str, PolicyValue],
) -> list[PolicyValue]:
    values: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        assert type(value) is float
        values[key] = value
    force = (
        0.5 * values["cart_position"]
        + 20.0 * values["pole_angle"]
        + 0.1 * values["cart_velocity"]
        + values["pole_angular_velocity"]
    )
    return [max(-3.0, min(3.0, force))]


if __name__ == "__main__":
    unittest.main()
