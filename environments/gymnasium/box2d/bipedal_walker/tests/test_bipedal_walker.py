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
from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

from bipedal_walker import (
    BipedalWalkerBenchmark,
    BipedalWalkerConfig,
    baseline_program,
)

_OBSERVATION_FIELDS = {
    "hull_angle",
    "hull_angular_velocity",
    "horizontal_velocity",
    "vertical_velocity",
    "left_hip_angle",
    "left_hip_angular_velocity",
    "left_knee_angle",
    "left_knee_angular_velocity",
    "left_foot_contact",
    "right_hip_angle",
    "right_hip_angular_velocity",
    "right_knee_angle",
    "right_knee_angular_velocity",
    "right_foot_contact",
    "lidar_ranges",
}


class BipedalWalkerBenchmarkTests(unittest.TestCase):
    def test_config_controls_environment_identity(self) -> None:
        normal = BipedalWalkerBenchmark()
        hardcore = BipedalWalkerBenchmark(
            BipedalWalkerConfig(hardcore=True)
        )

        self.assertEqual(
            normal.spec.id,
            "gymnasium/BipedalWalker-v3/mean-return-v1",
        )
        self.assertEqual(normal.spec.max_episode_steps, 1600)
        self.assertEqual(normal.spec.primary_metric, "mean_return")
        self.assertEqual(
            normal.spec.environment_parameters,
            {"hardcore": False},
        )
        self.assertEqual(
            hardcore.spec.environment_parameters,
            {"hardcore": True},
        )
        self.assertNotEqual(
            normal.spec.environment_digest,
            hardcore.spec.environment_digest,
        )

    def test_config_rejects_non_boolean_hardcore(self) -> None:
        with self.assertRaises(TypeError):
            BipedalWalkerConfig(hardcore=1)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = BipedalWalkerBenchmark()

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

    def test_environment_is_deterministic_and_semantic(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, -0.5, 0.25, -0.25],
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
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), _OBSERVATION_FIELDS)
            self.assertIsInstance(observation["left_foot_contact"], bool)
            self.assertIsInstance(observation["right_foot_contact"], bool)
            lidar = observation["lidar_ranges"]
            self.assertIsInstance(lidar, list)
            assert isinstance(lidar, list)
            self.assertEqual(len(lidar), 10)
            self.assertTrue(all(type(value) is float for value in lidar))
        finally:
            environment.close()
            environment.close()

    def test_environment_requires_four_exact_bounded_floats(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0, 0.0, 0.0),
            [0.0, 0.0, 0.0],
            [0, 0, 0, 0],
            [1.1, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0],
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

    def test_hardcore_environment_conforms(self) -> None:
        benchmark = BipedalWalkerBenchmark(
            BipedalWalkerConfig(hardcore=True)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.0, 0.0, 0.0, 0.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            BipedalWalkerBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"hardcore": True},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = BipedalWalkerBenchmark()
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
        self.assertEqual(feedback.content["failure_return"], -1000.0)

    def test_zero_torque_baseline_publishes_complete_trace(self) -> None:
        benchmark = BipedalWalkerBenchmark()
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
            "gymnasium/BipedalWalker-v3/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, 0.0)
        trace = result.feedback.artifacts[0]
        documents = tuple(
            json.loads(line)
            for line in trace.read_bytes().splitlines()
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
        self.assertEqual(
            transitions[0]["action"],
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_reference_heuristic_improves_on_zero_torque(self) -> None:
        benchmark = BipedalWalkerBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=4,
        )
        zero_torque: list[float] = []
        heuristic: list[float] = []

        for episode in episodes:
            zero_torque.append(
                _rollout(benchmark, episode, use_heuristic=False)
            )
            heuristic.append(
                _rollout(benchmark, episode, use_heuristic=True)
            )

        self.assertGreater(
            statistics.fmean(heuristic),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {
        "hull_angle": 0.0,
        "hull_angular_velocity": 0.0,
        "horizontal_velocity": 0.0,
        "vertical_velocity": 0.0,
        "left_hip_angle": 0.0,
        "left_hip_angular_velocity": 0.0,
        "left_knee_angle": 0.0,
        "left_knee_angular_velocity": 0.0,
        "left_foot_contact": True,
        "right_hip_angle": 0.0,
        "right_hip_angular_velocity": 0.0,
        "right_knee_angle": 0.0,
        "right_knee_angular_velocity": 0.0,
        "right_foot_contact": True,
        "lidar_ranges": [1.0] * 10,
    }


def _rollout(
    benchmark: BipedalWalkerBenchmark,
    episode: EpisodeSpec,
    *,
    use_heuristic: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    controller = BipedalWalkerHeuristics()
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(1600):
            assert isinstance(observation, dict)
            action: PolicyValue = (
                [
                    float(value)
                    for value in controller.step_heuristic(  # type: ignore[no-untyped-call]
                        _observation_vector(observation)
                    )
                ]
                if use_heuristic
                else [0.0, 0.0, 0.0, 0.0]
            )
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    return total


def _observation_vector(
    observation: dict[str, PolicyValue],
) -> list[float]:
    vector: list[float] = []
    for key in (
        "hull_angle",
        "hull_angular_velocity",
        "horizontal_velocity",
        "vertical_velocity",
        "left_hip_angle",
        "left_hip_angular_velocity",
        "left_knee_angle",
        "left_knee_angular_velocity",
        "left_foot_contact",
        "right_hip_angle",
        "right_hip_angular_velocity",
        "right_knee_angle",
        "right_knee_angular_velocity",
        "right_foot_contact",
    ):
        value = observation[key]
        if type(value) is bool:
            vector.append(float(value))
        else:
            assert type(value) is float
            vector.append(value)
    lidar = observation["lidar_ranges"]
    assert isinstance(lidar, list)
    for value in lidar:
        assert type(value) is float
        vector.append(value)
    return vector


if __name__ == "__main__":
    unittest.main()
