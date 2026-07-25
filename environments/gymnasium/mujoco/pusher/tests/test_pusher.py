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

from pusher import PusherBenchmark, PusherConfig, baseline_program

_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "upper_arm_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
)
_OBSERVATION_FIELDS = {
    *(f"{name}_angle" for name in _JOINT_NAMES),
    *(f"{name}_angular_velocity" for name in _JOINT_NAMES),
    "fingertip_x",
    "fingertip_y",
    "fingertip_z",
    "object_x",
    "object_y",
    "object_z",
    "goal_x",
    "goal_y",
    "goal_z",
}


class PusherBenchmarkTests(unittest.TestCase):
    def test_config_controls_environment_identity_and_failure_scale(
        self,
    ) -> None:
        default = PusherBenchmark()
        configured = PusherBenchmark(
            PusherConfig(
                frame_skip=4,
                reward_near_weight=0.25,
                reward_dist_weight=2.0,
                reward_control_weight=2.0,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/Pusher-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 100)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 5,
                "reward_near_weight": 0.5,
                "reward_dist_weight": 1.0,
                "reward_control_weight": 0.1,
            },
        )
        self.assertEqual(
            configured.spec.environment_parameters,
            {
                "frame_skip": 4,
                "reward_near_weight": 0.25,
                "reward_dist_weight": 2.0,
                "reward_control_weight": 2.0,
            },
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            configured.spec.environment_digest,
        )
        self.assertEqual(
            configured.spec.metadata["failure_return"],
            -6000.0,
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            PusherConfig(frame_skip=5.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PusherConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            PusherConfig(reward_near_weight=1)
        for invalid in (-1.0, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    PusherConfig(reward_control_weight=invalid)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = PusherBenchmark()

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
        benchmark = PusherBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0] * 7,
                        [0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            self.assertTrue(
                all(type(value) is float for value in observation.values())
            )
            step = environment.step([0.0] * 7)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(
                set(step.metrics),
                {
                    "reward_distance",
                    "reward_control",
                    "reward_near",
                },
            )
            terms: list[float] = []
            for value in step.metrics.values():
                self.assertIs(type(value), float)
                assert type(value) is float
                terms.append(value)
            self.assertAlmostEqual(step.reward, sum(terms))
        finally:
            environment.close()
            environment.close()

    def test_parameterized_environment_conforms(self) -> None:
        benchmark = PusherBenchmark(
            PusherConfig(
                frame_skip=4,
                reward_near_weight=0.25,
                reward_dist_weight=0.5,
                reward_control_weight=0.05,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25] * 7,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_environment_requires_seven_exact_bounded_floats(
        self,
    ) -> None:
        benchmark = PusherBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0,) * 7,
            [0.0] * 6,
            [0] * 7,
            [2.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            PusherBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"frame_skip": 4},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = PusherBenchmark()
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
        self.assertEqual(
            feedback.content["mean_final_object_goal_distance"],
            None,
        )

    def test_zero_torque_baseline_publishes_reward_breakdown(self) -> None:
        benchmark = PusherBenchmark()
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
            "gymnasium/Pusher-v5/mean-return-v1",
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
        transitions = tuple(
            document
            for document in documents
            if document["type"] == "transition"
        )
        self.assertEqual(len(transitions), 100)
        self.assertEqual(
            set(transitions[0]["observation"]),
            _OBSERVATION_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0] * 7)
        self.assertEqual(
            set(transitions[0]["reward_terms"]),
            {
                "reward_distance",
                "reward_control",
                "reward_near",
            },
        )

    def test_shoulder_lift_signal_improves_on_zero_torque(self) -> None:
        benchmark = PusherBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[float] = []
        shoulder_lift: list[float] = []

        for episode in episodes:
            zero_torque.append(
                _rollout(benchmark, episode, lift=False)
            )
            shoulder_lift.append(
                _rollout(benchmark, episode, lift=True)
            )

        self.assertGreater(
            statistics.fmean(shoulder_lift),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    observation: dict[str, PolicyValue] = {}
    for name in _JOINT_NAMES:
        observation[f"{name}_angle"] = 0.0
        observation[f"{name}_angular_velocity"] = 0.0
    observation.update(
        {
            "fingertip_x": 0.8,
            "fingertip_y": -0.6,
            "fingertip_z": 0.0,
            "object_x": 0.2,
            "object_y": -0.3,
            "object_z": -0.275,
            "goal_x": 0.45,
            "goal_y": -0.05,
            "goal_z": -0.323,
        }
    )
    return observation


def _rollout(
    benchmark: PusherBenchmark,
    episode: EpisodeSpec,
    *,
    lift: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        environment.reset()
        action: PolicyValue = (
            [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
            if lift
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        for _ in range(100):
            result = environment.step(action)
            total += result.reward
            if result.done:
                break
    finally:
        environment.close()
    return total


if __name__ == "__main__":
    unittest.main()
