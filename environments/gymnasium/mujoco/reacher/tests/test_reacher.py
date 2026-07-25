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

from reacher import ReacherBenchmark, ReacherConfig, baseline_program

_OBSERVATION_FIELDS = {
    "joint0_cos",
    "joint1_cos",
    "joint0_sin",
    "joint1_sin",
    "target_x",
    "target_y",
    "joint0_angular_velocity",
    "joint1_angular_velocity",
    "fingertip_target_x",
    "fingertip_target_y",
}


class ReacherBenchmarkTests(unittest.TestCase):
    def test_config_controls_environment_identity_and_failure_scale(
        self,
    ) -> None:
        default = ReacherBenchmark()
        configured = ReacherBenchmark(
            ReacherConfig(
                frame_skip=3,
                reward_dist_weight=2.0,
                reward_control_weight=0.5,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/Reacher-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 50)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 2,
                "reward_dist_weight": 1.0,
                "reward_control_weight": 1.0,
            },
        )
        self.assertEqual(
            configured.spec.environment_parameters,
            {
                "frame_skip": 3,
                "reward_dist_weight": 2.0,
                "reward_control_weight": 0.5,
            },
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            configured.spec.environment_digest,
        )
        self.assertEqual(
            configured.spec.metadata["failure_return"],
            -2000.0,
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            ReacherConfig(frame_skip=2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ReacherConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            ReacherConfig(reward_dist_weight=1)
        for invalid in (-1.0, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    ReacherConfig(reward_control_weight=invalid)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = ReacherBenchmark()

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
        benchmark = ReacherBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0, 0.0],
                        [0.1, -0.1],
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
            step = environment.step([0.0, 0.0])
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(
                set(step.metrics),
                {"reward_distance", "reward_control"},
            )
            reward_distance = step.metrics["reward_distance"]
            reward_control = step.metrics["reward_control"]
            self.assertIs(type(reward_distance), float)
            self.assertIs(type(reward_control), float)
            assert type(reward_distance) is float
            assert type(reward_control) is float
            self.assertAlmostEqual(
                step.reward,
                reward_distance + reward_control,
            )
        finally:
            environment.close()
            environment.close()

    def test_parameterized_environment_conforms(self) -> None:
        benchmark = ReacherBenchmark(
            ReacherConfig(
                frame_skip=3,
                reward_dist_weight=0.5,
                reward_control_weight=0.25,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25, -0.25],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_environment_requires_two_exact_bounded_floats(self) -> None:
        benchmark = ReacherBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0),
            [0.0],
            [0, 0],
            [1.1, 0.0],
            [math.nan, 0.0],
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
            ReacherBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"frame_skip": 3},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = ReacherBenchmark()
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
        self.assertEqual(feedback.content["mean_final_distance"], None)

    def test_zero_torque_baseline_publishes_reward_breakdown(self) -> None:
        benchmark = ReacherBenchmark()
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
            "gymnasium/Reacher-v5/mean-return-v1",
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
        self.assertEqual(len(transitions), 50)
        self.assertEqual(
            set(transitions[0]["observation"]),
            _OBSERVATION_FIELDS,
        )
        self.assertEqual(transitions[0]["action"], [0.0, 0.0])
        self.assertEqual(
            set(transitions[0]["reward_terms"]),
            {"reward_distance", "reward_control"},
        )

    def test_inverse_kinematics_controller_improves_on_zero_torque(
        self,
    ) -> None:
        benchmark = ReacherBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[float] = []
        controlled: list[float] = []

        for episode in episodes:
            zero_torque.append(
                _rollout(benchmark, episode, controlled=False)
            )
            controlled.append(
                _rollout(benchmark, episode, controlled=True)
            )

        self.assertGreater(
            statistics.fmean(controlled),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    return {
        "joint0_cos": 1.0,
        "joint1_cos": 1.0,
        "joint0_sin": 0.0,
        "joint1_sin": 0.0,
        "target_x": 0.1,
        "target_y": 0.0,
        "joint0_angular_velocity": 0.0,
        "joint1_angular_velocity": 0.0,
        "fingertip_target_x": 0.1,
        "fingertip_target_y": 0.0,
    }


def _rollout(
    benchmark: ReacherBenchmark,
    episode: EpisodeSpec,
    *,
    controlled: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(50):
            assert isinstance(observation, dict)
            action: PolicyValue
            if controlled:
                action = _inverse_kinematics_action(observation)
            else:
                action = [0.0, 0.0]
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    return total


def _inverse_kinematics_action(
    observation: dict[str, PolicyValue],
) -> list[PolicyValue]:
    values: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        assert type(value) is float
        values[key] = value

    current0 = math.atan2(values["joint0_sin"], values["joint0_cos"])
    current1 = math.atan2(values["joint1_sin"], values["joint1_cos"])
    target_x = values["target_x"]
    target_y = values["target_y"]
    link = 0.1
    cos_desired1 = (
        target_x * target_x
        + target_y * target_y
        - 2.0 * link * link
    ) / (2.0 * link * link)
    cos_desired1 = max(-1.0, min(1.0, cos_desired1))
    desired1 = math.acos(cos_desired1)
    desired0 = math.atan2(target_y, target_x) - math.atan2(
        link * math.sin(desired1),
        link + link * math.cos(desired1),
    )
    error0 = _wrapped_angle(desired0 - current0)
    error1 = _wrapped_angle(desired1 - current1)
    action0 = (
        0.12 * error0
        - 0.015 * values["joint0_angular_velocity"]
    )
    action1 = (
        0.12 * error1
        - 0.015 * values["joint1_angular_velocity"]
    )
    return [
        max(-1.0, min(1.0, action0)),
        max(-1.0, min(1.0, action1)),
    ]


def _wrapped_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    unittest.main()
