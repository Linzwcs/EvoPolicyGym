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

from ant import AntBenchmark, AntConfig, baseline_program

_BODY_FIELDS = {
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    "front_left_hip_angle",
    "front_left_ankle_angle",
    "front_right_hip_angle",
    "front_right_ankle_angle",
    "back_left_hip_angle",
    "back_left_ankle_angle",
    "back_right_hip_angle",
    "back_right_ankle_angle",
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    "front_left_hip_angular_velocity",
    "front_left_ankle_angular_velocity",
    "front_right_hip_angular_velocity",
    "front_right_ankle_angular_velocity",
    "back_left_hip_angular_velocity",
    "back_left_ankle_angular_velocity",
    "back_right_hip_angular_velocity",
    "back_right_ankle_angular_velocity",
}
_CONTACT_BODIES = {
    "torso",
    "front_left_leg",
    "front_left_aux",
    "front_left_ankle",
    "front_right_leg",
    "front_right_aux",
    "front_right_ankle",
    "back_left_leg",
    "back_left_aux",
    "back_left_ankle",
    "back_right_leg",
    "back_right_aux",
    "back_right_ankle",
}
_CONTACT_COMPONENTS = {
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
}
_METRIC_FIELDS = {
    "x_position",
    "y_position",
    "distance_from_origin",
    "x_velocity",
    "y_velocity",
    "reward_forward",
    "reward_control",
    "reward_contact",
    "reward_survive",
}
_BIAS_ACTION = [-0.017, 0.022, 0.014, -0.013, 0.018, 0.022, -0.012, 0.002]


class AntBenchmarkTests(unittest.TestCase):
    def test_config_controls_observation_schema_and_identity(self) -> None:
        default = AntBenchmark()
        configured = AntBenchmark(
            AntConfig(
                frame_skip=6,
                forward_reward_weight=2.0,
                ctrl_cost_weight=0.25,
                contact_cost_weight=0.001,
                healthy_reward=1.5,
                main_body=2,
                terminate_when_unhealthy=False,
                healthy_z_range=(0.1, 1.2),
                contact_force_range=(-2.0, 2.0),
                reset_noise_scale=0.05,
                exclude_current_positions_from_observation=False,
                include_cfrc_ext_in_observation=False,
            )
        )

        self.assertEqual(
            default.spec.id,
            "gymnasium/Ant-v5/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 1000)
        self.assertEqual(default.spec.primary_metric, "mean_return")
        self.assertEqual(
            default.spec.environment_parameters,
            {
                "frame_skip": 5,
                "forward_reward_weight": 1.0,
                "ctrl_cost_weight": 0.5,
                "contact_cost_weight": 0.0005,
                "healthy_reward": 1.0,
                "main_body": 1,
                "terminate_when_unhealthy": True,
                "healthy_z_range": [0.2, 1.0],
                "contact_force_range": [-1.0, 1.0],
                "reset_noise_scale": 0.1,
                "exclude_current_positions_from_observation": True,
                "include_cfrc_ext_in_observation": True,
            },
        )
        self.assertNotEqual(
            default.spec.environment_digest,
            configured.spec.environment_digest,
        )
        self.assertIsInstance(default.spec.observation_space, dict)
        self.assertIsInstance(configured.spec.observation_space, dict)
        assert isinstance(default.spec.observation_space, dict)
        assert isinstance(configured.spec.observation_space, dict)
        default_fields = default.spec.observation_space["fields"]
        configured_fields = configured.spec.observation_space["fields"]
        self.assertIsInstance(default_fields, dict)
        self.assertIsInstance(configured_fields, dict)
        assert isinstance(default_fields, dict)
        assert isinstance(configured_fields, dict)
        self.assertEqual(
            set(default_fields),
            {*_BODY_FIELDS, "contact_forces"},
        )
        self.assertEqual(
            set(configured_fields),
            {"torso_x_position", "torso_y_position", *_BODY_FIELDS},
        )

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            AntConfig(frame_skip=5.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AntConfig(frame_skip=0)
        with self.assertRaises(TypeError):
            AntConfig(contact_cost_weight=1)
        with self.assertRaises(TypeError):
            AntConfig(main_body="torso")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AntConfig(main_body=0)
        with self.assertRaises(ValueError):
            AntConfig(main_body=14)
        with self.assertRaises(TypeError):
            AntConfig(healthy_z_range=[0.2, 1.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AntConfig(contact_force_range=(1.0, -1.0))
        with self.assertRaises(ValueError):
            AntConfig(healthy_z_range=(math.nan, 1.0))
        with self.assertRaises(TypeError):
            AntConfig(
                include_cfrc_ext_in_observation=1  # type: ignore[arg-type]
            )
        for invalid in (-0.1, math.nan, math.inf, 1_000_001.0):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    AntConfig(ctrl_cost_weight=invalid)
        with self.assertRaises(ValueError):
            AntConfig(reset_noise_scale=1.1)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = AntBenchmark()

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

    def test_default_environment_is_nested_and_conformant(self) -> None:
        benchmark = AntBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0] * 8,
                        [0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 1.0, -1.0],
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
            self.assertEqual(
                set(observation),
                {*_BODY_FIELDS, "contact_forces"},
            )
            contacts = observation["contact_forces"]
            self.assertIsInstance(contacts, dict)
            assert isinstance(contacts, dict)
            self.assertEqual(set(contacts), _CONTACT_BODIES)
            torso = contacts["torso"]
            self.assertIsInstance(torso, dict)
            assert isinstance(torso, dict)
            self.assertEqual(set(torso), _CONTACT_COMPONENTS)
            step = environment.step([0.0] * 8)
            self.assertIsInstance(step.metrics, dict)
            assert isinstance(step.metrics, dict)
            self.assertEqual(set(step.metrics), _METRIC_FIELDS)
            forward = step.metrics["reward_forward"]
            control = step.metrics["reward_control"]
            contact = step.metrics["reward_contact"]
            survive = step.metrics["reward_survive"]
            assert type(forward) is float
            assert type(control) is float
            assert type(contact) is float
            assert type(survive) is float
            self.assertAlmostEqual(
                step.reward,
                forward + control + contact + survive,
            )
        finally:
            environment.close()
            environment.close()

    def test_positions_without_contact_environment_conforms(self) -> None:
        benchmark = AntBenchmark(
            AntConfig(
                exclude_current_positions_from_observation=False,
                include_cfrc_ext_in_observation=False,
            )
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=456),
                    actions=([0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)
        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=456)
        )
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(
                set(observation),
                {"torso_x_position", "torso_y_position", *_BODY_FIELDS},
            )
        finally:
            environment.close()

    def test_environment_requires_eight_exact_bounded_floats(self) -> None:
        benchmark = AntBenchmark()
        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            [0.0] * 7,
            [0] * 8,
            [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            AntBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"include_cfrc_ext_in_observation": False},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = AntBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_sample_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -4000.0)
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
        self.assertEqual(feedback.content["mean_final_x_position"], None)

    def test_zero_torque_baseline_publishes_complete_nested_trace(
        self,
    ) -> None:
        benchmark = AntBenchmark()
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
            "gymnasium/Ant-v5/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, 6000.0)
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
        self.assertEqual(len(transitions), 1000)
        self.assertEqual(
            set(transitions[0]["observation"]),
            {*_BODY_FIELDS, "contact_forces"},
        )
        self.assertEqual(
            set(transitions[0]["observation"]["contact_forces"]),
            _CONTACT_BODIES,
        )
        self.assertEqual(transitions[0]["action"], [0.0] * 8)
        self.assertEqual(set(transitions[0]["metrics"]), _METRIC_FIELDS)

    def test_small_torque_bias_improves_on_zero_torque(self) -> None:
        benchmark = AntBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=8,
        )
        zero_torque: list[float] = []
        biased: list[float] = []

        for episode in episodes:
            zero_torque.append(_rollout(benchmark, episode, biased=False))
            biased.append(_rollout(benchmark, episode, biased=True))

        self.assertGreater(
            statistics.fmean(biased),
            statistics.fmean(zero_torque),
        )


def _sample_observation() -> dict[str, PolicyValue]:
    observation: dict[str, PolicyValue] = {
        field: 0.0 for field in _BODY_FIELDS
    }
    observation["contact_forces"] = {
        body: {component: 0.0 for component in _CONTACT_COMPONENTS}
        for body in _CONTACT_BODIES
    }
    return observation


def _rollout(
    benchmark: AntBenchmark,
    episode: EpisodeSpec,
    *,
    biased: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        environment.reset()
        for _ in range(1000):
            action: PolicyValue = (
                [value for value in _BIAS_ACTION]
                if biased
                else [0.0] * 8
            )
            result = environment.step(action)
            total += result.reward
            if result.done:
                break
    finally:
        environment.close()
    return total


if __name__ == "__main__":
    unittest.main()
