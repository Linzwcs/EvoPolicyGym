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

from lunar_lander import (
    LunarLanderBenchmark,
    LunarLanderConfig,
    baseline_program,
)


class LunarLanderBenchmarkTests(unittest.TestCase):
    def test_config_controls_action_space_and_environment_identity(
        self,
    ) -> None:
        discrete = LunarLanderBenchmark()
        continuous = LunarLanderBenchmark(
            LunarLanderConfig(
                continuous=True,
                gravity=-9.0,
                enable_wind=True,
                wind_power=12.0,
                turbulence_power=1.0,
            )
        )

        self.assertEqual(
            discrete.spec.id,
            "gymnasium/LunarLander-v3/mean-return-v1",
        )
        self.assertEqual(discrete.spec.max_episode_steps, 1000)
        self.assertEqual(discrete.spec.primary_metric, "mean_return")
        self.assertNotEqual(
            discrete.spec.environment_digest,
            continuous.spec.environment_digest,
        )
        self.assertEqual(
            continuous.spec.environment_parameters,
            {
                "continuous": True,
                "gravity": -9.0,
                "enable_wind": True,
                "wind_power": 12.0,
                "turbulence_power": 1.0,
            },
        )
        self.assertIsInstance(discrete.spec.action_space, dict)
        self.assertIsInstance(continuous.spec.action_space, dict)
        assert isinstance(discrete.spec.action_space, dict)
        assert isinstance(continuous.spec.action_space, dict)
        self.assertEqual(discrete.spec.action_space["type"], "discrete")
        self.assertEqual(continuous.spec.action_space["shape"], [2])

    def test_config_rejects_invalid_types_and_gravity(self) -> None:
        with self.assertRaises(TypeError):
            LunarLanderConfig(continuous=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            LunarLanderConfig(enable_wind=0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            LunarLanderConfig(gravity=-10)
        for invalid in (-12.0, 0.0, math.nan, math.inf):
            with self.subTest(gravity=invalid):
                with self.assertRaises(ValueError):
                    LunarLanderConfig(gravity=invalid)
        with self.assertRaises(ValueError):
            LunarLanderConfig(wind_power=math.inf)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = LunarLanderBenchmark()

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

    def test_discrete_environment_is_deterministic_and_strict(self) -> None:
        benchmark = LunarLanderBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 2, 1, 3),
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
                {
                    "x_position",
                    "y_position",
                    "x_velocity",
                    "y_velocity",
                    "angle",
                    "angular_velocity",
                    "left_leg_contact",
                    "right_leg_contact",
                },
            )
            self.assertIsInstance(observation["left_leg_contact"], bool)
            self.assertIsInstance(observation["right_leg_contact"], bool)
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            4,
            True,
            0.0,
            [0.0, 0.0],
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

    def test_continuous_environment_requires_two_exact_floats(self) -> None:
        benchmark = LunarLanderBenchmark(
            LunarLanderConfig(continuous=True)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(
                        [0.0, 0.0],
                        [1.0, -1.0],
                        [-0.25, 0.75],
                    ),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        invalid_actions: tuple[PolicyValue, ...] = (
            (0.0, 0.0),
            [0.0],
            [0, 0],
            [1.1, 0.0],
            [math.nan, 0.0],
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
            LunarLanderBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"gravity": -5.0},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = LunarLanderBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "x_position": 0.0,
                "y_position": 1.0,
                "x_velocity": 0.0,
                "y_velocity": 0.0,
                "angle": 0.0,
                "angular_velocity": 0.0,
                "left_leg_contact": False,
                "right_leg_contact": False,
            },
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

    def test_no_thrust_baseline_publishes_complete_trace(self) -> None:
        benchmark = LunarLanderBenchmark()
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
            "gymnasium/LunarLander-v3/mean-return-v1",
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
            {
                "x_position",
                "y_position",
                "x_velocity",
                "y_velocity",
                "angle",
                "angular_velocity",
                "left_leg_contact",
                "right_leg_contact",
            },
        )
        self.assertEqual(transitions[0]["action"], 0)

    def test_reference_heuristic_improves_on_no_thrust(self) -> None:
        benchmark = LunarLanderBenchmark()
        episodes = benchmark.episodes(
            "validation",
            seed=17,
            count=10,
        )
        no_thrust: list[float] = []
        heuristic: list[float] = []

        for episode in episodes:
            no_thrust.append(_rollout(benchmark, episode, heuristic=False))
            heuristic.append(_rollout(benchmark, episode, heuristic=True))

        self.assertGreater(
            statistics.fmean(heuristic),
            statistics.fmean(no_thrust),
        )


def _rollout(
    benchmark: LunarLanderBenchmark,
    episode: EpisodeSpec,
    *,
    heuristic: bool,
) -> float:
    environment = benchmark.make_environment(episode)
    total = 0.0
    try:
        observation = environment.reset()
        for _ in range(1000):
            assert isinstance(observation, dict)
            action = _heuristic_action(observation) if heuristic else 0
            result = environment.step(action)
            total += result.reward
            observation = result.observation
            if result.done:
                break
    finally:
        environment.close()
    return total


def _heuristic_action(observation: dict[str, PolicyValue]) -> int:
    values: dict[str, float] = {}
    for key in (
        "x_position",
        "y_position",
        "x_velocity",
        "y_velocity",
        "angle",
        "angular_velocity",
    ):
        value = observation[key]
        assert type(value) is float
        values[key] = value

    angle_target = max(
        -0.4,
        min(
            0.4,
            values["x_position"] * 0.5 + values["x_velocity"],
        ),
    )
    hover_target = 0.55 * abs(values["x_position"])
    angle_todo = (
        (angle_target - values["angle"]) * 0.5
        - values["angular_velocity"]
    )
    hover_todo = (
        (hover_target - values["y_position"]) * 0.5
        - values["y_velocity"] * 0.5
    )
    left_contact = observation["left_leg_contact"]
    right_contact = observation["right_leg_contact"]
    assert type(left_contact) is bool
    assert type(right_contact) is bool
    if left_contact or right_contact:
        angle_todo = 0.0
        hover_todo = -values["y_velocity"] * 0.5

    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


if __name__ == "__main__":
    unittest.main()
