from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from taxi import TaxiBenchmark, TaxiConfig, baseline_program

_PLANNER_POLICY = """\
from collections import deque


class PlannerPolicy:
    def __init__(self, layout, landmarks):
        self.layout = tuple(layout)
        self.landmarks = {
            name: tuple(position)
            for name, position in landmarks.items()
        }

    def act(self, observation):
        row = observation["taxi_row"]
        column = observation["taxi_column"]
        passenger = observation["passenger_location"]
        destination = observation["destination"]
        legal = observation["legal_actions"]

        if passenger != "in_taxi":
            target = self.landmarks[passenger]
            if (row, column) == target and 4 in legal:
                return 4
        else:
            target = self.landmarks[destination]
            if (row, column) == target and 5 in legal:
                return 5
        return self._first_step((row, column), target)

    def _first_step(self, start, target):
        queue = deque([(start, None)])
        visited = {start}
        while queue:
            position, first_action = queue.popleft()
            if position == target:
                if first_action is None:
                    raise RuntimeError("planner is already at its target")
                return first_action
            for action, neighbor in self._neighbors(position):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(
                        (
                            neighbor,
                            action if first_action is None else first_action,
                        )
                    )
        raise RuntimeError("target is unreachable")

    def _neighbors(self, position):
        row, column = position
        if row < 4:
            yield 0, (row + 1, column)
        if row > 0:
            yield 1, (row - 1, column)
        if column < 4 and self.layout[row + 1][2 * column + 2] == ":":
            yield 2, (row, column + 1)
        if column > 0 and self.layout[row + 1][2 * column] == ":":
            yield 3, (row, column - 1)


def make_policy(context):
    return PlannerPolicy(
        context.metadata["map"],
        context.metadata["landmarks"],
    )
"""


class TaxiBenchmarkTests(unittest.TestCase):
    def test_config_is_typed_and_changes_environment_identity(self) -> None:
        dry = TaxiBenchmark()
        rainy = TaxiBenchmark(
            TaxiConfig(
                is_rainy=True,
                fickle_passenger=True,
                rainy_probability=0.7,
                fickle_probability=0.4,
            )
        )

        self.assertEqual(
            dry.spec.id,
            "gymnasium/Taxi-v4/mean-return-v1",
        )
        self.assertEqual(dry.spec.max_episode_steps, 200)
        self.assertEqual(dry.spec.primary_metric, "mean_return")
        self.assertNotEqual(
            dry.spec.environment_digest,
            rainy.spec.environment_digest,
        )
        self.assertEqual(
            rainy.spec.environment_parameters,
            {
                "is_rainy": True,
                "fickle_passenger": True,
                "rainy_probability": 0.7,
                "fickle_probability": 0.4,
            },
        )

    def test_config_rejects_ambiguous_probabilities(self) -> None:
        with self.assertRaises(TypeError):
            TaxiConfig(is_rainy=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TaxiConfig(fickle_passenger=0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TaxiConfig(rainy_probability=1)
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(probability=invalid):
                with self.assertRaises(ValueError):
                    TaxiConfig(fickle_probability=invalid)

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = TaxiBenchmark()

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

    def test_environment_is_deterministic_and_rejects_invalid_actions(
        self,
    ) -> None:
        benchmark = TaxiBenchmark()
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 3, 4, 5),
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
                    "state",
                    "taxi_row",
                    "taxi_column",
                    "passenger_location",
                    "destination",
                    "legal_actions",
                },
            )
            self.assertIsInstance(observation["legal_actions"], list)
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            6,
            True,
            1.0,
            [1],
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
            TaxiBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"is_rainy": True},
                )
            )

    def test_feedback_uses_failure_floor_and_keeps_identity_private(
        self,
    ) -> None:
        benchmark = TaxiBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation={
                "state": 29,
                "taxi_row": 0,
                "taxi_column": 1,
                "passenger_location": "yellow",
                "destination": "green",
                "legal_actions": [0, 3],
            },
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, -2000.0)
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
        self.assertEqual(feedback.content["failure_return"], -2000.0)

    def test_weak_baseline_publishes_complete_transition_trace(self) -> None:
        benchmark = TaxiBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=2,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "gymnasium/Taxi-v4/mean-return-v1",
        )
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertEqual(result.feedback.score, -200.0)
        self.assertEqual(
            tuple(episode.steps for episode in result.episodes),
            (200, 200),
        )
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
                "state",
                "taxi_row",
                "taxi_column",
                "passenger_location",
                "destination",
                "legal_actions",
            },
        )
        self.assertEqual(transitions[0]["action"], 0)
        self.assertEqual(transitions[0]["reward"], -1.0)

    def test_shortest_path_planner_improves_on_the_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "program"
            source.mkdir()
            (source / "policy.py").write_text(
                _PLANNER_POLICY,
                encoding="utf-8",
            )
            program = Program.from_directory(source)
            result = evaluate(
                program,
                TaxiBenchmark(),
                execution=ProcessExecution.unsafe(),
                config=EvaluationConfig(
                    split="validation",
                    episodes=20,
                    seed=11,
                    episode_timeout_seconds=10,
                ),
            )

        self.assertIsInstance(result.feedback.content, dict)
        assert isinstance(result.feedback.content, dict)
        self.assertEqual(
            result.feedback.content["successful_episodes"],
            20,
        )
        self.assertGreater(result.feedback.score, 0.0)


if __name__ == "__main__":
    unittest.main()
