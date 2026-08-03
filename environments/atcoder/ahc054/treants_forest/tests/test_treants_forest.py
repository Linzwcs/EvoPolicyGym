from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue

from treants_forest import TreantsForestBenchmark, baseline_program
from treants_forest.environment import (
    MAX_EPISODE_STEPS,
    TreantsForestEnvironment,
)
from treants_forest.simulation import (
    MAX_SIZE,
    MIN_SIZE,
    ForestCase,
    ForestSimulation,
    InvalidPlacement,
    generate_case,
)


class TreantsForestBenchmarkTests(unittest.TestCase):
    def test_spec_describes_independent_bounded_integration(self) -> None:
        spec = TreantsForestBenchmark().spec

        self.assertEqual(
            spec.id,
            "atcoder/AHC054/TreantsForest/capped-mean-turns-v1",
        )
        self.assertEqual(spec.max_episode_steps, MAX_EPISODE_STEPS)
        self.assertEqual(spec.primary_metric, "capped_mean_turns")
        self.assertEqual(spec.score_direction, "maximize")
        self.assertEqual(
            spec.environment_parameters["generator"],
            "evopolicygym-independent-v1",
        )
        self.assertEqual(spec.environment_parameters["minimum_size"], MIN_SIZE)
        self.assertEqual(spec.environment_parameters["maximum_size"], MAX_SIZE)
        self.assertEqual(
            spec.environment_parameters["turn_cap"],
            MAX_EPISODE_STEPS,
        )
        reward_semantics = spec.environment_parameters["reward_semantics"]
        placement_atomicity = spec.environment_parameters["placement_atomicity"]
        path_diagnostics = spec.environment_parameters["path_diagnostics"]
        assert isinstance(reward_semantics, str)
        assert isinstance(placement_atomicity, str)
        assert isinstance(path_diagnostics, str)
        self.assertIn("exactly 1 point", reward_semantics)
        self.assertIn("accepted or rejected", placement_atomicity)
        self.assertIn("private adventurer target", path_diagnostics)
        self.assertEqual(spec.metadata["implementation"], "independent")
        self.assertEqual(
            spec.metadata["upstream_specification_revision"],
            "2025-09-21",
        )
        self.assertEqual(
            spec.metadata["upstream_tool_license"],
            "not declared; not redistributed",
        )
        self.assertFalse(spec.metadata["upstream_code_included"])
        self.assertFalse(spec.metadata["upstream_inputs_included"])
        self.assertFalse(spec.metadata["upstream_assets_included"])

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = TreantsForestBenchmark()

        train = tuple(benchmark.episodes("train", seed=7, count=12))
        repeated = tuple(benchmark.episodes("train", seed=7, count=12))
        validation = tuple(benchmark.episodes("validation", seed=7, count=12))
        test = tuple(benchmark.episodes("test", seed=7, count=12))

        self.assertEqual(train, repeated)
        self.assertEqual(len(set(train)), 12)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))
        self.assertTrue(all(item.scenario is None for item in train))

    def test_case_generator_is_version_stable_and_satisfies_contract(
        self,
    ) -> None:
        expected = (
            (26, (1, 20), 65),
            (34, (10, 7), 32),
            (27, (17, 20), 76),
        )
        generated = tuple(generate_case(seed) for seed in range(3))

        self.assertEqual(
            tuple((case.size, case.flower, len(case.initial_trees)) for case in generated),
            expected,
        )
        for case in generated:
            self.assertGreaterEqual(case.size, MIN_SIZE)
            self.assertLessEqual(case.size, MAX_SIZE)
            self.assertNotIn(case.entrance, case.initial_trees)
            self.assertNotIn(case.flower, case.initial_trees)
            self.assertGreaterEqual(
                abs(case.flower[0] - case.entrance[0]) + abs(case.flower[1] - case.entrance[1]),
                5,
            )
            self.assertEqual(
                len(case.target_order),
                case.size * case.size - 1,
            )

    def test_initial_observation_exposes_no_private_case_identity(self) -> None:
        environment = TreantsForestEnvironment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
        finally:
            environment.close()

        self.assertIsInstance(observation, dict)
        assert isinstance(observation, dict)
        self.assertEqual(
            set(observation),
            {
                "turn",
                "adventurer",
                "newly_revealed",
                "revealed_cells",
                "placed_treants",
                "initial",
            },
        )
        self.assertEqual(observation["turn"], 0)
        initial = observation["initial"]
        self.assertIsInstance(initial, dict)
        assert isinstance(initial, dict)
        self.assertEqual(
            set(initial),
            {"size", "entrance", "flower", "trees"},
        )
        encoded = json.dumps(observation, sort_keys=True)
        self.assertNotIn("seed", encoded)
        self.assertNotIn("target", encoded)
        self.assertNotIn("scenario", encoded)

    def test_no_placement_follows_official_turn_order(self) -> None:
        simulation = ForestSimulation(_open_case())

        for expected_turn in range(1, 6):
            simulation.step(())
            self.assertEqual(simulation.turn, expected_turn)
            self.assertEqual(simulation.position, (expected_turn, 10))
        self.assertTrue(simulation.done)

    def test_reveal_includes_first_tree_and_stops_behind_it(self) -> None:
        case = _case_with(
            flower=(5, 5),
            trees=frozenset({(0, 12)}),
            first_target=(2, 9),
        )
        simulation = ForestSimulation(case)

        simulation.step(())

        self.assertIn((0, 11), simulation.newly_revealed)
        self.assertIn((0, 12), simulation.newly_revealed)
        self.assertNotIn((0, 13), simulation.newly_revealed)

    def test_shortest_path_ties_use_up_down_left_right_order(self) -> None:
        simulation = ForestSimulation(
            _case_with(
                flower=(5, 5),
                trees=frozenset(),
                first_target=(2, 9),
            )
        )

        simulation.step(())

        self.assertEqual(simulation.position, (1, 10))

    def test_placement_is_atomic_and_preserves_required_paths(self) -> None:
        simulation = ForestSimulation(_open_case())

        with self.assertRaises(InvalidPlacement):
            simulation.step(((1, 10), (0, 9), (0, 11)))

        self.assertEqual(simulation.turn, 0)
        self.assertEqual(simulation.placed_count, 0)
        simulation.step(())
        self.assertEqual(simulation.position, (1, 10))

    def test_environment_rejects_complete_invalid_actions(self) -> None:
        invalid_actions: tuple[PolicyValue, ...] = (
            None,
            [],
            {
                "placements": (),
            },
            {"placements": [], "extra": True},
            {"placements": [[1]]},
            {"placements": [[1.0, 2]]},
            {"placements": [[True, 2]]},
            {"placements": [[1, 2], [1, 2]]},
            {"placements": [[-1, 2]]},
        )
        for action in invalid_actions:
            environment = TreantsForestEnvironment(EpisodeSpec(environment_seed=17))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction, msg=repr(action)):
                    environment.step(action)
            finally:
                environment.close()

    def test_environment_rejects_revealed_tree_flower_and_disconnection(
        self,
    ) -> None:
        case = _open_case()
        simulation = ForestSimulation(case)

        for placements in (
            (case.entrance,),
            (case.flower,),
            ((1, 10), (0, 9), (0, 11)),
        ):
            with self.assertRaises(InvalidPlacement):
                simulation.step(placements)
            self.assertEqual(simulation.turn, 0)

        tree_case = _case_with(
            flower=(5, 5),
            trees=frozenset({(0, 12)}),
            first_target=(2, 9),
        )
        with self.assertRaises(InvalidPlacement):
            ForestSimulation(tree_case).step(((0, 12),))

    def test_environment_lifecycle_and_deterministic_replay(self) -> None:
        benchmark = TreantsForestBenchmark()
        episode = EpisodeSpec(environment_seed=123)
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=episode,
                    actions=(
                        {"placements": []},
                        {"placements": []},
                    ),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        before_reset = benchmark.make_environment(episode)
        with self.assertRaises(RuntimeError):
            before_reset.step({"placements": []})
        before_reset.close()
        before_reset.close()

        repeated_reset = benchmark.make_environment(episode)
        repeated_reset.reset()
        with self.assertRaises(RuntimeError):
            repeated_reset.reset()
        repeated_reset.close()

    def test_real_placement_and_no_placement_turns_have_actionable_metrics(self) -> None:
        placed_environment = TreantsForestEnvironment(EpisodeSpec(environment_seed=123))
        try:
            placed_environment.reset()
            placed = placed_environment.step({"placements": [[36, 0]]})
        finally:
            placed_environment.close()

        placed_metrics = _step_metrics(placed)
        self.assertEqual(placed_metrics["placement_count_this_turn"], 1)
        self.assertEqual(placed_metrics["placed_treants"], 1)
        self.assertEqual(placed_metrics["submitted_placement_count"], 1)
        self.assertEqual(placed_metrics["no_placement_this_turn"], False)
        self.assertGreater(_number_metric(placed_metrics, "newly_revealed_cell_count"), 0)
        self.assertGreater(_number_metric(placed_metrics, "revealed_cell_fraction"), 0.0)
        self.assertGreater(_number_metric(placed_metrics, "legal_candidate_cell_count"), 0)
        self.assertGreaterEqual(_number_metric(placed_metrics, "flower_path_length"), 0)
        self.assertEqual(placed_metrics["score_so_far"], 1.0)

        empty_environment = TreantsForestEnvironment(EpisodeSpec(environment_seed=123))
        try:
            empty_environment.reset()
            empty = empty_environment.step({"placements": []})
        finally:
            empty_environment.close()
        empty_metrics = _step_metrics(empty)
        self.assertEqual(empty_metrics["no_placement_this_turn"], True)
        self.assertEqual(empty_metrics["no_placement_turn_count"], 1)
        self.assertEqual(empty_metrics["mean_submitted_placements_per_turn"], 0.0)

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        environment = TreantsForestEnvironment(EpisodeSpec(environment_seed=123))
        try:
            initial = environment.reset()
        finally:
            environment.close()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=123),
            policy_seed=456,
            initial_observation=initial,
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = TreantsForestBenchmark().feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        self.assertEqual(len(feedback.artifacts), 1)
        trace = feedback.artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b"target_order", trace)
        document = json.loads(trace)
        self.assertEqual(document["status"], "policy_failed")

    def test_baseline_program_completes_direct_evaluation(self) -> None:
        benchmark = TreantsForestBenchmark()
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

        self.assertEqual(result.benchmark_id, benchmark.spec.id)
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertGreater(result.feedback.score, 0.0)
        self.assertLessEqual(
            result.feedback.score,
            float(MAX_EPISODE_STEPS),
        )
        documents = [
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        ]
        transitions = [document for document in documents if document["type"] == "transition"]
        content = result.feedback.content
        assert isinstance(content, dict)
        self.assertTrue(transitions)
        self.assertEqual(
            transitions[0]["action"],
            {"placements": []},
        )
        self.assertIsNotNone(transitions[0]["observation"]["initial"])
        self.assertIsNone(transitions[0]["next_observation"]["initial"])
        self.assertIn("flower_path_length", transitions[0]["metrics"])
        self.assertIn("newly_revealed_cell_count", transitions[0]["metrics"])
        self.assertEqual(content["mean_no_placement_turn_fraction"], 1.0)
        self.assertEqual(content["placement_episode_rate"], 0.0)

    def test_feedback_bounds_long_transition_traces(self) -> None:
        environment = TreantsForestEnvironment(EpisodeSpec(environment_seed=123))
        try:
            initial = environment.reset()
            first = environment.step({"placements": []})
        finally:
            environment.close()
        self.assertIsInstance(first.metrics, dict)
        assert isinstance(first.metrics, dict)
        metrics: dict[str, PolicyValue] = dict(first.metrics)
        metrics.update(
            {
                "turns": MAX_EPISODE_STEPS,
                "remaining_turns": 0,
                "score_so_far": float(MAX_EPISODE_STEPS),
                "flower_reached": False,
                "turn_cap_reached": True,
                "terminal_reason": "turn_cap",
            }
        )
        ordinary = Transition(
            action={"placements": []},
            step=Step(
                observation=first.observation,
                reward=1.0,
                terminated=False,
                metrics=metrics,
            ),
        )
        terminal = Transition(
            action={"placements": []},
            step=Step(
                observation=first.observation,
                reward=1.0,
                terminated=False,
                truncated=True,
                metrics=metrics,
            ),
        )
        records = tuple(
            EpisodeRecord(
                episode=EpisodeSpec(environment_seed=index),
                policy_seed=index,
                initial_observation=initial,
                transitions=(
                    *((ordinary,) * (MAX_EPISODE_STEPS - 1)),
                    terminal,
                ),
            )
            for index in range(3)
        )

        feedback = TreantsForestBenchmark().feedback(records)

        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["traced_transitions"], 4_096)
        self.assertEqual(
            feedback.content["trace_transitions_omitted"],
            2_048,
        )


def _open_case() -> ForestCase:
    return _case_with(
        flower=(5, 10),
        trees=frozenset(),
        first_target=(5, 10),
    )


def _step_metrics(step: Step) -> dict[str, PolicyValue]:
    metrics = step.metrics
    assert isinstance(metrics, dict)
    return metrics


def _number_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _case_with(
    *,
    flower: tuple[int, int],
    trees: frozenset[tuple[int, int]],
    first_target: tuple[int, int],
) -> ForestCase:
    size = 20
    entrance = (0, size // 2)
    remaining = [
        (row, column)
        for row in range(size)
        for column in range(size)
        if (row, column) not in {entrance, first_target}
    ]
    return ForestCase(
        size=size,
        flower=flower,
        initial_trees=trees,
        target_order=(first_target, *remaining),
    )


if __name__ == "__main__":
    unittest.main()
