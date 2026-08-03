from __future__ import annotations

import json
import math
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

from molecules import MoleculesBenchmark, baseline_program
from molecules.environment import MoleculesEnvironment
from molecules.simulation import (
    GENERATOR_ID,
    POINTS,
    SPACE_SIZE,
    TARGET_COMPONENTS,
    TARGET_SIZE,
    TURNS,
    InvalidBond,
    MoleculesCase,
    MoleculesSimulation,
    final_score,
    generate_case,
)


class MoleculesBenchmarkTests(unittest.TestCase):
    def test_spec_describes_independent_full_task(self) -> None:
        spec = MoleculesBenchmark().spec

        self.assertEqual(
            spec.id,
            "atcoder/AHC057/Molecules/mean-log-cost-score-v1",
        )
        self.assertEqual(spec.max_episode_steps, TURNS)
        self.assertEqual(spec.primary_metric, "mean_log_cost_score")
        self.assertEqual(spec.score_direction, "maximize")
        self.assertEqual(spec.environment_parameters["generator"], GENERATOR_ID)
        self.assertEqual(spec.environment_parameters["points"], POINTS)
        self.assertEqual(spec.environment_parameters["turns"], TURNS)
        self.assertEqual(spec.environment_parameters["space_size"], SPACE_SIZE)
        self.assertEqual(
            spec.environment_parameters["target_components"],
            TARGET_COMPONENTS,
        )
        self.assertEqual(spec.environment_parameters["target_size"], TARGET_SIZE)
        reward_semantics = spec.environment_parameters["reward_semantics"]
        bond_atomicity = spec.environment_parameters["bond_atomicity"]
        score_formula = spec.environment_parameters["score_formula"]
        assert isinstance(reward_semantics, str)
        assert isinstance(bond_atomicity, str)
        assert isinstance(score_formula, str)
        self.assertIn("0 for turns 1-999", reward_semantics)
        self.assertIn("atomically", bond_atomicity)
        self.assertIn("log2", score_formula)
        self.assertEqual(spec.metadata["implementation"], "independent")
        self.assertEqual(
            spec.metadata["upstream_tool_license"],
            "not declared; not redistributed",
        )
        self.assertEqual(
            spec.metadata["upstream_tool_archive_revision"],
            "BJTm8xSg",
        )
        self.assertFalse(spec.metadata["upstream_code_included"])
        self.assertFalse(spec.metadata["upstream_inputs_included"])
        self.assertFalse(spec.metadata["upstream_assets_included"])

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = MoleculesBenchmark()

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
            ((4_502, 32_914), (83, -72), (14_806, 61_842), (28, -5)),
            ((63_666, 84_136), (-36, 58), (21_477, 85_935), (24, -50)),
            ((54_962, 4_915), (14, -51), (92_275, 93_328), (-29, -66)),
        )
        generated = tuple(generate_case(seed) for seed in range(3))

        self.assertEqual(
            tuple(
                (
                    case.positions[0],
                    case.velocities[0],
                    case.positions[-1],
                    case.velocities[-1],
                )
                for case in generated
            ),
            expected,
        )
        for case in generated:
            self.assertEqual(len(case.positions), POINTS)
            self.assertEqual(len(case.velocities), POINTS)
            self.assertTrue(
                all(0 <= x < SPACE_SIZE and 0 <= y < SPACE_SIZE for x, y in case.positions)
            )
            self.assertTrue(
                all(-100 <= vx <= 100 and -100 <= vy <= 100 for vx, vy in case.velocities)
            )

    def test_bond_uses_toroidal_cost_and_momentum(self) -> None:
        case = _simple_case()
        simulation = MoleculesSimulation(case)

        cost = simulation.step(((0, 1),))

        self.assertEqual(cost, 1)
        self.assertEqual(simulation.total_cost, 1)
        self.assertEqual(simulation.component_count, POINTS - 1)
        self.assertEqual(simulation.component_labels[0], 0)
        self.assertEqual(simulation.component_labels[1], 0)
        self.assertEqual(simulation.velocities[0], (0.0, 0.0))
        self.assertEqual(simulation.velocities[1], (0.0, 0.0))
        self.assertEqual(simulation.positions[0], (0.0, 0.0))
        self.assertEqual(simulation.positions[1], (99_999.0, 0.0))

    def test_complete_bond_set_is_atomic_and_rejects_cycles(self) -> None:
        simulation = MoleculesSimulation(_simple_case())

        with self.assertRaises(InvalidBond):
            simulation.step(((0, 1), (1, 0)))

        self.assertEqual(simulation.turn, 0)
        self.assertEqual(simulation.component_count, POINTS)
        self.assertEqual(simulation.total_cost, 0)
        simulation.step(((0, 1), (0, 2)))
        self.assertEqual(simulation.component_count, POINTS - 2)

    def test_component_cannot_exceed_target_size(self) -> None:
        simulation = MoleculesSimulation(_simple_case())
        simulation.step(tuple((0, point) for point in range(1, TARGET_SIZE)))

        with self.assertRaises(InvalidBond):
            simulation.step(((0, TARGET_SIZE),))

        self.assertEqual(simulation.turn, 1)
        self.assertEqual(simulation.component_sizes[-1], TARGET_SIZE)

    def test_final_turn_requires_exact_target_partition_atomically(
        self,
    ) -> None:
        simulation = MoleculesSimulation(_simple_case())
        for _ in range(TURNS - 1):
            simulation.step(())

        with self.assertRaises(InvalidBond):
            simulation.step(())

        self.assertEqual(simulation.turn, TURNS - 1)
        simulation.step(_baseline_bonds())
        self.assertTrue(simulation.done)
        self.assertEqual(
            simulation.component_sizes,
            (TARGET_SIZE,) * TARGET_COMPONENTS,
        )
        self.assertEqual(simulation.total_bonds, POINTS - TARGET_COMPONENTS)

    def test_final_score_matches_public_formula(self) -> None:
        expected = math.floor(
            1_000_000 * math.log2(SPACE_SIZE * (POINTS - TARGET_COMPONENTS)) + 0.5
        )
        self.assertEqual(final_score(0), expected)

    def test_initial_observation_hides_case_identity(self) -> None:
        environment = MoleculesEnvironment(EpisodeSpec(environment_seed=123))
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
                "turns_remaining",
                "positions",
                "velocities",
                "components",
                "component_count",
                "total_cost",
                "initial",
            },
        )
        self.assertEqual(observation["turn"], 0)
        self.assertEqual(observation["component_count"], POINTS)
        initial = observation["initial"]
        self.assertIsInstance(initial, dict)
        assert isinstance(initial, dict)
        self.assertEqual(
            set(initial),
            {
                "space_size",
                "target_components",
                "target_size",
                "turns",
            },
        )
        encoded = json.dumps(observation, sort_keys=True)
        self.assertNotIn("seed", encoded)
        self.assertNotIn("split", encoded)
        self.assertNotIn("scenario", encoded)

    def test_environment_rejects_malformed_actions(self) -> None:
        invalid_actions: tuple[PolicyValue, ...] = (
            None,
            [],
            {"bonds": None},
            {"bonds": [], "extra": True},
            {"bonds": [[0]]},
            {"bonds": [[True, 1]]},
            {"bonds": [[0.0, 1]]},
            {"bonds": [[-1, 1]]},
            {"bonds": [[0, POINTS]]},
            {"bonds": [[0, 0]]},
            {"bonds": [[0, 1], [1, 0]]},
        )
        for action in invalid_actions:
            environment = MoleculesEnvironment(EpisodeSpec(environment_seed=17))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction, msg=repr(action)):
                    environment.step(action)
            finally:
                environment.close()

    def test_environment_lifecycle_and_deterministic_replay(self) -> None:
        episode = EpisodeSpec(environment_seed=123)
        wait: PolicyValue = {"bonds": []}
        actions: tuple[PolicyValue, ...] = (
            {"bonds": [list(bond) for bond in _baseline_bonds()]},
            *((wait,) * (TURNS - 1)),
        )
        report = check_benchmark(
            MoleculesBenchmark(),
            fixtures=(BenchmarkFixture(episode=episode, actions=actions),),
        )
        self.assertTrue(report.passed, report.issues)

        before_reset = MoleculesEnvironment(episode)
        with self.assertRaises(RuntimeError):
            before_reset.step({"bonds": []})
        before_reset.close()
        before_reset.close()

        repeated_reset = MoleculesEnvironment(episode)
        repeated_reset.reset()
        with self.assertRaises(RuntimeError):
            repeated_reset.reset()
        repeated_reset.close()

    def test_real_incremental_and_complete_bonding_metrics_are_actionable(self) -> None:
        incremental = MoleculesEnvironment(EpisodeSpec(environment_seed=123))
        try:
            incremental.reset()
            first = incremental.step({"bonds": [[0, 1]]})
            waited = incremental.step({"bonds": []})
        finally:
            incremental.close()

        first_metrics = _step_metrics(first)
        waited_metrics = _step_metrics(waited)
        self.assertEqual(first.reward, 0.0)
        self.assertEqual(first_metrics["bond_count_this_turn"], 1)
        self.assertEqual(first_metrics["components"], POINTS - 1)
        self.assertEqual(
            first_metrics["required_bonds_remaining"],
            POINTS - 1 - TARGET_COMPONENTS,
        )
        self.assertEqual(first_metrics["total_bonds"], 1)
        self.assertEqual(first_metrics["bond_completion_fraction"], 1 / 290)
        self.assertEqual(first_metrics["singleton_component_count"], 298)
        self.assertEqual(first_metrics["largest_component_size"], 2)
        self.assertEqual(sum(_int_list_metric(first_metrics, "component_size_histogram")), 299)
        self.assertGreater(_number_metric(first_metrics, "action_cost_per_bond"), 0.0)
        self.assertGreater(
            _number_metric(first_metrics, "score_upper_bound_if_no_further_cost"), 0
        )
        self.assertEqual(waited_metrics["bond_action_this_turn"], False)
        self.assertEqual(waited_metrics["empty_bond_action_count"], 1)
        self.assertEqual(waited_metrics["empty_bond_action_fraction"], 0.5)

        complete = MoleculesEnvironment(EpisodeSpec(environment_seed=123))
        try:
            complete.reset()
            grouped = complete.step({"bonds": [list(bond) for bond in _baseline_bonds()]})
        finally:
            complete.close()
        grouped_metrics = _step_metrics(grouped)
        self.assertEqual(grouped_metrics["components"], TARGET_COMPONENTS)
        self.assertEqual(grouped_metrics["required_bonds_remaining"], 0)
        self.assertEqual(grouped_metrics["target_size_component_count"], 10)
        self.assertEqual(grouped_metrics["target_partition_ready"], True)
        self.assertEqual(grouped_metrics["target_partition_first_ready_turn"], 1)
        self.assertEqual(
            grouped_metrics["component_size_histogram"],
            [0] * 29 + [10],
        )

    def test_feedback_publishes_bounded_bond_trace(self) -> None:
        episode = EpisodeSpec(environment_seed=123)
        environment = MoleculesEnvironment(episode)
        transitions: list[Transition] = []
        try:
            initial = environment.reset()
            wait: PolicyValue = {"bonds": []}
            actions: tuple[PolicyValue, ...] = (
                {"bonds": [list(bond) for bond in _baseline_bonds()]},
                *((wait,) * (TURNS - 1)),
            )
            for action in actions:
                transitions.append(
                    Transition(
                        action=action,
                        step=environment.step(action),
                    )
                )
        finally:
            environment.close()
        completed = EpisodeRecord(
            episode=episode,
            policy_seed=456,
            initial_observation=initial,
            transitions=tuple(transitions),
        )
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=999),
            policy_seed=789,
            initial_observation=initial,
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = MoleculesBenchmark().feedback((completed, failed))

        self.assertGreater(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["completed"], 1)
        self.assertEqual(feedback.content["bond_events"], 1)
        trace = feedback.artifacts[0].read_bytes()
        self.assertLess(len(trace), 1_000_000)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b"scenario", trace)
        documents = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(documents[0]["status"], "completed")
        self.assertEqual(documents[1]["type"], "bond_event")
        self.assertEqual(documents[1]["bond_count"], POINTS - TARGET_COMPONENTS)
        self.assertEqual(documents[1]["required_bonds_remaining"], 0)
        self.assertEqual(documents[1]["target_partition_ready"], True)
        self.assertEqual(
            documents[1]["score_upper_bound_if_no_further_cost"],
            documents[0]["score"],
        )
        self.assertEqual(documents[-1]["status"], "policy_failed")
        self.assertEqual(feedback.content["mean_total_bonds"], 290.0)
        self.assertEqual(feedback.content["mean_bond_action_count"], 1.0)
        self.assertEqual(
            feedback.content["mean_empty_bond_action_fraction"],
            0.999,
        )
        self.assertEqual(
            feedback.content["mean_target_partition_first_ready_turn"],
            1.0,
        )

    def test_baseline_program_completes_direct_evaluation(self) -> None:
        benchmark = MoleculesBenchmark()
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
        self.assertEqual(result.episodes[0].status, "completed")
        self.assertEqual(result.episodes[0].steps, TURNS)


def _baseline_bonds() -> tuple[tuple[int, int], ...]:
    return tuple(
        (group_start, point)
        for group_start in range(0, POINTS, TARGET_SIZE)
        for point in range(group_start + 1, group_start + TARGET_SIZE)
    )


def _step_metrics(step: Step) -> dict[str, PolicyValue]:
    metrics = step.metrics
    assert isinstance(metrics, dict)
    return metrics


def _number_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _int_list_metric(metrics: dict[str, PolicyValue], name: str) -> list[int]:
    value = metrics[name]
    assert isinstance(value, list)
    assert all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    return [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool)
    ]


def _simple_case() -> MoleculesCase:
    positions = [(index * 101 % SPACE_SIZE, index * 307 % SPACE_SIZE) for index in range(POINTS)]
    positions[0] = (0, 0)
    positions[1] = (SPACE_SIZE - 1, 0)
    velocities = [(0, 0) for _ in range(POINTS)]
    velocities[0] = (100, 0)
    velocities[1] = (-100, 0)
    return MoleculesCase(tuple(positions), tuple(velocities))


if __name__ == "__main__":
    unittest.main()
