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

from warehouseman import WarehousemanBenchmark, baseline_program
from warehouseman.benchmark import FAILURE_COST
from warehouseman.environment import WarehousemanEnvironment
from warehouseman.programs.baseline.policy import _solve
from warehouseman.simulation import (
    GENERATOR_ID,
    MAX_COLUMNS,
    MAX_INSTRUCTION_CHARACTERS,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
    InvalidInstruction,
    WarehouseCase,
    WarehouseSimulation,
    generate_case,
    normalized_cost,
)


class WarehousemanBenchmarkTests(unittest.TestCase):
    def test_spec_describes_independent_full_range_integration(self) -> None:
        spec = WarehousemanBenchmark().spec

        self.assertEqual(
            spec.id,
            "codechef/WAREHOUS/Warehouseman/mean-normalized-cost-v1",
        )
        self.assertEqual(spec.max_episode_steps, 1)
        self.assertEqual(spec.primary_metric, "mean_normalized_cost")
        self.assertEqual(spec.score_direction, "minimize")
        self.assertEqual(spec.environment_parameters["generator"], GENERATOR_ID)
        self.assertEqual(spec.environment_parameters["minimum_rows"], MIN_ROWS)
        self.assertEqual(spec.environment_parameters["maximum_rows"], MAX_ROWS)
        self.assertEqual(
            spec.environment_parameters["minimum_columns"],
            MIN_COLUMNS,
        )
        self.assertEqual(
            spec.environment_parameters["maximum_columns"],
            MAX_COLUMNS,
        )
        self.assertEqual(
            spec.environment_parameters["instruction_limit"],
            MAX_INSTRUCTION_CHARACTERS,
        )
        self.assertEqual(spec.environment_parameters["failure_cost"], FAILURE_COST)
        solution_atomicity = spec.environment_parameters["solution_atomicity"]
        score_formula = spec.environment_parameters["score_formula"]
        handling_lower_bound = spec.environment_parameters["handling_lower_bound"]
        assert isinstance(solution_atomicity, str)
        assert isinstance(score_formula, str)
        assert isinstance(handling_lower_bound, str)
        self.assertIn("one Action", solution_atomicity)
        self.assertIn("instruction_characters", score_formula)
        self.assertIn("at least 6 characters", handling_lower_bound)
        self.assertEqual(spec.metadata["implementation"], "independent")
        self.assertEqual(
            spec.metadata["upstream_material_license"],
            "CodeChef terms; not redistributed",
        )
        self.assertFalse(spec.metadata["upstream_code_included"])
        self.assertFalse(spec.metadata["upstream_inputs_included"])
        self.assertFalse(spec.metadata["upstream_assets_included"])

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = WarehousemanBenchmark()

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
            (19, 19, 211),
            (17, 14, 195),
            (20, 14, 267),
        )
        generated = tuple(generate_case(seed) for seed in range(3))

        self.assertEqual(
            tuple((case.rows, case.columns, case.arrivals[-1]) for case in generated),
            expected,
        )
        for case in generated:
            self.assertGreaterEqual(case.rows, MIN_ROWS)
            self.assertLessEqual(case.rows, MAX_ROWS)
            self.assertGreaterEqual(case.columns, MIN_COLUMNS)
            self.assertLessEqual(case.columns, MAX_COLUMNS)
            self.assertEqual(
                set(case.arrivals),
                set(range(1, case.rows * case.columns)),
            )
            self.assertNotEqual(case.arrivals[-1], 1)

    def test_initial_observation_is_public_input_without_case_identity(
        self,
    ) -> None:
        environment = WarehousemanEnvironment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
        finally:
            environment.close()

        self.assertIsInstance(observation, dict)
        assert isinstance(observation, dict)
        self.assertEqual(
            set(observation),
            {"rows", "columns", "arrivals", "instruction_limit"},
        )
        self.assertEqual(
            observation["instruction_limit"],
            MAX_INSTRUCTION_CHARACTERS,
        )
        encoded = json.dumps(observation, sort_keys=True)
        self.assertNotIn("seed", encoded)
        self.assertNotIn("split", encoded)
        self.assertNotIn("scenario", encoded)

    def test_simulator_accepts_constructive_solution_and_exact_score(
        self,
    ) -> None:
        case = _ordered_case(6, 6)
        instructions = _solve(case.rows, case.columns, case.arrivals)
        simulation = WarehouseSimulation(case)

        simulation.execute(instructions)

        self.assertLessEqual(len(instructions), MAX_INSTRUCTION_CHARACTERS)
        self.assertEqual(simulation.picks, len(case.arrivals))
        self.assertEqual(simulation.drops, len(case.arrivals))
        self.assertGreater(simulation.loads, 0)
        self.assertGreater(simulation.unloads, 0)
        self.assertEqual(
            normalized_cost(len(instructions), 6, 6),
            (len(instructions) + 2) / 11 - 72 + 20,
        )

    def test_constructive_solution_covers_maximum_dimensions(self) -> None:
        arrivals = (1, *range(MAX_ROWS * MAX_COLUMNS - 1, 1, -1))
        case = WarehouseCase(MAX_ROWS, MAX_COLUMNS, arrivals)
        instructions = _solve(case.rows, case.columns, case.arrivals)

        self.assertLess(len(instructions), MAX_INSTRUCTION_CHARACTERS)
        WarehouseSimulation(case).execute(instructions)

    def test_simulator_rejects_illegal_and_incomplete_instructions(self) -> None:
        case = _ordered_case(6, 6)
        for instructions in (
            "",
            "N",
            "P",
            "D",
            "L",
            "UX",
            "🙂",
            "E" * (MAX_INSTRUCTION_CHARACTERS + 1),
        ):
            with self.assertRaises(
                InvalidInstruction,
                msg=repr(instructions[:20]),
            ):
                WarehouseSimulation(case).execute(instructions)

    def test_environment_action_is_strict_and_atomic(self) -> None:
        episode = EpisodeSpec(environment_seed=31)
        case = generate_case(episode.environment_seed)
        valid = _solve(case.rows, case.columns, case.arrivals)
        environment = WarehousemanEnvironment(episode)
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step({"instructions": valid})
            with self.assertRaises(InvalidAction):
                environment.step("P")
            terminal = environment.step(valid)
        finally:
            environment.close()

        self.assertTrue(terminal.terminated)
        self.assertFalse(terminal.truncated)
        self.assertEqual(terminal.observation, {"status": "completed"})
        self.assertIsInstance(terminal.metrics, dict)
        assert isinstance(terminal.metrics, dict)
        self.assertEqual(
            terminal.reward,
            terminal.metrics["normalized_cost"],
        )
        shipment_count = case.rows * case.columns - 1
        self.assertEqual(terminal.metrics["shipments"], shipment_count)
        self.assertEqual(terminal.metrics["picks"], shipment_count)
        self.assertEqual(terminal.metrics["drops"], shipment_count)
        self.assertEqual(terminal.metrics["loads"], terminal.metrics["unloads"])
        self.assertEqual(
            terminal.metrics["minimum_handling_characters"],
            6 * shipment_count,
        )
        self.assertEqual(
            terminal.metrics["instruction_characters"],
            _number_metric(terminal.metrics, "moves")
            + _number_metric(terminal.metrics, "handling_characters"),
        )
        self.assertEqual(
            terminal.metrics["excess_characters_above_handling_lower_bound"],
            _number_metric(terminal.metrics, "moves")
            + 4 * _number_metric(terminal.metrics, "relocation_cycles"),
        )
        self.assertAlmostEqual(
            _number_metric(terminal.metrics, "movement_character_fraction")
            + _number_metric(terminal.metrics, "handling_character_fraction"),
            1.0,
        )
        self.assertEqual(
            terminal.metrics["terminal_reason"],
            "complete_valid_solution",
        )

    def test_environment_lifecycle_and_deterministic_replay(self) -> None:
        episode = EpisodeSpec(environment_seed=123)
        case = generate_case(episode.environment_seed)
        action = _solve(case.rows, case.columns, case.arrivals)
        report = check_benchmark(
            WarehousemanBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    episode=episode,
                    actions=(action,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        before_reset = WarehousemanEnvironment(episode)
        with self.assertRaises(RuntimeError):
            before_reset.step(action)
        before_reset.close()
        before_reset.close()

        repeated_reset = WarehousemanEnvironment(episode)
        repeated_reset.reset()
        with self.assertRaises(RuntimeError):
            repeated_reset.reset()
        repeated_reset.close()

        completed = WarehousemanEnvironment(episode)
        completed.reset()
        completed.step(action)
        with self.assertRaises(RuntimeError):
            completed.step(action)
        completed.close()

    def test_feedback_publishes_bounded_semantic_diagnostics(self) -> None:
        episode = EpisodeSpec(environment_seed=123)
        case = generate_case(episode.environment_seed)
        action = _solve(case.rows, case.columns, case.arrivals)
        environment = WarehousemanEnvironment(episode)
        try:
            initial = environment.reset()
            step = environment.step(action)
        finally:
            environment.close()
        record = EpisodeRecord(
            episode=episode,
            policy_seed=456,
            initial_observation=initial,
            transitions=(Transition(action=action, step=step),),
        )

        feedback = WarehousemanBenchmark().feedback((record,))

        self.assertEqual(feedback.score, step.reward)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["completed"], 1)
        diagnostics = feedback.artifacts[0].read_bytes()
        self.assertLess(len(diagnostics), 2_000)
        self.assertNotIn(b"environment_seed", diagnostics)
        self.assertNotIn(b"policy_seed", diagnostics)
        self.assertNotIn(b"arrivals", diagnostics)
        self.assertNotIn(action[:100].encode("ascii"), diagnostics)
        document = json.loads(diagnostics)
        self.assertEqual(document["status"], "completed")
        self.assertEqual(
            document["operations"]["instruction_characters"],
            len(action),
        )
        self.assertEqual(
            document["operations"]["instruction_characters"],
            document["operations"]["moves"] + document["operations"]["handling_characters"],
        )
        self.assertEqual(
            document["operations"]["relocation_characters"],
            4 * document["operations"]["relocation_cycles"],
        )
        self.assertGreater(document["efficiency"]["characters_per_shipment"], 6.0)
        self.assertAlmostEqual(
            document["efficiency"]["movement_character_fraction"]
            + document["efficiency"]["handling_character_fraction"],
            1.0,
        )
        self.assertEqual(
            feedback.content["mean_characters_per_shipment"],
            _step_metrics(step)["characters_per_shipment"],
        )
        self.assertEqual(
            feedback.content["mean_relocation_cycles"],
            _number_metric(_step_metrics(step), "relocation_cycles"),
        )
        self.assertEqual(
            feedback.content["mean_instruction_budget_fraction"],
            len(action) / MAX_INSTRUCTION_CHARACTERS,
        )

    def test_feedback_assigns_failure_cost_without_private_evidence(
        self,
    ) -> None:
        environment = WarehousemanEnvironment(EpisodeSpec(environment_seed=123))
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

        feedback = WarehousemanBenchmark().feedback((failed,))

        self.assertEqual(feedback.score, FAILURE_COST)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        diagnostics = feedback.artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", diagnostics)
        self.assertNotIn(b"policy_seed", diagnostics)
        self.assertNotIn(b"arrivals", diagnostics)
        document = json.loads(diagnostics)
        self.assertEqual(document["status"], "policy_failed")

    def test_baseline_program_completes_direct_evaluation(self) -> None:
        benchmark = WarehousemanBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=3,
                seed=5,
                episode_timeout_seconds=30,
            ),
        )

        self.assertEqual(result.benchmark_id, benchmark.spec.id)
        self.assertEqual(
            result.environment_digest,
            benchmark.spec.environment_digest,
        )
        self.assertLess(result.feedback.score, FAILURE_COST)
        self.assertTrue(all(episode.status == "completed" for episode in result.episodes))
        diagnostics = [
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        ]
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(
            all(
                document["operations"]["instruction_characters"] <= MAX_INSTRUCTION_CHARACTERS
                for document in diagnostics
            )
        )


def _ordered_case(rows: int, columns: int) -> WarehouseCase:
    arrivals = list(range(1, rows * columns))
    arrivals[-1], arrivals[-2] = arrivals[-2], arrivals[-1]
    return WarehouseCase(rows, columns, tuple(arrivals))


def _step_metrics(step: Step) -> dict[str, PolicyValue]:
    metrics = step.metrics
    assert isinstance(metrics, dict)
    return metrics


def _number_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


if __name__ == "__main__":
    unittest.main()
