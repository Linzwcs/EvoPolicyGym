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

from apple_incremental_game import (
    AppleIncrementalGameBenchmark,
    baseline_program,
)
from apple_incremental_game.environment import (
    AppleIncrementalGameEnvironment,
)
from apple_incremental_game.simulation import (
    GENERATOR_ID,
    INITIAL_APPLES,
    LEVELS,
    MACHINE_IDS,
    TURNS,
    AppleCase,
    AppleSimulation,
    InvalidUpgrade,
    final_score,
    generate_case,
)


class AppleIncrementalGameBenchmarkTests(unittest.TestCase):
    def test_spec_describes_independent_full_task(self) -> None:
        spec = AppleIncrementalGameBenchmark().spec

        self.assertEqual(
            spec.id,
            "atcoder/AHC058/AppleIncrementalGame/mean-log2-score-v1",
        )
        self.assertEqual(spec.max_episode_steps, TURNS)
        self.assertEqual(spec.primary_metric, "mean_log2_score")
        self.assertEqual(spec.score_direction, "maximize")
        self.assertEqual(spec.environment_parameters["generator"], GENERATOR_ID)
        self.assertEqual(spec.environment_parameters["machine_ids"], MACHINE_IDS)
        self.assertEqual(spec.environment_parameters["levels"], LEVELS)
        self.assertEqual(spec.environment_parameters["turns"], TURNS)
        self.assertEqual(
            spec.environment_parameters["initial_apples"],
            INITIAL_APPLES,
        )
        turn_order = spec.environment_parameters["turn_order"]
        reward_semantics = spec.environment_parameters["reward_semantics"]
        score_formula = spec.environment_parameters["score_formula"]
        invalid_upgrade_semantics = spec.environment_parameters["invalid_upgrade_semantics"]
        assert isinstance(turn_order, str)
        assert isinstance(reward_semantics, str)
        assert isinstance(score_formula, str)
        assert isinstance(invalid_upgrade_semantics, str)
        self.assertIn("levels 0, 1, 2, 3", turn_order)
        self.assertIn("0 for turns 1-499", reward_semantics)
        self.assertIn("log2", score_formula)
        self.assertIn("never clipped", invalid_upgrade_semantics)
        self.assertEqual(spec.metadata["implementation"], "independent")
        self.assertEqual(
            spec.metadata["upstream_tool_license"],
            "not declared; not redistributed",
        )
        self.assertEqual(
            spec.metadata["upstream_tool_archive_revision"],
            "UpvAVdx6",
        )
        self.assertFalse(spec.metadata["upstream_code_included"])
        self.assertFalse(spec.metadata["upstream_inputs_included"])
        self.assertFalse(spec.metadata["upstream_assets_included"])

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = AppleIncrementalGameBenchmark()

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
            (
                (1, 2, 4, 4, 5, 6, 16, 23, 36, 66),
                (1, 112, 35),
                (42_831_827_916, 31_518_287_128, 58_249_564_479),
            ),
            (
                (1, 1, 3, 5, 10, 11, 14, 20, 28, 29),
                (1, 10, 13),
                (9_978_273_203, 29_882_751_050, 4_497_035_941),
            ),
            (
                (1, 2, 5, 7, 8, 9, 12, 16, 49, 63),
                (1, 3, 30),
                (32_285_649_828, 391_759_891_791, 17_219_159_720),
            ),
        )
        generated = tuple(generate_case(seed) for seed in range(3))

        self.assertEqual(
            tuple(
                (
                    case.capacities,
                    case.costs[0][:3],
                    case.costs[3][-3:],
                )
                for case in generated
            ),
            expected,
        )
        for case in generated:
            self.assertEqual(len(case.capacities), MACHINE_IDS)
            self.assertEqual(case.capacities, tuple(sorted(case.capacities)))
            self.assertTrue(all(1 <= value <= 100 for value in case.capacities))
            self.assertEqual(len(case.costs), LEVELS)
            self.assertTrue(all(len(row) == MACHINE_IDS for row in case.costs))
            self.assertEqual(case.costs[0][0], 1)

    def test_simulation_uses_level_order_and_exact_costs(self) -> None:
        simulation = AppleSimulation(_simple_case())

        simulation.step((0, 0))
        self.assertEqual(simulation.apples, 1)
        self.assertEqual(simulation.powers[0][0], 1)
        simulation.step((1, 0))
        self.assertEqual(simulation.apples, 1)
        self.assertEqual(simulation.counts[0][0], 2)
        simulation.step(None)
        self.assertEqual(simulation.apples, 3)
        self.assertEqual(simulation.counts[0][0], 3)
        self.assertEqual(simulation.turn, 3)
        self.assertEqual(simulation.upgrades, 2)

    def test_invalid_upgrade_is_rejected_before_state_changes(self) -> None:
        simulation = AppleSimulation(_simple_case(cost=10))

        with self.assertRaises(InvalidUpgrade):
            simulation.step((0, 1))

        self.assertEqual(simulation.turn, 0)
        self.assertEqual(simulation.apples, INITIAL_APPLES)
        self.assertEqual(simulation.upgrades, 0)
        simulation.step((0, 0))
        self.assertEqual(simulation.turn, 1)

    def test_final_score_matches_public_formula(self) -> None:
        self.assertEqual(final_score(1), 0)
        self.assertEqual(final_score(2), 100_000)
        self.assertEqual(final_score(8), 300_000)

    def test_initial_observation_hides_case_identity(self) -> None:
        environment = AppleIncrementalGameEnvironment(EpisodeSpec(environment_seed=123))
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
                "apples",
                "machines",
                "powers",
                "initial",
            },
        )
        self.assertEqual(observation["turn"], 0)
        self.assertEqual(observation["turns_remaining"], TURNS)
        initial = observation["initial"]
        self.assertIsInstance(initial, dict)
        assert isinstance(initial, dict)
        self.assertEqual(set(initial), {"capacities", "costs"})
        encoded = json.dumps(observation, sort_keys=True)
        self.assertNotIn("seed", encoded)
        self.assertNotIn("split", encoded)
        self.assertNotIn("scenario", encoded)

    def test_environment_rejects_malformed_and_unaffordable_actions(
        self,
    ) -> None:
        invalid_actions: tuple[PolicyValue, ...] = (
            [],
            {"upgrade": None},
            {"upgrade": [0]},
            {"upgrade": [0, 0], "extra": True},
            {"upgrade": [True, 0]},
            {"upgrade": [0.0, 0]},
            {"upgrade": [-1, 0]},
            {"upgrade": [LEVELS, 0]},
            {"upgrade": [0, MACHINE_IDS]},
        )
        for action in invalid_actions:
            environment = AppleIncrementalGameEnvironment(EpisodeSpec(environment_seed=17))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction, msg=repr(action)):
                    environment.step(action)
            finally:
                environment.close()

        environment = AppleIncrementalGameEnvironment(EpisodeSpec(environment_seed=17))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step({"upgrade": [3, 9]})
            valid = environment.step({"upgrade": [0, 0]})
        finally:
            environment.close()
        self.assertFalse(valid.done)
        self.assertIsInstance(valid.metrics, dict)
        assert isinstance(valid.metrics, dict)
        self.assertEqual(valid.metrics["turn"], 1)

    def test_environment_lifecycle_and_deterministic_replay(self) -> None:
        episode = EpisodeSpec(environment_seed=123)
        actions: tuple[PolicyValue, ...] = (
            {"upgrade": [0, 0]},
            *((None,) * (TURNS - 1)),
        )
        report = check_benchmark(
            AppleIncrementalGameBenchmark(),
            fixtures=(BenchmarkFixture(episode=episode, actions=actions),),
        )
        self.assertTrue(report.passed, report.issues)

        before_reset = AppleIncrementalGameEnvironment(episode)
        with self.assertRaises(RuntimeError):
            before_reset.step(None)
        before_reset.close()
        before_reset.close()

        repeated_reset = AppleIncrementalGameEnvironment(episode)
        repeated_reset.reset()
        with self.assertRaises(RuntimeError):
            repeated_reset.reset()
        repeated_reset.close()

    def test_real_upgrade_and_wait_turns_report_economic_diagnostics(self) -> None:
        environment = AppleIncrementalGameEnvironment(EpisodeSpec(environment_seed=17))
        try:
            environment.reset()
            upgraded = environment.step({"upgrade": [0, 0]})
            waited = environment.step(None)
        finally:
            environment.close()

        upgraded_metrics = _step_metrics(upgraded)
        waited_metrics = _step_metrics(waited)
        self.assertEqual(upgraded.reward, 0.0)
        self.assertEqual(upgraded_metrics["upgrade_made_this_turn"], True)
        self.assertEqual(upgraded_metrics["upgrade_level"], 0)
        self.assertEqual(upgraded_metrics["upgrade_machine_id"], 0)
        self.assertEqual(upgraded_metrics["upgrade_cost"], 1)
        self.assertEqual(upgraded_metrics["apples_before_action"], 1)
        self.assertEqual(upgraded_metrics["apples_after_purchase"], 0)
        self.assertEqual(upgraded_metrics["production_this_turn"], 1)
        self.assertEqual(upgraded_metrics["apple_net_change_this_turn"], 0)
        self.assertEqual(upgraded_metrics["level_zero_production_rate"], 1)
        self.assertEqual(upgraded_metrics["total_spent"], 1)
        self.assertEqual(upgraded_metrics["total_produced"], 1)
        self.assertEqual(upgraded_metrics["score_if_ended_now"], 0)
        self.assertEqual(upgraded_metrics["upgrade_counts_by_level"], [1, 0, 0, 0])
        self.assertEqual(upgraded_metrics["power_totals_by_level"], [1, 0, 0, 0])

        self.assertEqual(waited_metrics["wait_action_this_turn"], True)
        self.assertEqual(waited_metrics["production_this_turn"], 1)
        self.assertEqual(waited_metrics["apple_net_change_this_turn"], 1)
        self.assertEqual(waited_metrics["apples"], 2)
        self.assertEqual(waited_metrics["wait_turn_count"], 1)
        self.assertEqual(waited_metrics["wait_turn_fraction"], 0.5)
        self.assertEqual(waited_metrics["cheapest_affordable_upgrade_cost"], 2)
        self.assertGreaterEqual(_number_metric(waited_metrics, "affordable_upgrade_count"), 1)

    def test_feedback_publishes_trace_and_penalizes_failure(self) -> None:
        episode = EpisodeSpec(environment_seed=123)
        environment = AppleIncrementalGameEnvironment(episode)
        transitions: list[Transition] = []
        try:
            initial = environment.reset()
            actions: tuple[PolicyValue, ...] = (
                {"upgrade": [0, 0]},
                *((None,) * (TURNS - 1)),
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

        feedback = AppleIncrementalGameBenchmark().feedback((completed, failed))

        self.assertGreater(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["completed"], 1)
        self.assertEqual(feedback.content["policy_failures"], 1)
        trace = feedback.artifacts[0].read_bytes()
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b"scenario", trace)
        documents = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(documents[0]["status"], "completed")
        self.assertEqual(documents[-1]["status"], "policy_failed")
        self.assertEqual(
            sum(document["type"] == "transition" for document in documents),
            TURNS,
        )
        first_transition = documents[1]
        self.assertEqual(first_transition["metrics"]["upgrade_cost"], 1)
        self.assertEqual(first_transition["metrics"]["production_this_turn"], 1)
        self.assertEqual(
            first_transition["metrics"]["upgrade_counts_by_level"],
            [1, 0, 0, 0],
        )
        self.assertEqual(feedback.content["mean_total_spent"], 1.0)
        self.assertEqual(feedback.content["mean_total_produced"], 500.0)
        self.assertEqual(feedback.content["mean_wait_turn_fraction"], 0.998)
        self.assertEqual(feedback.content["mean_level_0_upgrade_count"], 1.0)
        self.assertEqual(feedback.content["mean_level_1_upgrade_count"], 0.0)

    def test_baseline_program_completes_direct_evaluation(self) -> None:
        benchmark = AppleIncrementalGameBenchmark()
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
        self.assertGreater(result.feedback.score, 0.0)
        self.assertTrue(all(episode.status == "completed" for episode in result.episodes))
        self.assertTrue(all(episode.steps == TURNS for episode in result.episodes))


def _simple_case(*, cost: int = 1) -> AppleCase:
    capacities = tuple(range(1, MACHINE_IDS + 1))
    costs = tuple(
        tuple(1 if level == 0 and machine_id == 0 else cost for machine_id in range(MACHINE_IDS))
        for level in range(LEVELS)
    )
    return AppleCase(capacities, costs)


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
