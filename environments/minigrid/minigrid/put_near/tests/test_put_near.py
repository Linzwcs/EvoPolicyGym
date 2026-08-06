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
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_put_near import PutNearBenchmark, PutNearConfig, baseline_program

_SUCCESS_ACTIONS = (1, 2, 1, 2, 2, 0, 3, 1, 1, 2, 2, 2, 1, 4)
_CORRECT_PICKUP_PREFIX = (1, 2, 1, 2, 2, 0, 3)


class PutNearBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = PutNearBenchmark()
        small = PutNearBenchmark(PutNearConfig(profile="6x6-N2"))

        self.assertEqual(default.spec.id, "minigrid/PutNear-v0/mean-return-v1")
        self.assertEqual(default.spec.max_episode_steps, 40)
        self.assertEqual(small.spec.max_episode_steps, 30)
        self.assertNotEqual(default.spec.environment_digest, small.spec.environment_digest)
        self.assertEqual(default.spec.environment_parameters["profile"], "8x8-N3")
        self.assertEqual(default.spec.environment_parameters["object_count"], 3)

    def test_spec_explains_exact_terminal_and_box_toggle_semantics(self) -> None:
        parameters = PutNearBenchmark().spec.environment_parameters
        action_notes = parameters["action_notes"]
        assert isinstance(action_notes, dict)
        pickup_note = action_notes["pick_up"]
        drop_note = action_notes["drop"]
        toggle_note = action_notes["toggle"]
        natural_termination = parameters["natural_termination"]
        assert isinstance(pickup_note, str)
        assert isinstance(drop_note, str)
        assert isinstance(toggle_note, str)
        assert isinstance(natural_termination, str)

        self.assertEqual(parameters["see_through_walls"], True)
        self.assertEqual(parameters["initial_objects_are_non_adjacent"], True)
        self.assertIn("terminates immediately", pickup_note)
        self.assertIn(
            "any drop attempted while carrying terminates", drop_note
        )
        self.assertIn("destroys it without terminating", toggle_note)
        self.assertIn("wrong-object pickup", natural_termination)
        self.assertEqual(
            parameters["success_reward_formula"],
            "1 - 0.9*step_count/max_episode_steps",
        )

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            PutNearConfig(profile="8x8")
        with self.assertRaises(ValueError):
            PutNearConfig(profile=8)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = PutNearBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        validation = tuple(benchmark.episodes("validation", seed=7, count=10))

        self.assertEqual(train, repeated)
        self.assertEqual(len({item.environment_seed for item in train}), 10)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in validation
            )
        )
        self.assertTrue(all(item.scenario is None for item in train))

    def test_environment_is_conformant_and_rejects_invalid_actions(self) -> None:
        benchmark = PutNearBenchmark(PutNearConfig(profile="6x6-N2"))
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 5, 6),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(set(observation), {"image", "direction", "mission"})
            self.assertIsInstance(observation["image"], TensorValue)
            self.assertTrue(str(observation["mission"]).startswith("put the "))
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (-1, 7, True, 2.0, [2])
        for invalid in invalid_actions:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_real_success_requires_actual_adjacent_drop(self) -> None:
        benchmark = PutNearBenchmark(PutNearConfig(profile="8x8-N3"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, _SUCCESS_ACTIONS)

        pickup = steps[6]
        pickup_metrics = _step_metrics(pickup)
        self.assertFalse(pickup.terminated)
        self.assertEqual(pickup_metrics["move_object_label"], "yellow_ball")
        self.assertEqual(pickup_metrics["target_object_label"], "red_box")
        self.assertEqual(pickup_metrics["correct_object_picked_up_this_step"], True)
        self.assertEqual(pickup_metrics["correct_object_pickup_step"], 7)
        self.assertEqual(pickup_metrics["carrying_move_object"], True)

        final = steps[-1]
        final_metrics = _step_metrics(final)
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertAlmostEqual(final.reward, 1.0 - 0.9 * 14 / 40)
        self.assertEqual(final_metrics["valid_success_drop_before_action"], True)
        self.assertEqual(final_metrics["object_dropped_this_step"], True)
        self.assertEqual(final_metrics["misplaced_drop"], False)
        self.assertEqual(final_metrics["blocked_terminal_drop"], False)
        self.assertEqual(final_metrics["success"], True)
        self.assertEqual(final_metrics["terminal_reason"], "success")

    def test_wrong_pickup_terminates_and_names_carried_decoy(self) -> None:
        benchmark = PutNearBenchmark(PutNearConfig(profile="8x8-N3"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        final = _run_actions(benchmark, episode, (1, 2, 3))[-1]
        final_metrics = _step_metrics(final)

        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["wrong_object_picked_up"], True)
        self.assertEqual(final_metrics["carried_object"], "yellow_box")
        self.assertEqual(final_metrics["terminal_reason"], "wrong_object_pickup")

    def test_misplaced_and_blocked_drops_have_distinct_outcomes(self) -> None:
        benchmark = PutNearBenchmark(PutNearConfig(profile="8x8-N3"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        misplaced = _run_actions(
            benchmark,
            episode,
            (*_CORRECT_PICKUP_PREFIX, 4),
        )[-1]
        blocked = _run_actions(
            benchmark,
            episode,
            (*_CORRECT_PICKUP_PREFIX, 1, 4),
        )[-1]
        misplaced_metrics = _step_metrics(misplaced)
        blocked_metrics = _step_metrics(blocked)

        self.assertTrue(misplaced.terminated)
        self.assertEqual(misplaced_metrics["object_dropped_this_step"], True)
        self.assertEqual(misplaced_metrics["misplaced_drop"], True)
        self.assertEqual(misplaced_metrics["failed_drop"], False)
        self.assertEqual(misplaced_metrics["terminal_reason"], "misplaced_drop")
        self.assertTrue(blocked.terminated)
        self.assertEqual(blocked_metrics["object_dropped_this_step"], False)
        self.assertEqual(blocked_metrics["blocked_terminal_drop"], True)
        self.assertEqual(blocked_metrics["failed_drop"], True)
        self.assertEqual(blocked_metrics["carried_object"], "yellow_ball")
        self.assertEqual(blocked_metrics["terminal_reason"], "blocked_drop")

    def test_toggling_mission_box_destroys_it_without_termination(self) -> None:
        benchmark = PutNearBenchmark(PutNearConfig(profile="8x8-N3"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, (0, 2, 2, 5))
        final = steps[-1]
        final_metrics = _step_metrics(final)

        self.assertFalse(final.terminated)
        self.assertFalse(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["front_object_before_action"], "red_box")
        self.assertEqual(final_metrics["box_destroyed_this_step"], True)
        self.assertEqual(final_metrics["target_object_destroyed_this_step"], True)
        self.assertEqual(final_metrics["mission_object_destroyed"], True)
        self.assertEqual(final_metrics["mission_object_destroyed_step"], 4)
        self.assertEqual(final_metrics["task_stage"], "mission_object_destroyed")

    def test_timeout_is_distinct_from_natural_failures(self) -> None:
        benchmark = PutNearBenchmark(PutNearConfig(profile="8x8-N3"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        final = _run_actions(benchmark, episode, (6,) * 40)[-1]
        final_metrics = _step_metrics(final)

        self.assertFalse(final.terminated)
        self.assertTrue(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["step_count"], 40)
        self.assertEqual(final_metrics["remaining_steps"], 0)
        self.assertEqual(final_metrics["done_action_count"], 40)
        self.assertEqual(final_metrics["terminal_reason"], "time_limit")

    def test_episode_scenario_cannot_override_benchmark_configuration(self) -> None:
        benchmark = PutNearBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(environment_seed=1, scenario={"profile": "6x6-N2"})
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = PutNearBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = benchmark.feedback((failed,))
        trace = feedback.artifacts[0]
        content = feedback.content
        assert isinstance(content, dict)

        self.assertEqual(feedback.score, 0.0)
        self.assertEqual(trace.name, "trace.jsonl")
        self.assertNotIn(b"environment_seed", trace.read_bytes())
        self.assertNotIn(b"policy_seed", trace.read_bytes())
        self.assertNotIn(b'"profile"', trace.read_bytes())
        self.assertEqual(content["policy_failures"], 1)
        self.assertEqual(content["wrong_object_picked_up_episodes"], 0)
        self.assertEqual(content["misplaced_drop_episodes"], 0)
        self.assertIsNone(content["mean_pickup_event_count"])

    def test_baseline_solves_every_public_profile_and_publishes_trace(self) -> None:
        for profile in ("6x6-N2", "8x8-N3"):
            with self.subTest(profile=profile):
                benchmark = PutNearBenchmark(PutNearConfig(profile=profile))
                result = evaluate(
                    baseline_program(),
                    benchmark,
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="validation",
                        episodes=8,
                        seed=5,
                        episode_timeout_seconds=10,
                    ),
                )

                content = result.feedback.content
                assert isinstance(content, dict)
                self.assertEqual(
                    result.benchmark_id,
                    "minigrid/PutNear-v0/mean-return-v1",
                )
                self.assertEqual(result.environment_digest, benchmark.spec.environment_digest)
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertEqual(content["wrong_object_picked_up_rate"], 0.0)
                self.assertEqual(content["misplaced_drop_rate"], 0.0)
                self.assertEqual(content["blocked_terminal_drop_rate"], 0.0)
                trace = result.feedback.artifacts[0]
                documents = tuple(json.loads(line) for line in trace.read_bytes().splitlines())
                transitions = tuple(
                    document for document in documents if document["type"] == "transition"
                )
                self.assertEqual(trace.name, "trace.jsonl")
                self.assertTrue(transitions)
                self.assertTrue(
                    all(
                        "move_object" in item["next_observation"]
                        and "target_object" in item["next_observation"]
                        for item in transitions
                    )
                )
                self.assertTrue(
                    all("valid_success_drop_available" in item["metrics"] for item in transitions)
                )


def _run_actions(
    benchmark: PutNearBenchmark,
    episode: EpisodeSpec,
    actions: tuple[int, ...],
) -> tuple[Step, ...]:
    environment = benchmark.make_environment(episode)
    steps: list[Step] = []
    try:
        environment.reset()
        for action in actions:
            steps.append(environment.step(action))
    finally:
        environment.close()
    return tuple(steps)


def _step_metrics(step: Step) -> dict[str, PolicyValue]:
    metrics = step.metrics
    assert isinstance(metrics, dict)
    return metrics


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(dtype="uint8", shape=(7, 7, 3), data=bytes(147)),
        "direction": 0,
        "mission": "put the purple ball near the red key",
    }


if __name__ == "__main__":
    unittest.main()
