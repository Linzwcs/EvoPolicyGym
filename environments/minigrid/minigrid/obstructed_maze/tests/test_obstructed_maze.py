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

from minigrid_obstructed_maze import (
    ObstructedMazeBenchmark,
    ObstructedMazeConfig,
    baseline_program,
)

_SIMPLE_ACTIONS = (
    1,
    1,
    1,
    2,
    0,
    2,
    1,
    3,
    2,
    5,
    1,
    1,
    4,
    1,
    1,
    2,
    2,
    0,
    2,
    1,
    3,
)
_BLOCKED_BOX_ACTIONS = (
    1,
    2,
    0,
    3,
    1,
    1,
    4,
    1,
    2,
    0,
    5,
    3,
    1,
    1,
    2,
    1,
    2,
    0,
    5,
    1,
    1,
    4,
    1,
    1,
    2,
    2,
    1,
    2,
    0,
    3,
)


class ObstructedMazeTests(unittest.TestCase):
    def test_spec_and_split_planning(self) -> None:
        benchmark = ObstructedMazeBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "minigrid/ObstructedMaze-v0/success-rate-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 288)
        full = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="Full-v1"))
        self.assertEqual(full.spec.max_episode_steps, 3600)
        self.assertNotEqual(benchmark.spec.environment_digest, full.spec.environment_digest)
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertEqual(train, repeated)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )

    def test_spec_explains_profile_objects_actions_and_upstream_bug(self) -> None:
        legacy = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="2Dlhb-v0")).spec
        fixed = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="2Dlhb-v1")).spec
        parameters = legacy.environment_parameters
        action_notes = parameters["action_notes"]
        assert isinstance(action_notes, dict)
        drop_note = action_notes["drop"]
        natural_termination = parameters["natural_termination"]
        generation_warning = parameters["generation_warning"]
        assert isinstance(drop_note, str)
        assert isinstance(natural_termination, str)
        assert isinstance(generation_warning, str)

        self.assertEqual(parameters["target_object"], "blue_ball")
        self.assertEqual(parameters["door_blocker_object"], "green_ball")
        self.assertEqual(parameters["key_container_object"], "grey_box")
        self.assertEqual(parameters["locked_door_count"], 2)
        self.assertEqual(parameters["unlocked_door_count"], 1)
        self.assertIn("required to relocate blockers", drop_note)
        self.assertIn(
            "picking up the blue mission ball",
            natural_termination,
        )
        self.assertEqual(parameters["key_blocker_overlap_possible"], True)
        self.assertIn("structurally unsolvable", generation_warning)
        self.assertEqual(fixed.environment_parameters["key_blocker_overlap_possible"], False)
        self.assertIsNone(fixed.environment_parameters["generation_warning"])

    def test_environment_contract_and_invalid_actions(self) -> None:
        benchmark = ObstructedMazeBenchmark()
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
            self.assertIsInstance(observation["image"], TensorValue)
            self.assertEqual(observation["mission"], "pick up the blue ball")
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_real_blocked_box_chain_reports_each_distinct_milestone(self) -> None:
        benchmark = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="1Dlhb-v0"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, _BLOCKED_BOX_ACTIONS)
        pickup_metrics = _step_metrics(steps[3])
        relocation_metrics = _step_metrics(steps[6])
        box_metrics = _step_metrics(steps[10])
        key_metrics = _step_metrics(steps[11])
        door_metrics = _step_metrics(steps[18])

        self.assertEqual(pickup_metrics["blocker_picked_up_this_step"], True)
        self.assertEqual(pickup_metrics["first_blocker_pickup_step"], 4)
        self.assertEqual(relocation_metrics["blocker_dropped_this_step"], True)
        self.assertEqual(relocation_metrics["blocker_relocated"], True)
        self.assertEqual(relocation_metrics["first_blocker_relocated_step"], 7)
        self.assertEqual(box_metrics["box_opened_this_step"], True)
        self.assertEqual(box_metrics["first_box_opened_step"], 11)
        self.assertEqual(key_metrics["key_picked_up_this_step"], True)
        self.assertEqual(key_metrics["first_key_pickup_step"], 12)
        self.assertEqual(door_metrics["locked_door_opened_this_step"], True)
        self.assertEqual(door_metrics["first_locked_door_opened_step"], 19)
        self.assertFalse(steps[18].terminated)

        final = steps[-1]
        final_metrics = _step_metrics(final)
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertAlmostEqual(final.reward, 1.0 - 0.9 * 30 / 288)
        self.assertEqual(final_metrics["target_picked_up_this_step"], True)
        self.assertEqual(final_metrics["front_object_before_action"], "blue_ball")
        self.assertEqual(final_metrics["carried_object"], "blue_ball")
        self.assertEqual(final_metrics["success"], True)
        self.assertEqual(final_metrics["terminal_reason"], "success")

    def test_target_ball_is_not_misreported_as_green_blocker(self) -> None:
        benchmark = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="1Dl-v0"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, _SIMPLE_ACTIONS)
        final = steps[-1]
        final_metrics = _step_metrics(final)

        self.assertTrue(final.terminated)
        self.assertEqual(final_metrics["target_picked_up_this_step"], True)
        self.assertEqual(final_metrics["blocker_found"], False)
        self.assertEqual(final_metrics["blocker_picked_up"], False)
        self.assertEqual(final_metrics["blocker_relocated"], False)
        self.assertEqual(final_metrics["box_found"], False)
        self.assertEqual(final_metrics["box_opened"], False)
        self.assertEqual(final_metrics["key_picked_up"], True)
        self.assertEqual(final_metrics["locked_door_opened"], True)

    def test_target_pickup_with_full_hands_fails_without_termination(self) -> None:
        benchmark = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="1Dl-v0"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        actions = (*_SIMPLE_ACTIONS[:12], 6, *_SIMPLE_ACTIONS[13:])
        steps = _run_actions(benchmark, episode, actions)
        final = steps[-1]
        final_metrics = _step_metrics(final)

        self.assertFalse(final.terminated)
        self.assertFalse(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["target_in_front_before_action"], True)
        self.assertEqual(final_metrics["target_pickup_blocked_by_carried_object"], True)
        self.assertEqual(final_metrics["target_pickup_blocked_count"], 1)
        self.assertEqual(final_metrics["failed_pickup"], True)
        self.assertEqual(final_metrics["carried_object"], "red_key")
        self.assertEqual(final_metrics["task_stage"], "free_hands_for_target")

    def test_timeout_reports_horizon_and_unused_done_actions(self) -> None:
        benchmark = ObstructedMazeBenchmark(ObstructedMazeConfig(profile="1Dl-v0"))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, (6,) * 288)
        final = steps[-1]
        final_metrics = _step_metrics(final)

        self.assertFalse(final.terminated)
        self.assertTrue(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["step_count"], 288)
        self.assertEqual(final_metrics["remaining_steps"], 0)
        self.assertEqual(final_metrics["done_action_count"], 288)
        self.assertEqual(final_metrics["done_count"], 288)
        self.assertEqual(final_metrics["terminal_reason"], "time_limit")
        self.assertEqual(final_metrics["task_stage"], "time_limit")

    def test_scenario_override_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObstructedMazeBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"room_size": 8})
            )

    def test_feedback_keeps_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = ObstructedMazeBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        content = feedback.content
        assert isinstance(content, dict)

        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"scenario"', trace)
        self.assertNotIn(b'"profile"', trace)
        self.assertEqual(content["policy_failures"], 1)
        self.assertIsNone(content["mean_box_open_event_count"])

    def test_baseline_completes_public_progress_ladder(self) -> None:
        profiles = (
            "1Dl-v0",
            "1Dlh-v0",
            "1Dlhb-v0",
            "2Dl-v0",
            "2Dlh-v0",
            "2Dlhb-v0",
            "2Dlhb-v1",
            "1Q-v0",
            "1Q-v1",
            "2Q-v0",
            "2Q-v1",
            "Full-v0",
            "Full-v1",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                result = evaluate(
                    baseline_program(),
                    ObstructedMazeBenchmark(ObstructedMazeConfig(profile=profile)),
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="validation",
                        episodes=1,
                        seed=5,
                        episode_timeout_seconds=10,
                    ),
                )
                self.assertEqual(result.feedback.score, 1.0)
                self.assertEqual(result.feedback.artifacts[0].name, "trace.jsonl")
                documents = tuple(
                    json.loads(line)
                    for line in result.feedback.artifacts[0].read_bytes().splitlines()
                )
                transitions = tuple(item for item in documents if item["type"] == "transition")
                self.assertTrue(transitions)
                self.assertTrue(all("task_stage" in item["metrics"] for item in transitions))


def _run_actions(
    benchmark: ObstructedMazeBenchmark,
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
        "mission": "pick up the blue ball",
    }


if __name__ == "__main__":
    unittest.main()
