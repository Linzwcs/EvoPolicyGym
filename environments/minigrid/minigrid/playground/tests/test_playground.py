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

from minigrid_playground import PlaygroundBenchmark, baseline_program

_EARLY_COVERAGE_ACTIONS = (
    2,
    2,
    2,
    5,
    1,
    1,
    0,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    5,
    1,
    2,
    2,
    2,
    2,
    5,
    2,
    2,
    0,
    5,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    0,
    5,
    2,
    2,
    0,
    2,
    5,
    2,
)


class PlaygroundTests(unittest.TestCase):
    def test_spec_defines_custom_coverage_semantics(self) -> None:
        spec = PlaygroundBenchmark().spec
        parameters = spec.environment_parameters

        self.assertEqual(spec.id, "minigrid/Playground-v0/room-coverage-v1")
        self.assertEqual(spec.max_episode_steps, 1000)
        self.assertEqual(spec.metadata["environment"], "MiniGrid-Playground-v0")
        self.assertEqual(parameters["upstream_has_goal"], False)
        self.assertEqual(parameters["upstream_has_reward"], False)
        self.assertEqual(parameters["upstream_default_max_episode_steps"], 100)
        self.assertEqual(parameters["benchmark_time_limit"], 1000)
        self.assertEqual(parameters["initial_room_counts_toward_coverage"], True)
        self.assertEqual(parameters["initial_room_reward"], 0.0)
        self.assertEqual(parameters["new_room_reward"], 1.0)
        self.assertEqual(parameters["maximum_episode_return"], 8.0)
        coverage_definition = parameters["coverage_definition"]
        natural_termination = parameters["natural_termination"]
        action_notes = parameters["action_notes"]
        assert isinstance(coverage_definition, str)
        assert isinstance(natural_termination, str)
        assert isinstance(action_notes, dict)
        toggle_note = action_notes["toggle"]
        assert isinstance(toggle_note, str)
        self.assertIn("without exposing room coordinates", coverage_definition)
        self.assertIn("all nine rooms", natural_termination)
        self.assertIn("destroys that box", toggle_note)

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = PlaygroundBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        test = tuple(benchmark.episodes("test", seed=7, count=10))
        self.assertEqual(train, repeated)
        self.assertTrue(
            {item.environment_seed for item in train}.isdisjoint(
                item.environment_seed for item in test
            )
        )
        with self.assertRaises(ValueError):
            benchmark.make_environment(EpisodeSpec(environment_seed=1, scenario={"size": 11}))

    def test_environment_contract_and_invalid_actions(self) -> None:
        benchmark = PlaygroundBenchmark()
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
            self.assertEqual(observation["mission"], "")
            self.assertIsInstance(observation["image"], TensorValue)
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()
            environment.close()

    def test_real_trajectory_rewards_only_first_room_entries(self) -> None:
        benchmark = PlaygroundBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, _EARLY_COVERAGE_ACTIONS)
        door_metrics = _step_metrics(steps[3])

        self.assertEqual(door_metrics["door_opened_this_step"], True)
        self.assertEqual(door_metrics["door_open_event_count"], 1)
        for index, room_count in ((22, 2), (35, 3), (39, 4)):
            step = steps[index]
            metrics = _step_metrics(step)
            self.assertEqual(step.reward, 1.0)
            self.assertEqual(metrics["new_room"], True)
            self.assertEqual(metrics["rooms_visited"], room_count)
            self.assertEqual(metrics["rooms_remaining"], 9 - room_count)
            self.assertAlmostEqual(_number_metric(metrics, "room_coverage"), room_count / 9)
            self.assertEqual(metrics[f"room_{room_count}_first_entry_step"], index + 1)

        final_metrics = _step_metrics(steps[-1])
        self.assertFalse(steps[-1].terminated)
        self.assertFalse(steps[-1].truncated)
        self.assertEqual(final_metrics["rooms_visited"], 4)
        self.assertEqual(final_metrics["new_room_entry_count"], 3)
        self.assertEqual(final_metrics["cumulative_return"], 3.0)
        self.assertEqual(sum(step.reward for step in steps), 3.0)

    def test_failed_toggle_and_timeout_are_actionable(self) -> None:
        benchmark = PlaygroundBenchmark()
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        environment = benchmark.make_environment(episode)
        try:
            environment.reset()
            first = environment.step(5)
            first_metrics = _step_metrics(first)
            self.assertEqual(first_metrics["failed_toggle"], True)
            self.assertEqual(first_metrics["failed_toggle_count"], 1)
            final = first
            for _ in range(999):
                final = environment.step(6)
        finally:
            environment.close()

        final_metrics = _step_metrics(final)
        self.assertFalse(final.terminated)
        self.assertTrue(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["rooms_visited"], 1)
        self.assertAlmostEqual(_number_metric(final_metrics, "room_coverage"), 1 / 9)
        self.assertEqual(final_metrics["step_count"], 1000)
        self.assertEqual(final_metrics["remaining_steps"], 0)
        self.assertEqual(final_metrics["done_action_count"], 999)
        self.assertEqual(final_metrics["terminal_reason"], "time_limit")
        self.assertEqual(final_metrics["task_stage"], "time_limit")

    def test_feedback_privacy_and_failure_penalty(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = PlaygroundBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        content = feedback.content
        assert isinstance(content, dict)

        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b"agent_pos", trace)
        self.assertEqual(content["policy_failures"], 1)
        self.assertIsNone(content["mean_door_open_event_count"])

    def test_baseline_solves_seeded_episodes_with_bounded_semantic_trace(self) -> None:
        benchmark = PlaygroundBenchmark()
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=6,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        content = result.feedback.content
        assert isinstance(content, dict)
        self.assertEqual(result.feedback.score, 1.0)
        self.assertEqual(content["mean_room_coverage"], 1.0)
        self.assertEqual(content["mean_return"], 8.0)
        self.assertEqual(content["room_9_reached_rate"], 1.0)
        self.assertEqual(content["door_opened_rate"], 1.0)
        self.assertGreater(_number_metric(content, "object_picked_up_rate"), 0.0)
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        episodes = tuple(item for item in documents if item["type"] == "episode")
        transitions = tuple(item for item in documents if item["type"] == "transition")
        self.assertEqual(len(episodes), 4)
        self.assertTrue(all(item["room_coverage"] == 1.0 for item in episodes))
        self.assertTrue(all(len(item["room_first_entry_steps"]) == 8 for item in episodes))
        self.assertTrue(all(item["traced_steps"] <= 160 for item in episodes))
        self.assertTrue(all("visible_objects" in item["next_observation"] for item in transitions))
        self.assertTrue(all("steps_since_new_room" in item["metrics"] for item in transitions))


def _run_actions(
    benchmark: PlaygroundBenchmark,
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


def _number_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(dtype="uint8", shape=(7, 7, 3), data=bytes(147)),
        "direction": 0,
        "mission": "",
    }


if __name__ == "__main__":
    unittest.main()
