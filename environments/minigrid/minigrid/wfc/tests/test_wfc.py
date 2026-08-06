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

from minigrid_wfc import WFC_PROFILES, WFCBenchmark, WFCConfig, baseline_program

_SUCCESS_ACTIONS = (
    1,
    2,
    1,
    1,
    1,
    2,
    2,
    2,
    1,
    0,
    2,
    2,
    2,
    0,
    0,
    2,
    2,
    2,
    0,
    2,
    2,
    2,
    0,
    2,
    2,
    1,
    2,
    0,
    2,
    1,
    2,
    2,
    0,
    2,
    2,
)


class WFCTests(unittest.TestCase):
    def test_spec_defines_dynamic_generation_and_navigation_semantics(self) -> None:
        spec = WFCBenchmark().spec
        parameters = spec.environment_parameters

        self.assertEqual(spec.id, "minigrid/WFC-v0/mean-return-v1")
        self.assertEqual(spec.max_episode_steps, 2500)
        self.assertEqual(spec.metadata["environment"], "MiniGrid-WFC-MazeSimple-v0")
        self.assertEqual(parameters["procedurally_generated_each_episode"], True)
        self.assertEqual(parameters["pattern_source"], "packaged preset PNG")
        self.assertEqual(parameters["ensure_connected"], True)
        self.assertEqual(parameters["wfc_attempt_limit_per_reset"], 1000)
        self.assertEqual(parameters["deterministic_generation_reset_retries"], 8)
        self.assertEqual(parameters["unused_actions"], [3, 4, 5, 6])
        connectivity_rule = parameters["connectivity_rule"]
        natural_termination = parameters["natural_termination"]
        assert isinstance(connectivity_rule, str)
        assert isinstance(natural_termination, str)
        self.assertIn("largest generated navigable component", connectivity_rule)
        self.assertIn("moving forward onto the green goal", natural_termination)

    def test_profile_generation_classes_and_identity_are_distinct(self) -> None:
        default = WFCConfig(profile="MazeSimple", size=15)
        inconsistent = WFCConfig(profile="MazeKnot", size=15)
        slow = WFCConfig(profile="DungeonSpirals", size=15)

        self.assertEqual(default.generation_class, "default")
        self.assertEqual(inconsistent.generation_class, "inconsistent")
        self.assertEqual(slow.generation_class, "slow")
        self.assertNotEqual(
            WFCBenchmark(default).spec.environment_digest,
            WFCBenchmark(slow).spec.environment_digest,
        )

    def test_split_planning_and_scenario_rejection(self) -> None:
        benchmark = WFCBenchmark()
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

    def test_feedback_privacy_and_failure_penalty(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = WFCBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        content = feedback.content
        assert isinstance(content, dict)

        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"profile"', trace)
        self.assertNotIn(b"generation_seed", trace)
        self.assertEqual(content["policy_failures"], 1)
        self.assertIsNone(content["mean_known_cell_count"])

    def test_all_requested_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(WFC_PROFILES), 22)
        for profile in WFC_PROFILES:
            with self.subTest(profile=profile):
                benchmark = WFCBenchmark(WFCConfig(profile=profile, size=15))
                environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    step = environment.step(0)
                    metrics = _step_metrics(step)
                    self.assertIsInstance(step.reward, float)
                    self.assertGreater(_number_metric(metrics, "known_cell_count"), 0)
                    self.assertGreaterEqual(_number_metric(metrics, "known_frontier_count"), 0)
                finally:
                    environment.close()
                    environment.close()

    def test_replay_conformance_and_invalid_action(self) -> None:
        benchmark = WFCBenchmark(WFCConfig(size=15))
        report = check_benchmark(
            benchmark,
            fixtures=(BenchmarkFixture(EpisodeSpec(environment_seed=123), (0,)),),
        )
        self.assertTrue(report.passed, report.issues)
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()

    def test_real_navigation_trace_exposes_public_map_progress(self) -> None:
        benchmark = WFCBenchmark(WFCConfig(profile="MazeSimple", size=15))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        steps = _run_actions(benchmark, episode, _SUCCESS_ACTIONS)

        first_goal = steps[22]
        first_goal_metrics = _step_metrics(first_goal)
        self.assertEqual(first_goal_metrics["goal_visible"], True)
        self.assertEqual(first_goal_metrics["goal_found"], True)
        self.assertEqual(first_goal_metrics["goal_first_seen_step"], 23)
        self.assertEqual(first_goal_metrics["known_goal_distance"], 8)
        self.assertEqual(first_goal_metrics["task_stage"], "navigate_to_goal")
        self.assertGreater(_number_metric(first_goal_metrics, "known_cell_count"), 0)
        self.assertGreater(
            _number_metric(first_goal_metrics, "known_walkable_cell_count"), 0
        )
        self.assertGreaterEqual(
            _number_metric(first_goal_metrics, "known_frontier_count"), 0
        )
        self.assertGreater(_number_metric(first_goal_metrics, "known_map_fraction"), 0.0)
        self.assertLessEqual(_number_metric(first_goal_metrics, "known_map_fraction"), 1.0)

        final = steps[-1]
        final_metrics = _step_metrics(final)
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertAlmostEqual(final.reward, 1.0 - 0.9 * 35 / 900)
        self.assertEqual(final_metrics["goal_in_front_before_action"], True)
        self.assertEqual(final_metrics["known_goal_distance"], 0)
        self.assertEqual(final_metrics["success"], True)
        self.assertEqual(final_metrics["terminal_reason"], "success")

    def test_blocked_forward_and_unused_action_are_diagnostic(self) -> None:
        benchmark = WFCBenchmark(WFCConfig(profile="MazeSimple", size=15))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        environment = benchmark.make_environment(episode)
        try:
            environment.reset()
            blocked = environment.step(2)
            unused = environment.step(6)
        finally:
            environment.close()

        blocked_metrics = _step_metrics(blocked)
        unused_metrics = _step_metrics(unused)
        self.assertEqual(blocked_metrics["front_object_before_action"], "wall")
        self.assertEqual(blocked_metrics["blocked_forward"], True)
        self.assertEqual(blocked_metrics["move_succeeded"], False)
        self.assertEqual(unused_metrics["unused_action"], True)
        self.assertEqual(unused_metrics["unused_action_count"], 1)
        self.assertEqual(unused_metrics["done_count"], 1)
        self.assertEqual(unused_metrics["ineffective_action"], True)

    def test_timeout_is_distinct_from_navigation_failure(self) -> None:
        benchmark = WFCBenchmark(WFCConfig(profile="MazeSimple", size=15))
        episode = benchmark.episodes("validation", seed=5, count=1)[0]
        final = _run_actions(benchmark, episode, (6,) * 900)[-1]
        final_metrics = _step_metrics(final)

        self.assertFalse(final.terminated)
        self.assertTrue(final.truncated)
        self.assertEqual(final.reward, 0.0)
        self.assertEqual(final_metrics["step_count"], 900)
        self.assertEqual(final_metrics["remaining_steps"], 0)
        self.assertEqual(final_metrics["unused_action_count"], 900)
        self.assertEqual(final_metrics["terminal_reason"], "time_limit")
        self.assertEqual(final_metrics["task_stage"], "time_limit")

    def test_baseline_solves_representative_generation_families(self) -> None:
        profiles = (
            "MazeSimple",
            "DungeonMazeScaled",
            "RoomsFabric",
            "ObstaclesBlackdots",
            "ObstaclesAngular",
            "ObstaclesHogs3",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = WFCBenchmark(WFCConfig(profile=profile, size=15))
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
                content = result.feedback.content
                assert isinstance(content, dict)
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertEqual(content["goal_found_rate"], 1.0)
                documents = tuple(
                    json.loads(line)
                    for line in result.feedback.artifacts[0].read_bytes().splitlines()
                )
                transitions = tuple(item for item in documents if item["type"] == "transition")
                self.assertTrue(transitions)
                self.assertTrue(
                    all("known_map_fraction" in item["metrics"] for item in transitions)
                )
                self.assertTrue(
                    all("visible_objects" in item["next_observation"] for item in transitions)
                )


def _run_actions(
    benchmark: WFCBenchmark,
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
        "mission": "traverse the maze to get to the goal",
    }


if __name__ == "__main__":
    unittest.main()
