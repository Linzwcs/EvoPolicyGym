from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_go_to_object import GoToObjectBenchmark, GoToObjectConfig, baseline_program


class GoToObjectBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = GoToObjectBenchmark()
        small = GoToObjectBenchmark(GoToObjectConfig(profile="6x6-N2"))

        self.assertEqual(
            default.spec.id,
            "minigrid/GoToObject-v0/mean-return-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 320)
        self.assertEqual(small.spec.max_episode_steps, 180)
        self.assertNotEqual(
            default.spec.environment_digest,
            small.spec.environment_digest,
        )
        self.assertEqual(
            default.spec.environment_parameters["profile"],
            "8x8-N2",
        )
        self.assertEqual(
            default.spec.environment_parameters["object_count"],
            2,
        )
        self.assertEqual(
            default.spec.environment_parameters["image_axis_order"],
            ["view_x", "view_y", "channel"],
        )
        self.assertEqual(
            default.spec.environment_parameters["direction_encoding"],
            {"east": 0, "south": 1, "west": 2, "north": 3},
        )
        self.assertEqual(
            default.spec.environment_parameters["success_reward_formula"],
            "1 - 0.9*step_count/max_episode_steps",
        )

        exposed = default.spec.environment_parameters["color_encoding"]
        self.assertIsInstance(exposed, dict)
        assert isinstance(exposed, dict)
        exposed["purple"] = 100
        fresh = default.spec.environment_parameters["color_encoding"]
        self.assertIsInstance(fresh, dict)
        assert isinstance(fresh, dict)
        self.assertEqual(fresh["purple"], 3)

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            GoToObjectConfig(profile="8x8-N3")
        with self.assertRaises(ValueError):
            GoToObjectConfig(profile=8)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = GoToObjectBenchmark()

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

    def test_environment_is_conformant_and_rejects_invalid_actions(
        self,
    ) -> None:
        benchmark = GoToObjectBenchmark(GoToObjectConfig(profile="6x6-N2"))
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 2, 6),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(
                set(observation),
                {"image", "direction", "mission"},
            )
            image = observation["image"]
            self.assertIsInstance(image, TensorValue)
            assert isinstance(image, TensorValue)
            self.assertEqual(image.dtype, "uint8")
            self.assertEqual(image.shape, (7, 7, 3))
            self.assertEqual(len(image.data), 147)
            mission = observation["mission"]
            self.assertIsInstance(mission, str)
            assert isinstance(mission, str)
            self.assertTrue(mission.endswith((" key", " ball", " box")))
        finally:
            environment.close()
            environment.close()

        invalid_actions: tuple[PolicyValue, ...] = (
            -1,
            7,
            True,
            2.0,
            [2],
        )
        for invalid in invalid_actions:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
            try:
                environment.reset()
                with self.assertRaises(InvalidAction):
                    environment.step(invalid)
            finally:
                environment.close()

    def test_step_feedback_exposes_target_and_candidate_search(self) -> None:
        benchmark = GoToObjectBenchmark(GoToObjectConfig(profile="6x6-N2"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=0))
        try:
            environment.reset()
            step = environment.step(0)
        finally:
            environment.close()
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["step_count"], 1)
        self.assertEqual(step.metrics["remaining_steps"], 179)
        self.assertEqual(step.metrics["turn_left_count"], 1)
        self.assertEqual(step.metrics["move_forward_count"], 0)
        self.assertEqual(step.metrics["target_label"], "blue_key")
        self.assertIsInstance(step.metrics["target_visible"], bool)
        self.assertIsInstance(step.metrics["visible_candidate_count"], int)
        self.assertIsInstance(
            step.metrics["unique_candidate_count_found"],
            int,
        )
        self.assertIn(
            step.metrics["task_stage"],
            {"explore_candidates", "approach_target", "declare_done"},
        )
        self.assertEqual(step.metrics["terminal_reason"], "none")

    def test_episode_scenario_cannot_override_benchmark_configuration(
        self,
    ) -> None:
        benchmark = GoToObjectBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "6x6-N2"},
                )
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = GoToObjectBenchmark()
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = benchmark.feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
        self.assertEqual(len(feedback.artifacts), 1)
        trace = feedback.artifacts[0]
        self.assertEqual(trace.name, "trace.jsonl")
        self.assertNotIn(b"environment_seed", trace.read_bytes())
        self.assertNotIn(b"policy_seed", trace.read_bytes())
        self.assertNotIn(b'"profile"', trace.read_bytes())
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["policy_failures"], 1)
        self.assertEqual(feedback.content["target_found_episodes"], 0)
        self.assertEqual(feedback.content["wrong_completion_episodes"], 0)

    def test_real_completion_errors_are_distinguished(self) -> None:
        benchmark = GoToObjectBenchmark(GoToObjectConfig(profile="6x6-N2"))
        toggle_record = _run_episode(benchmark, (5,))
        done_record = _run_episode(benchmark, (6,))
        toggle_metrics = toggle_record.transitions[-1].step.metrics
        done_metrics = done_record.transitions[-1].step.metrics
        self.assertIsInstance(toggle_metrics, dict)
        self.assertIsInstance(done_metrics, dict)
        assert isinstance(toggle_metrics, dict)
        assert isinstance(done_metrics, dict)
        self.assertEqual(toggle_metrics["premature_toggle"], True)
        self.assertEqual(toggle_metrics["wrong_done"], False)
        self.assertEqual(
            toggle_metrics["terminal_reason"],
            "premature_toggle",
        )
        self.assertEqual(done_metrics["premature_toggle"], False)
        self.assertEqual(done_metrics["wrong_done"], True)
        self.assertEqual(done_metrics["terminal_reason"], "wrong_done")

        feedback = benchmark.feedback((toggle_record, done_record))
        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["wrong_completion_rate"], 1.0)
        self.assertEqual(feedback.content["premature_toggle_rate"], 0.5)
        self.assertEqual(feedback.content["wrong_done_rate"], 0.5)
        documents = tuple(
            json.loads(line) for line in feedback.artifacts[0].read_bytes().splitlines()
        )
        outcomes = tuple(
            document["outcome"] for document in documents if document["type"] == "episode"
        )
        self.assertEqual(outcomes, ("premature_toggle", "wrong_done"))

    def test_time_limit_is_not_a_completion_error(self) -> None:
        benchmark = GoToObjectBenchmark(GoToObjectConfig(profile="6x6-N2"))
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            environment.reset()
            step = environment.step(0)
            for _ in range(benchmark.spec.max_episode_steps - 1):
                step = environment.step(0)
        finally:
            environment.close()
        self.assertFalse(step.terminated)
        self.assertTrue(step.truncated)
        self.assertIsInstance(step.metrics, dict)
        assert isinstance(step.metrics, dict)
        self.assertEqual(step.metrics["wrong_completion"], False)
        self.assertEqual(step.metrics["completion_step"], -1)
        self.assertEqual(step.metrics["remaining_steps"], 0)
        self.assertEqual(step.metrics["terminal_reason"], "time_limit")

    def test_baseline_solves_every_public_profile_and_publishes_trace(
        self,
    ) -> None:
        for profile in ("6x6-N2", "8x8-N2"):
            with self.subTest(profile=profile):
                benchmark = GoToObjectBenchmark(GoToObjectConfig(profile=profile))
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

                self.assertEqual(
                    result.benchmark_id,
                    "minigrid/GoToObject-v0/mean-return-v1",
                )
                self.assertEqual(
                    result.environment_digest,
                    benchmark.spec.environment_digest,
                )
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(result.feedback.content["success_rate"], 1.0)
                self.assertEqual(
                    result.feedback.score, result.feedback.content["mean_return"]
                )
                self.assertEqual(
                    result.feedback.content["target_found_rate"],
                    1.0,
                )
                self.assertEqual(
                    result.feedback.content["wrong_completion_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["premature_toggle_rate"],
                    0.0,
                )
                self.assertEqual(
                    result.feedback.content["wrong_done_rate"],
                    0.0,
                )
                trace = result.feedback.artifacts[0]
                documents = tuple(json.loads(line) for line in trace.read_bytes().splitlines())
                transitions = tuple(
                    document for document in documents if document["type"] == "transition"
                )
                self.assertEqual(trace.name, "trace.jsonl")
                self.assertEqual(
                    trace.media_type,
                    "application/x-ndjson",
                )
                self.assertTrue(transitions)
                self.assertTrue(
                    all(
                        "target_color" in item["next_observation"]
                        and "target_type" in item["next_observation"]
                        for item in transitions
                    )
                )
                episode_documents = tuple(
                    document for document in documents if document["type"] == "episode"
                )
                self.assertTrue(
                    all(document["outcome"] == "success" for document in episode_documents)
                )


def _run_episode(
    benchmark: GoToObjectBenchmark,
    actions: tuple[int, ...],
) -> EpisodeRecord:
    episode = EpisodeSpec(environment_seed=0)
    environment = benchmark.make_environment(episode)
    transitions: list[Transition] = []
    try:
        initial_observation = environment.reset()
        for action in actions:
            step = environment.step(action)
            transitions.append(Transition(action=action, step=step))
    finally:
        environment.close()
    return EpisodeRecord(
        episode=episode,
        policy_seed=0,
        initial_observation=initial_observation,
        transitions=tuple(transitions),
    )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "go to the purple ball",
    }


if __name__ == "__main__":
    unittest.main()
