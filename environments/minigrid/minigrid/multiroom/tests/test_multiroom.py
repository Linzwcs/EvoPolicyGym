from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_multiroom import (
    MultiRoomBenchmark,
    MultiRoomConfig,
    baseline_program,
)


class MultiRoomBenchmarkTests(unittest.TestCase):
    def test_config_profiles_define_distinct_environment_identity(self) -> None:
        default = MultiRoomBenchmark()
        small = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
        fixed = MultiRoomBenchmark(MultiRoomConfig(profile="N4-S5"))
        legacy = MultiRoomBenchmark(
            MultiRoomConfig(profile="N4-S5-v0-legacy-N6")
        )

        self.assertEqual(
            default.spec.id,
            "minigrid/MultiRoom-v0/success-rate-v1",
        )
        self.assertEqual(default.spec.max_episode_steps, 120)
        self.assertEqual(small.spec.max_episode_steps, 40)
        self.assertEqual(fixed.spec.max_episode_steps, 80)
        self.assertEqual(legacy.spec.max_episode_steps, 120)
        self.assertNotEqual(
            default.spec.environment_digest,
            small.spec.environment_digest,
        )
        self.assertNotEqual(
            fixed.spec.environment_digest,
            legacy.spec.environment_digest,
        )
        self.assertEqual(
            default.spec.environment_parameters["profile"],
            "N6-S10",
        )
        self.assertEqual(
            legacy.spec.environment_parameters["maximum_rooms"],
            6,
        )

    def test_config_rejects_unsupported_or_ambiguous_profiles(self) -> None:
        with self.assertRaises(ValueError):
            MultiRoomConfig(profile="N4-S5-v0")
        with self.assertRaises(ValueError):
            MultiRoomConfig(profile=6)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = MultiRoomBenchmark()

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

    def test_environment_is_conformant_and_rejects_invalid_actions(
        self,
    ) -> None:
        benchmark = MultiRoomBenchmark(MultiRoomConfig(profile="N2-S4"))
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

        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=123)
        )
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
            self.assertEqual(
                observation["mission"],
                "traverse the rooms to get to the goal",
            )
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
        benchmark = MultiRoomBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "N2-S4"},
                )
            )

    def test_feedback_penalizes_failure_and_keeps_identity_private(self) -> None:
        benchmark = MultiRoomBenchmark()
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
        self.assertEqual(feedback.content["goal_found_episodes"], 0)
        self.assertEqual(feedback.content["opened_doors"], 0)

    def test_baseline_solves_every_public_profile_and_publishes_trace(
        self,
    ) -> None:
        profiles = (
            "N2-S4",
            "N4-S5",
            "N4-S5-v0-legacy-N6",
            "N6-S10",
        )
        for profile in profiles:
            with self.subTest(profile=profile):
                benchmark = MultiRoomBenchmark(
                    MultiRoomConfig(profile=profile)
                )
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

                self.assertEqual(
                    result.benchmark_id,
                    "minigrid/MultiRoom-v0/success-rate-v1",
                )
                self.assertEqual(
                    result.environment_digest,
                    benchmark.spec.environment_digest,
                )
                self.assertEqual(result.feedback.score, 1.0)
                self.assertIsInstance(result.feedback.content, dict)
                assert isinstance(result.feedback.content, dict)
                self.assertEqual(
                    result.feedback.content["goal_found_rate"],
                    1.0,
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
                self.assertEqual(trace.name, "trace.jsonl")
                self.assertEqual(
                    trace.media_type,
                    "application/x-ndjson",
                )
                self.assertTrue(transitions)
                self.assertTrue(
                    all(
                        "visible_objects" in item["next_observation"]
                        for item in transitions
                    )
                )


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "traverse the rooms to get to the goal",
    }


if __name__ == "__main__":
    unittest.main()
