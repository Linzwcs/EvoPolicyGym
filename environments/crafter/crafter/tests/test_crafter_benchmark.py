from __future__ import annotations

import gzip
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import crafter.constants
import crafter.engine
import crafter.objects
import imageio_ffmpeg
import numpy
from evopolicygym import EvaluationConfig, EvaluationResult, Program, evaluate
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
from evopolicygym.policy import PolicyValue, TensorValue
from evopolicygym.skills import AgentSkill

from crafter_benchmarks import (
    ACHIEVEMENTS,
    ACTIONS,
    CrafterBenchmark,
    CrafterConfig,
    CrafterLongHorizonSurvivalBenchmark,
    baseline_program,
    local_symbolic_baseline_program,
)
from crafter_benchmarks.constants import (
    SYMBOLIC_INVENTORY_KEYS,
    SYMBOLIC_PLAYER_CENTER,
)
from crafter_benchmarks.lhs_scoring import (
    LHS_ALIVE_ALPHA,
    LHS_FIRST_UNLOCK_CREDITS,
    LHS_PRODUCTIVITY_REPEAT_FRACTION,
    LHS_PRODUCTIVITY_REPEAT_QUOTAS,
    LHS_VITAL_ALPHA,
    LHSScoringState,
    lhs_feedback_score,
)
from crafter_benchmarks.programs.baseline.policy import (
    ActionProposal,
    BaselinePolicy,
    CombatDefenseModule,
    Coordinator,
    ExplorationModule,
    ProductionModule,
    SurvivalModule,
    VisualTranslationModule,
    WorldMemoryModule,
)
from crafter_benchmarks.programs.local_symbolic_baseline.policy import (
    LocalSymbolicBaselinePolicy,
)
from crafter_benchmarks.symbolic import local_symbolic_observation

_ZERO_OBSERVATION = TensorValue(
    dtype="uint8",
    shape=(64, 64, 3),
    data=bytes(64 * 64 * 3),
)
_ZERO_SYMBOLIC_OBSERVATION: dict[str, PolicyValue] = {
    "terrain": TensorValue(dtype="uint8", shape=(7, 9), data=bytes(63)),
    "entities": TensorValue(
        dtype="uint8",
        shape=(7, 9),
        data=bytes(31) + b"\x01" + bytes(31),
    ),
    "inventory": {
        name: 9 if name in {"health", "food", "drink", "energy"} else 0
        for name in SYMBOLIC_INVENTORY_KEYS
    },
    "facing": "down",
    "sleeping": False,
    "daylight": 1.0,
}


class _CountingCrafter:
    action_names = ACTIONS

    def __init__(
        self,
        *,
        done: bool = False,
        discount: float = 1.0,
        achievement_name: str | None = None,
        energy: int = 9,
        reward: float = 0.0,
    ) -> None:
        self.steps = 0
        self.done = done
        self.discount = discount
        self.achievement_name = achievement_name
        self.energy = energy
        self.reward = reward
        self.achievements = {name: 0 for name in ACHIEVEMENTS}

    def reset(self) -> numpy.ndarray:
        self.achievements = {name: 0 for name in ACHIEVEMENTS}
        return numpy.zeros((64, 64, 3), dtype=numpy.uint8)

    def step(self, action: int) -> tuple[object, float, bool, dict[str, object]]:
        del action
        self.steps += 1
        if self.achievement_name is not None:
            self.achievements[self.achievement_name] += 1
        return (
            numpy.zeros((64, 64, 3), dtype=numpy.uint8),
            self.reward,
            self.done,
            {
                "discount": self.discount,
                "achievements": dict(self.achievements),
                "inventory": {
                    "health": 9,
                    "food": 9,
                    "drink": 9,
                    "energy": self.energy,
                },
            },
        )


class CrafterBenchmarkTests(unittest.TestCase):
    def test_local_symbolic_configuration_and_spec_are_separate(self) -> None:
        rgb = CrafterBenchmark()
        symbolic_config = CrafterConfig(
            observation_profile="local-symbolic-v1"
        )
        symbolic = CrafterBenchmark(symbolic_config)

        self.assertEqual(
            rgb.spec.environment_digest,
            "sha256:9777c328423dee6989889d83b67976b2946ce0b68e2cef7f692ff9ee90dfee55",
        )
        self.assertNotIn("observation_profile", rgb.spec.environment_parameters)
        self.assertEqual(
            symbolic.spec.id,
            "crafter/CrafterReward-v1/local-symbolic-v1/achievement-score-v1",
        )
        self.assertNotEqual(
            rgb.spec.environment_digest,
            symbolic.spec.environment_digest,
        )
        observation_space = symbolic.spec.observation_space
        assert isinstance(observation_space, dict)
        self.assertEqual(observation_space["type"], "mapping")
        fields = observation_space["fields"]
        assert isinstance(fields, dict)
        self.assertEqual(set(fields), {
            "terrain", "entities", "inventory", "facing", "sleeping", "daylight"
        })
        parameters = dict(symbolic.spec.environment_parameters)
        self.assertEqual(parameters["symbolic_player_row"], 3)
        self.assertEqual(parameters["symbolic_player_column"], 4)
        with self.assertRaises(TypeError):
            CrafterConfig(observation_profile=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CrafterConfig(observation_profile="global")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "MP4|mp4"):
            CrafterConfig(
                observation_profile="local-symbolic-v1",
                include_mp4_feedback=True,
            )

    def test_every_scoring_profile_supports_local_symbolic(self) -> None:
        config = CrafterConfig(observation_profile="local-symbolic-v1")
        benchmarks = (
            CrafterBenchmark(config),
            CrafterLongHorizonSurvivalBenchmark(config),
        )
        for benchmark in benchmarks:
            with self.subTest(benchmark=benchmark.spec.primary_metric):
                self.assertIn("/local-symbolic-v1/", benchmark.spec.id)
                self.assertEqual(
                    benchmark.spec.environment_parameters["observation_profile"],
                    "local-symbolic-v1",
                )

    def test_real_local_symbolic_observation_and_rgb_dynamics_match(self) -> None:
        episode = EpisodeSpec(environment_seed=0xC0FFEE)
        rgb_benchmark = CrafterBenchmark(CrafterConfig(max_episode_steps=32))
        symbolic_benchmark = CrafterBenchmark(
            CrafterConfig(max_episode_steps=32, observation_profile="local-symbolic-v1")
        )
        rgb_environment = rgb_benchmark.make_environment(episode)
        symbolic_environment = symbolic_benchmark.make_environment(episode)
        rgb_transitions: list[Transition] = []
        symbolic_transitions: list[Transition] = []
        try:
            rgb_initial = rgb_environment.reset()
            symbolic_initial = symbolic_environment.reset()
            self.assertIsInstance(rgb_initial, TensorValue)
            self.assertIsInstance(symbolic_initial, dict)
            assert isinstance(symbolic_initial, dict)
            self.assertEqual(
                set(symbolic_initial),
                {"terrain", "entities", "inventory", "facing", "sleeping", "daylight"},
            )
            entities = symbolic_initial["entities"]
            assert isinstance(entities, TensorValue)
            entity_array = numpy.frombuffer(entities.data, dtype=numpy.uint8).reshape(7, 9)
            self.assertEqual(entity_array[SYMBOLIC_PLAYER_CENTER], 1)
            inventory = symbolic_initial["inventory"]
            assert isinstance(inventory, dict)
            self.assertEqual(tuple(inventory), SYMBOLIC_INVENTORY_KEYS)

            for action in (0, 1, 5, 3, 5, 4, 0):
                rgb_step = rgb_environment.step(action)
                symbolic_step = symbolic_environment.step(action)
                self.assertEqual(rgb_step.reward, symbolic_step.reward)
                self.assertEqual(rgb_step.metrics, symbolic_step.metrics)
                self.assertEqual(rgb_step.terminated, symbolic_step.terminated)
                self.assertEqual(rgb_step.truncated, symbolic_step.truncated)
                rgb_transitions.append(Transition(action=action, step=rgb_step))
                symbolic_transitions.append(
                    Transition(action=action, step=symbolic_step)
                )
        finally:
            rgb_environment.close()
            symbolic_environment.close()
        rgb_record = EpisodeRecord(
            episode=episode,
            policy_seed=1,
            initial_observation=rgb_initial,
            transitions=tuple(rgb_transitions),
        )
        symbolic_record = EpisodeRecord(
            episode=episode,
            policy_seed=1,
            initial_observation=symbolic_initial,
            transitions=tuple(symbolic_transitions),
        )
        self.assertEqual(
            rgb_benchmark.feedback((rgb_record,)).score,
            symbolic_benchmark.feedback((symbolic_record,)).score,
        )

    def test_local_symbolic_padding_and_entity_variants(self) -> None:
        world = crafter.engine.World(
            (64, 64), crafter.constants.materials, (12, 12)
        )
        world._mat_map.fill(world._mat_ids["grass"])
        player = crafter.objects.Player(world, (0, 0))
        world.add(player)
        arrow = crafter.objects.Arrow(world, (1, 0), (1, 0))
        world.add(arrow)
        plant = crafter.objects.Plant(world, (0, 1))
        plant.grown = 301
        world.add(plant)
        world.daylight = 0.25
        environment = SimpleNamespace(_world=world, _player=player)

        observation = local_symbolic_observation(environment)
        terrain = observation["terrain"]
        entities = observation["entities"]
        assert isinstance(terrain, TensorValue)
        assert isinstance(entities, TensorValue)
        terrain_array = numpy.frombuffer(terrain.data, dtype=numpy.uint8).reshape(7, 9)
        entity_array = numpy.frombuffer(entities.data, dtype=numpy.uint8).reshape(7, 9)
        self.assertTrue(numpy.all(terrain_array[:3, :] == 0))
        self.assertTrue(numpy.all(terrain_array[:, :4] == 0))
        self.assertEqual(terrain_array[3, 4], 2)
        self.assertEqual(entity_array[3, 4], 1)
        self.assertEqual(entity_array[3, 5], 6)
        self.assertEqual(entity_array[4, 4], 10)
        self.assertEqual(observation["daylight"], 0.25)

    def test_local_symbolic_feedback_round_trips_policy_observations(self) -> None:
        benchmark = CrafterBenchmark(
            CrafterConfig(
                max_episode_steps=8,
                observation_profile="local-symbolic-v1",
            )
        )
        episode = benchmark.episodes("train", seed=9, count=1)[0]
        environment = benchmark.make_environment(episode)
        try:
            initial = environment.reset()
            step = environment.step(0)
        finally:
            environment.close()
        record = EpisodeRecord(
            episode=episode,
            policy_seed=10,
            initial_observation=initial,
            transitions=(Transition(action=0, step=step),),
        )

        feedback = benchmark.feedback((record,))
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        self.assertEqual(
            manifest["schema"],
            "crafter/local-symbolic-feedback-manifest/v1",
        )
        self.assertEqual(manifest["observation_profile"], "local-symbolic-v1")
        self.assertNotIn("visual_evidence", manifest)
        observation_artifact = next(
            artifact for artifact in feedback.artifacts if artifact.name.endswith(".npz")
        )
        with numpy.load(
            io.BytesIO(observation_artifact.read_bytes()), allow_pickle=False
        ) as archive:
            self.assertEqual(
                set(archive.files),
                {
                    "terrain", "entities", "inventory", "facing", "sleeping",
                    "daylight", "observation_indices",
                },
            )
            self.assertEqual(archive["terrain"].shape, (2, 7, 9))
            self.assertEqual(archive["entities"].shape, (2, 7, 9))
            self.assertEqual(archive["inventory"].shape, (2, 16))
            numpy.testing.assert_array_equal(
                archive["observation_indices"], numpy.asarray([0, 1], dtype=numpy.uint32)
            )
        trajectory = tuple(
            json.loads(line)
            for line in gzip.decompress(feedback.artifacts[0].read_bytes()).splitlines()
        )
        self.assertEqual(trajectory[0]["observation_profile"], "local-symbolic-v1")
        public = b"".join(artifact.read_bytes() for artifact in feedback.artifacts)
        self.assertNotIn(b"environment_seed", public)
        self.assertNotIn(b"player_pos", public)
        self.assertNotIn(b"semantic", public)

    def test_lossless_observation_feedback_contract_is_public(self) -> None:
        disabled = CrafterBenchmark()
        enabled = CrafterBenchmark(
            CrafterConfig(include_mp4_feedback=True)
        )
        observations = disabled.spec.metadata["public_observations"]
        assert isinstance(observations, dict)
        self.assertEqual(observations["format"], "compressed NumPy NPZ")
        self.assertEqual(observations["dtype"], "uint8")
        self.assertEqual(observations["shape"], [64, 64, 3])
        self.assertEqual(observations["observations_per_artifact"], 1_024)
        self.assertEqual(observations["complete_artifact_episode_limit"], 64)
        self.assertEqual(observations["frame_sampling"], "none")
        self.assertIs(observations["pixel_exact"], True)
        self.assertEqual(observations["detailed_artifact_splits"], ["train"])
        optional_mp4 = observations["optional_mp4"]
        assert isinstance(optional_mp4, dict)
        self.assertIs(optional_mp4["enabled"], False)
        enabled_observations = enabled.spec.metadata["public_observations"]
        assert isinstance(enabled_observations, dict)
        enabled_mp4 = enabled_observations["optional_mp4"]
        assert isinstance(enabled_mp4, dict)
        self.assertIs(
            enabled_mp4["enabled"],
            True,
        )
        self.assertIs(
            disabled.spec.environment_parameters["include_mp4_feedback"],
            False,
        )
        self.assertIs(
            enabled.spec.environment_parameters["include_mp4_feedback"],
            True,
        )
        self.assertNotEqual(
            disabled.spec.environment_digest,
            enabled.spec.environment_digest,
        )
        with self.assertRaises(TypeError):
            CrafterConfig(include_mp4_feedback=1)  # type: ignore[arg-type]

    def test_episode_planning_is_reproducible_and_split_scoped(self) -> None:
        benchmark = CrafterBenchmark()
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
        self.assertTrue(
            all(
                item.scenario == {"publish_detailed_artifacts": True}
                for item in train
            )
        )
        self.assertTrue(
            all(
                item.scenario == {"publish_detailed_artifacts": False}
                for item in validation
            )
        )

    def test_validation_and_test_feedback_do_not_generate_artifacts(self) -> None:
        benchmark = CrafterBenchmark(
            CrafterConfig(include_mp4_feedback=True)
        )
        train_episode = benchmark.episodes("train", seed=7, count=1)[0]
        validation_episode = benchmark.episodes(
            "validation",
            seed=7,
            count=1,
        )[0]
        test_episode = benchmark.episodes("test", seed=7, count=1)[0]

        train = benchmark.feedback(
            (_record(("collect_wood",), reward=1.0, episode=train_episode),)
        )
        self.assertTrue(train.artifacts)
        self.assertTrue(
            any(artifact.name.endswith(".mp4") for artifact in train.artifacts)
        )
        for episode in (validation_episode, test_episode):
            with self.subTest(episode=episode):
                feedback = benchmark.feedback(
                    (_record(("collect_wood",), reward=1.0, episode=episode),)
                )
                self.assertEqual(feedback.artifacts, ())
                assert isinstance(feedback.content, dict)
                detailed = feedback.content["detailed_feedback"]
                assert isinstance(detailed, dict)
                self.assertIs(detailed["complete"], False)
                self.assertEqual(
                    detailed["reason"],
                    "split_disables_detailed_artifacts",
                )

        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"unexpected": True},
                )
            )

    def test_environment_replays_deterministically(self) -> None:
        fixtures = (
            BenchmarkFixture(
                episode=EpisodeSpec(environment_seed=123),
                actions=(0, 1, 3, 5, 2, 4),
            ),
        )
        benchmarks = (
            CrafterBenchmark(CrafterConfig(max_episode_steps=32)),
            CrafterBenchmark(
                CrafterConfig(
                    max_episode_steps=32,
                    observation_profile="local-symbolic-v1",
                )
            ),
            CrafterLongHorizonSurvivalBenchmark(
                CrafterConfig(max_episode_steps=32)
            ),
        )
        for benchmark in benchmarks:
            with self.subTest(benchmark=benchmark.spec.id):
                report = check_benchmark(benchmark, fixtures=fixtures)
                self.assertTrue(report.passed, report.issues)

    def test_invalid_actions_do_not_advance_upstream(self) -> None:
        fake = _CountingCrafter()
        with patch(
            "crafter_benchmarks.environment.crafter.Env",
            return_value=fake,
        ):
            environment = CrafterBenchmark(
                CrafterConfig(max_episode_steps=8)
            ).make_environment(EpisodeSpec(environment_seed=1))
            try:
                observation = environment.reset()
                self.assertIsInstance(observation, TensorValue)
                invalid: tuple[PolicyValue, ...] = (
                    True,
                    -1,
                    17,
                    1.0,
                    None,
                    [],
                    {},
                )
                for action in invalid:
                    with self.assertRaises(InvalidAction):
                        environment.step(action)
                self.assertEqual(fake.steps, 0)
                environment.step(0)
                self.assertEqual(fake.steps, 1)
            finally:
                environment.close()
                environment.close()

    def test_policy_failure_stops_before_upstream_step(self) -> None:
        fake = _CountingCrafter()
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "policy.py").write_text(
                "class Policy:\n"
                "    def act(self, observation):\n"
                "        del observation\n"
                "        return None\n"
                "\n"
                "def make_policy(context):\n"
                "    del context\n"
                "    return Policy()\n",
                encoding="utf-8",
            )
            program = Program.from_directory(temporary)
            with patch(
                "crafter_benchmarks.environment.crafter.Env",
                return_value=fake,
            ):
                result = evaluate(
                    program,
                    CrafterBenchmark(CrafterConfig(max_episode_steps=8)),
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        episodes=1,
                        seed=3,
                        episode_timeout_seconds=30,
                    ),
                )

        self.assertEqual(result.episodes[0].failure, "invalid_action")
        self.assertEqual(fake.steps, 0)

    def test_horizon_is_translated_to_truncation(self) -> None:
        environment = CrafterBenchmark(
            CrafterConfig(max_episode_steps=1)
        ).make_environment(EpisodeSpec(environment_seed=5))
        try:
            environment.reset()
            step = environment.step(0)
            self.assertFalse(step.terminated)
            self.assertTrue(step.truncated)
        finally:
            environment.close()

    def test_death_is_translated_to_termination(self) -> None:
        fake = _CountingCrafter(done=True, discount=0.0)
        with patch(
            "crafter_benchmarks.environment.crafter.Env",
            return_value=fake,
        ):
            environment = CrafterBenchmark(
                CrafterConfig(max_episode_steps=8)
            ).make_environment(EpisodeSpec(environment_seed=1))
            try:
                environment.reset()
                step = environment.step(0)
                self.assertTrue(step.terminated)
                self.assertFalse(step.truncated)
            finally:
                environment.close()

    def test_environment_reports_first_and_repeated_achievement_events(self) -> None:
        fake = _CountingCrafter(achievement_name="collect_drink")
        with patch(
            "crafter_benchmarks.environment.crafter.Env",
            return_value=fake,
        ):
            environment = CrafterBenchmark(
                CrafterConfig(max_episode_steps=8)
            ).make_environment(EpisodeSpec(environment_seed=1))
            try:
                environment.reset()
                first = environment.step(5)
                repeated = environment.step(5)
            finally:
                environment.close()

        self.assertEqual(
            first.metrics,
            {
                "achievements_unlocked": ["collect_drink"],
                "achievement_event_counts": {"collect_drink": 1},
                "maintenance_vitals": {"health": 9, "food": 9, "drink": 9},
            },
        )
        self.assertEqual(
            repeated.metrics,
            {
                "achievements_unlocked": [],
                "achievement_event_counts": {"collect_drink": 1},
                "maintenance_vitals": {"health": 9, "food": 9, "drink": 9},
            },
        )

    def test_feedback_uses_official_score_and_penalizes_failure(self) -> None:
        completed = _record(("collect_wood",), reward=1.0)
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_ZERO_OBSERVATION,
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = CrafterBenchmark().feedback((completed, failed))

        expected = math.expm1(math.log1p(50.0) / len(ACHIEVEMENTS))
        self.assertAlmostEqual(feedback.score, expected)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        rates = feedback.content["achievement_success_percent"]
        self.assertIsInstance(rates, dict)
        assert isinstance(rates, dict)
        self.assertEqual(rates["collect_wood"], 50.0)
        self.assertEqual(rates["collect_diamond"], 0.0)
        self.assertEqual(feedback.content["policy_failures"], 1)
        names = tuple(artifact.name for artifact in feedback.artifacts)
        self.assertEqual(
            names,
            (
                "trajectories/episode-000000/trajectory-000000.jsonl.gz",
                "observations/episode-000000/observations-000000.npz",
                "trajectories/episode-000001/trajectory-000000.jsonl.gz",
                "observations/episode-000001/observations-000000.npz",
                "artifact-manifest.json",
            ),
        )
        self.assertEqual(
            tuple(artifact.retention for artifact in feedback.artifacts),
            (
                "permanent",
                "bulk",
                "permanent",
                "bulk",
                "permanent",
            ),
        )
        trajectory = tuple(
            json.loads(line)
            for line in gzip.decompress(
                feedback.artifacts[0].read_bytes()
            ).splitlines()
        )
        self.assertEqual([item["type"] for item in trajectory], ["episode", "transition"])
        self.assertEqual(trajectory[1]["observation_index"], 0)
        self.assertEqual(trajectory[1]["next_observation_index"], 1)
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        self.assertEqual(
            manifest["schema"],
            "crafter/complete-feedback-manifest/v6",
        )
        self.assertIs(manifest["complete"], True)
        self.assertEqual(manifest["score_profile"], "upstream")
        self.assertEqual(manifest["episodes"], 2)
        self.assertEqual(manifest["transitions"], 1)
        self.assertEqual(manifest["observations"], 3)
        self.assertEqual(manifest["replay_artifacts"], [])
        self.assertEqual(
            manifest["visual_evidence"]["frame_sampling"], "none"
        )

        self.assertIs(manifest["visual_evidence"]["pixel_exact"], True)
        self.assertIs(
            manifest["visual_evidence"]["mp4_replays"]["enabled"],
            False,
        )
        self.assertEqual(len(manifest["observation_artifacts"]), 2)
        self.assertEqual(
            manifest["observation_artifacts"][0]["observations"],
            2,
        )
        with numpy.load(
            io.BytesIO(feedback.artifacts[1].read_bytes()),
            allow_pickle=False,
        ) as archive:
            self.assertEqual(archive["observations"].shape, (2, 64, 64, 3))
            numpy.testing.assert_array_equal(
                archive["observation_indices"],
                numpy.asarray([0, 1], dtype=numpy.uint32),
            )

        public = b"".join(
            artifact.read_bytes() for artifact in feedback.artifacts
        )
        self.assertNotIn(b"environment_seed", public)
        self.assertNotIn(b"policy_seed", public)
        self.assertNotIn(b"player_pos", public)
        self.assertNotIn(b"semantic", public)

    def test_feedback_artifacts_remain_bounded_for_large_batches(self) -> None:
        record = _record(("collect_wood",), reward=1.0)
        feedback = CrafterBenchmark().feedback((record,) * 1_000)

        self.assertEqual(feedback.artifacts, ())
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        detailed = feedback.content["detailed_feedback"]
        self.assertIsInstance(detailed, dict)
        assert isinstance(detailed, dict)
        self.assertIs(detailed["complete"], False)
        self.assertEqual(detailed["detail_scope"], "aggregate-only")
        self.assertEqual(detailed["detailed_artifact_episode_limit"], 64)
        self.assertEqual(detailed["episodes"], 1_000)
        self.assertEqual(detailed["transitions"], 1_000)
        self.assertEqual(detailed["observations"], 2_000)

    def test_feedback_preserves_every_policy_observation_losslessly(self) -> None:
        initial = TensorValue(
            dtype="uint8",
            shape=(64, 64, 3),
            data=bytes(range(256)) * 48,
        )
        following = TensorValue(
            dtype="uint8",
            shape=(64, 64, 3),
            data=bytes(reversed(range(256))) * 48,
        )
        record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=10),
            policy_seed=20,
            initial_observation=initial,
            transitions=(
                Transition(
                    action=5,
                    step=Step(
                        observation=following,
                        reward=0.0,
                        terminated=True,
                        metrics={"achievements_unlocked": []},
                    ),
                ),
            ),
        )

        feedback = CrafterBenchmark().feedback((record,))
        self.assertFalse(
            any(artifact.name.endswith(".mp4") for artifact in feedback.artifacts)
        )
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        chunks = manifest["observation_artifacts"]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["observations"], 2)
        self.assertEqual(chunks[0]["first_observation_index"], 0)
        self.assertEqual(chunks[0]["last_observation_index"], 1)
        self.assertEqual(manifest["observations"], 2)
        observation_artifact = next(
            artifact
            for artifact in feedback.artifacts
            if artifact.name.endswith(".npz")
        )
        with numpy.load(
            io.BytesIO(observation_artifact.read_bytes()),
            allow_pickle=False,
        ) as archive:
            numpy.testing.assert_array_equal(
                archive["observations"][0],
                numpy.frombuffer(initial.data, dtype=numpy.uint8).reshape(
                    initial.shape
                ),
            )
            numpy.testing.assert_array_equal(
                archive["observations"][1],
                numpy.frombuffer(following.data, dtype=numpy.uint8).reshape(
                    following.shape
                ),
            )

    def test_mp4_switch_adds_one_complete_replay_and_keeps_npz(self) -> None:
        feedback = CrafterBenchmark(
            CrafterConfig(include_mp4_feedback=True)
        ).feedback((_record(("collect_wood",), reward=1.0),))

        names = tuple(artifact.name for artifact in feedback.artifacts)
        self.assertIn(
            "observations/episode-000000/observations-000000.npz",
            names,
        )
        self.assertIn("replays/episode-000000/replay.mp4", names)
        replay = next(
            artifact
            for artifact in feedback.artifacts
            if artifact.name.endswith(".mp4")
        )
        self.assertEqual(replay.media_type, "video/mp4")
        self.assertEqual(replay.retention, "bulk")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.mp4"
            path.write_bytes(replay.read_bytes())
            frames, duration = imageio_ffmpeg.count_frames_and_secs(path)
        self.assertEqual(frames, 2)
        self.assertAlmostEqual(duration, 0.2)

        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        self.assertEqual(len(manifest["observation_artifacts"]), 1)
        self.assertEqual(len(manifest["replay_artifacts"]), 1)
        entry = manifest["replay_artifacts"][0]
        self.assertEqual(entry["video_frames"], 2)
        self.assertEqual(entry["first_observation_index"], 0)
        self.assertEqual(entry["last_observation_index"], 1)
        self.assertIs(
            manifest["visual_evidence"]["mp4_replays"]["enabled"],
            True,
        )

    def test_feedback_keeps_complete_artifacts_for_64_episodes(self) -> None:
        record = _record(("collect_wood",), reward=1.0)

        feedback = CrafterBenchmark().feedback((record,) * 64)

        self.assertEqual(len(feedback.artifacts), 129)
        self.assertFalse(
            any(artifact.name.endswith(".mp4") for artifact in feedback.artifacts)
        )
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        self.assertIs(manifest["complete"], True)
        self.assertEqual(manifest["episodes"], 64)
        self.assertEqual(len(manifest["trajectory_artifacts"]), 64)
        self.assertEqual(len(manifest["observation_artifacts"]), 64)

        with_mp4 = CrafterBenchmark(
            CrafterConfig(include_mp4_feedback=True)
        ).feedback((record,) * 64)
        self.assertEqual(len(with_mp4.artifacts), 193)
        mp4_manifest = json.loads(with_mp4.artifacts[-1].read_bytes())
        self.assertEqual(len(mp4_manifest["trajectory_artifacts"]), 64)
        self.assertEqual(len(mp4_manifest["observation_artifacts"]), 64)
        self.assertEqual(len(mp4_manifest["replay_artifacts"]), 64)

    def test_observation_chunks_are_episode_local_and_kernel_bounded(self) -> None:
        transition = Transition(
            action=5,
            step=Step(
                observation=_ZERO_OBSERVATION,
                reward=0.0,
                terminated=False,
                metrics={"achievements_unlocked": []},
            ),
        )
        record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=10),
            policy_seed=20,
            initial_observation=_ZERO_OBSERVATION,
            transitions=(transition,) * 1_024,
        )

        feedback = CrafterBenchmark().feedback((record,))

        chunks = tuple(
            artifact
            for artifact in feedback.artifacts
            if artifact.name.endswith(".npz")
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            tuple(artifact.name for artifact in chunks),
            (
                "observations/episode-000000/observations-000000.npz",
                "observations/episode-000000/observations-000001.npz",
            ),
        )
        self.assertTrue(
            all(artifact.size <= 16 * 1024 * 1024 for artifact in chunks)
        )
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        entries = manifest["observation_artifacts"]
        self.assertEqual(entries[0]["first_observation_index"], 0)
        self.assertEqual(entries[0]["last_observation_index"], 1_023)
        self.assertEqual(entries[1]["first_observation_index"], 1_024)
        self.assertEqual(entries[1]["last_observation_index"], 1_024)

    def test_feedback_reports_unscored_action_diagnostics(self) -> None:
        actions = (1, 2, 1, 2, 1, 2, 1, 2, 5, 3, 4)
        transitions = tuple(
            Transition(
                action=action,
                step=Step(
                    observation=_ZERO_OBSERVATION,
                    reward=0.0,
                    terminated=index == len(actions) - 1,
                    metrics={"achievements_unlocked": []},
                ),
            )
            for index, action in enumerate(actions)
        )
        record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=10),
            policy_seed=20,
            initial_observation=_ZERO_OBSERVATION,
            transitions=transitions,
        )

        feedback = CrafterBenchmark().feedback((record,))

        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        diagnostics = feedback.content["action_diagnostics"]
        self.assertIsInstance(diagnostics, dict)
        assert isinstance(diagnostics, dict)
        self.assertIs(diagnostics["scored"], False)
        self.assertEqual(diagnostics["total_actions"], 11)
        self.assertEqual(diagnostics["movement_actions"], 10)
        movement_percent = diagnostics["movement_action_percent"]
        self.assertIsInstance(movement_percent, float)
        assert isinstance(movement_percent, float)
        self.assertAlmostEqual(movement_percent, 1000 / 11)
        self.assertEqual(diagnostics["adjacent_movement_pairs"], 8)
        self.assertEqual(diagnostics["immediate_reverse_movement_pairs"], 8)
        self.assertEqual(diagnostics["immediate_reverse_movement_percent"], 100.0)
        self.assertEqual(diagnostics["longest_immediate_reverse_action_run"], 8)
        self.assertEqual(
            diagnostics["episodes_with_immediate_reverse_run_at_least_8"],
            1,
        )
        self.assertEqual(
            diagnostics["longest_repeated_short_action_cycle_run"],
            8,
        )
        self.assertEqual(
            diagnostics["longest_repeated_short_action_cycle_period"],
            2,
        )
        self.assertEqual(
            diagnostics[
                "episodes_with_repeated_short_action_cycle_run_at_least_16"
            ],
            0,
        )
        self.assertEqual(diagnostics["longest_same_action_run"], 1)
        counts = diagnostics["action_counts"]
        self.assertIsInstance(counts, dict)
        assert isinstance(counts, dict)
        self.assertEqual(counts["move_left"], 4)
        self.assertEqual(counts["move_right"], 4)
        self.assertEqual(counts["do"], 1)

    def test_action_diagnostics_detect_period_four_cycles(self) -> None:
        actions = (1, 2, 4, 3) * 5
        transitions = tuple(
            Transition(
                action=action,
                step=Step(
                    observation=_ZERO_OBSERVATION,
                    reward=0.0,
                    terminated=index == len(actions) - 1,
                    metrics={"achievements_unlocked": []},
                ),
            )
            for index, action in enumerate(actions)
        )
        record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=10),
            policy_seed=20,
            initial_observation=_ZERO_OBSERVATION,
            transitions=transitions,
        )

        feedback = CrafterBenchmark().feedback((record,))

        assert isinstance(feedback.content, dict)
        diagnostics = feedback.content["action_diagnostics"]
        assert isinstance(diagnostics, dict)
        self.assertEqual(
            diagnostics["longest_repeated_short_action_cycle_run"],
            20,
        )
        self.assertEqual(
            diagnostics["longest_repeated_short_action_cycle_period"],
            4,
        )
        self.assertEqual(
            diagnostics[
                "episodes_with_repeated_short_action_cycle_run_at_least_16"
            ],
            1,
        )
        self.assertEqual(diagnostics["longest_same_action_run"], 1)

    def test_lhs_scoring_keeps_alive_signal_and_scales_development(
        self,
    ) -> None:
        state = LHSScoringState()
        first, _ = state.transition(
            terminated=False,
            unlocked=("collect_wood",),
            event_counts={"collect_wood": 1},
            vitals={"health": 9, "food": 9, "drink": 9},
        )
        repeated, _ = state.transition(
            terminated=False,
            unlocked=(),
            event_counts={"collect_wood": 1},
            vitals={"health": 9, "food": 9, "drink": 9},
        )
        depleted, _ = state.transition(
            terminated=False,
            unlocked=(),
            event_counts={},
            vitals={"health": 9, "food": 0, "drink": 9},
        )
        terminal, _ = state.transition(
            terminated=True,
            unlocked=(),
            event_counts={},
            vitals={"health": 0, "food": 0, "drink": 9},
        )

        self.assertAlmostEqual(first["alive_survival"], LHS_ALIVE_ALPHA)
        self.assertAlmostEqual(first["vital_survival"], LHS_VITAL_ALPHA)
        self.assertAlmostEqual(
            first["first_unlock"],
            LHS_FIRST_UNLOCK_CREDITS["collect_wood"],
        )
        self.assertAlmostEqual(
            repeated["productivity_repeat"],
            LHS_PRODUCTIVITY_REPEAT_FRACTION
            * LHS_FIRST_UNLOCK_CREDITS["collect_wood"]
            / LHS_PRODUCTIVITY_REPEAT_QUOTAS["collect_wood"],
        )
        self.assertEqual(depleted["alive_survival"], LHS_ALIVE_ALPHA)
        self.assertEqual(depleted["vital_survival"], 0.0)
        self.assertEqual(terminal["alive_survival"], 0.0)
        self.assertEqual(terminal["vital_survival"], 0.0)

    def test_lhs_feedback_selects_lower_tail_by_survival_only(self) -> None:
        mean, lower, secondary, count, score = lhs_feedback_score(
            (1.0, 2.0, 3.0, 4.0),
            (100.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(mean, 2.5)
        self.assertEqual(lower, 1.0)
        self.assertEqual(secondary, 25.0)
        self.assertEqual(count, 1)
        self.assertEqual(score, 27.125)

    def test_lhs_environment_feedback_and_trajectory_reconstruct_reward(
        self,
    ) -> None:
        fake = _CountingCrafter(
            achievement_name="collect_wood",
            reward=-0.75,
        )
        benchmark = CrafterLongHorizonSurvivalBenchmark(
            CrafterConfig(max_episode_steps=8)
        )
        with patch(
            "crafter_benchmarks.environment.crafter.Env",
            return_value=fake,
        ):
            environment = benchmark.make_environment(
                EpisodeSpec(environment_seed=1)
            )
            try:
                initial = environment.reset()
                first = environment.step(5)
                repeated = environment.step(5)
            finally:
                environment.close()

        assert isinstance(first.metrics, dict)
        first_components = cast(
            dict[str, PolicyValue],
            first.metrics["lhs_score_delta_components"],
        )
        self.assertAlmostEqual(
            cast(float, first_components["alive_survival"]), 0.01
        )
        self.assertAlmostEqual(
            cast(float, first_components["vital_survival"]), 0.03
        )
        self.assertAlmostEqual(
            cast(float, first_components["first_unlock"]),
            LHS_FIRST_UNLOCK_CREDITS["collect_wood"],
        )
        self.assertEqual(first.metrics["upstream_reward"], -0.75)
        self.assertNotAlmostEqual(first.reward, -0.75)

        record = EpisodeRecord(
            episode=benchmark.episodes("train", seed=2, count=1)[0],
            policy_seed=3,
            initial_observation=initial,
            transitions=(
                Transition(action=5, step=first),
                Transition(action=5, step=repeated),
            ),
        )
        feedback = benchmark.feedback((record,))
        self.assertAlmostEqual(feedback.score, first.reward + repeated.reward)
        assert isinstance(feedback.content, dict)
        aggregation = feedback.content["feedback_aggregation"]
        assert isinstance(aggregation, dict)
        self.assertEqual(aggregation["tail_selection"], "survival_return_only")
        self.assertEqual(aggregation["upper_tail_weight"], 0.0)
        survival_at = feedback.content["survival_at_steps"]
        assert isinstance(survival_at, dict)
        self.assertEqual(set(survival_at), {"150", "200", "250", "300", "400"})
        vital_quality = feedback.content["vital_quality"]
        assert isinstance(vital_quality, dict)
        by_age = vital_quality["by_episode_age"]
        assert isinstance(by_age, dict)
        first_band = by_age["0-99"]
        assert isinstance(first_band, dict)
        self.assertEqual(first_band["alive_steps"], 2)
        trajectory = tuple(
            json.loads(line)
            for line in gzip.decompress(
                feedback.artifacts[0].read_bytes()
            ).splitlines()
        )
        self.assertEqual(
            trajectory[0]["reward_profile"],
            "lhs",
        )
        self.assertIn("lhs_score_delta_components", trajectory[1])
        self.assertIn("lhs_repeat_diagnostics", trajectory[2])

    def test_lhs_policy_failure_is_zero_with_explicit_partial_trace(
        self,
    ) -> None:
        fake = _CountingCrafter(achievement_name="collect_wood")
        benchmark = CrafterLongHorizonSurvivalBenchmark(
            CrafterConfig(max_episode_steps=8)
        )
        episode = benchmark.episodes("train", seed=2, count=1)[0]
        with patch(
            "crafter_benchmarks.environment.crafter.Env",
            return_value=fake,
        ):
            environment = benchmark.make_environment(episode)
            try:
                initial = environment.reset()
                step = environment.step(5)
            finally:
                environment.close()
        record = EpisodeRecord(
            episode=episode,
            policy_seed=3,
            initial_observation=initial,
            transitions=(Transition(action=5, step=step),),
            policy_failure="invalid_action",
        )
        feedback = benchmark.feedback((record,))
        self.assertEqual(feedback.score, 0.0)
        assert isinstance(feedback.content, dict)
        summaries = feedback.content["episode_score_summaries"]
        assert isinstance(summaries, list)
        summary = summaries[0]
        assert isinstance(summary, dict)
        self.assertEqual(summary["return"], 0.0)
        self.assertEqual(summary["partial_return"], step.reward)
        self.assertIs(summary["partial_credit_discarded"], True)
        trajectory = tuple(
            json.loads(line)
            for line in gzip.decompress(
                feedback.artifacts[0].read_bytes()
            ).splitlines()
        )
        self.assertEqual(trajectory[0]["return"], 0.0)
        self.assertEqual(trajectory[0]["partial_return"], step.reward)
        self.assertEqual(trajectory[0]["failure"]["code"], "invalid_action")
        self.assertIs(trajectory[0]["included_in_feedback"], True)

    def test_spec_baseline_and_agent_skill_are_packaged(self) -> None:
        benchmark = CrafterBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "crafter/CrafterReward-v1/achievement-score-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 10_000)
        self.assertIn("policy.py", baseline_program().files)
        self.assertIn("PLAYER_GUIDE.md", baseline_program().files)
        self.assertIn(
            b"Crafter is an open-world survival game",
            baseline_program().read_bytes("PLAYER_GUIDE.md"),
        )
        self.assertIn(
            b"Ground creatures do not mine or remove terrain",
            baseline_program().read_bytes("PLAYER_GUIDE.md"),
        )
        self.assertIn(
            b"Stone shelter and safe waiting",
            baseline_program().read_bytes("PLAYER_GUIDE.md"),
        )
        self.assertIn(
            b"food and drink are consumed at half their awake rates",
            baseline_program().read_bytes("PLAYER_GUIDE.md"),
        )
        self.assertIn(
            b"Closing its remaining walkable opening",
            baseline_program().read_bytes("PLAYER_GUIDE.md"),
        )
        self.assertIn(
            b"with placed stone can join the entrance",
            baseline_program().read_bytes("PLAYER_GUIDE.md"),
        )
        agent_skill = AgentSkill.from_directory(
            Path(__file__).parents[1]
            / "skills"
            / "optimize-crafter-policy"
        )
        skill_instructions = agent_skill.read_bytes("SKILL.md").decode("utf-8")
        self.assertEqual(agent_skill.name, "optimize-crafter-policy")
        self.assertIn(
            "verifiable resource-facility-craft state machine",
            skill_instructions,
        )
        self.assertIn(
            "Do not mark an achievement complete merely because",
            skill_instructions,
        )
        self.assertIn(
            "Audit inherited controller bias",
            skill_instructions,
        )
        self.assertIn("expanding-square", skill_instructions)
        self.assertNotIn("environment_seed", skill_instructions)
        symbolic_program = local_symbolic_baseline_program()
        self.assertIn("policy.py", symbolic_program.files)
        self.assertIn("PLAYER_GUIDE.md", symbolic_program.files)
        self.assertIn(
            b"local-symbolic Crafter starting Policy",
            symbolic_program.read_bytes("policy.py"),
        )
        symbolic_skill = AgentSkill.from_directory(
            Path(__file__).parents[1]
            / "skills"
            / "optimize-crafter-local-symbolic-policy"
        )
        self.assertEqual(
            symbolic_skill.name,
            "optimize-crafter-local-symbolic-policy",
        )
        self.assertIn(
            b"player is at\n  `[3, 4]`",
            symbolic_skill.read_bytes("SKILL.md"),
        )

        lhs = CrafterLongHorizonSurvivalBenchmark()
        self.assertEqual(
            lhs.spec.id,
            (
                "crafter/CrafterReward-v1/"
                "long-horizon-survival-score-v1"
            ),
        )
        self.assertEqual(
            lhs.spec.primary_metric,
            "long_horizon_survival_score",
        )
        self.assertEqual(
            lhs.spec.environment_parameters[
                "reward_profile"
            ],
            "lhs",
        )
        lhs_aggregation = lhs.spec.metadata[
            "feedback_aggregation"
        ]
        assert isinstance(lhs_aggregation, dict)
        self.assertEqual(
            lhs_aggregation["tail_selection"], "survival_return_only"
        )
        self.assertEqual(lhs_aggregation["upper_tail_weight"], 0.0)

    def test_baseline_direct_evaluation(self) -> None:
        def run_evaluation() -> EvaluationResult:
            return evaluate(
                baseline_program(),
                CrafterBenchmark(CrafterConfig(max_episode_steps=256)),
                execution=ProcessExecution.unsafe(),
                config=EvaluationConfig(
                    split="validation",
                    episodes=1,
                    seed=5,
                    episode_timeout_seconds=30,
                ),
            )

        result = run_evaluation()
        repeated = run_evaluation()
        self.assertEqual(result, repeated)
        self.assertEqual(
            result.benchmark_id,
            "crafter/CrafterReward-v1/achievement-score-v1",
        )
        self.assertEqual(len(result.episodes), 1)
        self.assertIsNone(result.episodes[0].failure)
        self.assertTrue(math.isfinite(result.feedback.score))

    def test_local_symbolic_baseline_direct_evaluation(self) -> None:
        result = evaluate(
            local_symbolic_baseline_program(),
            CrafterBenchmark(
                CrafterConfig(
                    max_episode_steps=64,
                    observation_profile="local-symbolic-v1",
                )
            ),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=30,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            "crafter/CrafterReward-v1/local-symbolic-v1/achievement-score-v1",
        )
        self.assertIsNone(result.episodes[0].failure)
        self.assertTrue(math.isfinite(result.feedback.score))

    def test_lhs_baseline_direct_evaluation(self) -> None:
        result = evaluate(
            baseline_program(),
            CrafterLongHorizonSurvivalBenchmark(
                CrafterConfig(max_episode_steps=256)
            ),
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=30,
            ),
        )

        self.assertEqual(
            result.benchmark_id,
            (
                "crafter/CrafterReward-v1/"
                "long-horizon-survival-score-v1"
            ),
        )
        self.assertIsNone(result.episodes[0].failure)
        assert isinstance(result.feedback.content, dict)
        components = result.feedback.content["score_components"]
        assert isinstance(components, dict)
        self.assertLessEqual(
            abs(cast(float, components["reconstruction_error"])),
            1e-12,
        )
        aggregation = result.feedback.content["feedback_aggregation"]
        assert isinstance(aggregation, dict)
        self.assertEqual(aggregation["tail_selection"], "survival_return_only")
        self.assertEqual(aggregation["upper_tail_weight"], 0.0)
        self.assertEqual(result.feedback.artifacts, ())

    def test_baseline_exposes_executable_empty_capability_scaffold(self) -> None:
        perception = VisualTranslationModule().translate(_ZERO_OBSERVATION.data)
        self.assertEqual(perception.rgb, _ZERO_OBSERVATION.data)
        self.assertIsNone(perception.health)
        self.assertEqual(perception.tiles, ())
        self.assertEqual(perception.entities, ())

        world = WorldMemoryModule()
        memory = world.update(
            perception,
            last_action=2,
            last_proposal_source="test",
        )
        self.assertEqual(memory.step, 1)
        self.assertEqual(memory.last_action, 2)
        self.assertEqual(memory.last_proposal_source, "test")
        self.assertEqual(memory.explored, set())
        self.assertEqual(memory.water_sources, set())
        self.assertEqual(memory.facility_locations, {})

        capabilities = (
            ExplorationModule(),
            SurvivalModule(),
            ProductionModule(),
            CombatDefenseModule(),
        )
        self.assertTrue(
            all(
                capability.propose(perception, memory) is None
                for capability in capabilities
            )
        )

        coordinator = Coordinator()
        proposal = ActionProposal(action=5, source="test")
        competing = ActionProposal(action=1, source="other")
        self.assertIsNone(coordinator.select(()))
        self.assertEqual(coordinator.select((proposal,)), proposal)
        self.assertIsNone(coordinator.select((proposal, competing)))

        policy = BaselinePolicy(7)
        first_action = policy.act(_ZERO_OBSERVATION)
        policy.act(_ZERO_OBSERVATION)
        self.assertEqual(policy.world.state.step, 2)
        self.assertEqual(policy.world.state.last_action, first_action)
        self.assertEqual(policy.world.state.last_proposal_source, "fallback")

        class InvalidCapability:
            def propose(
                self,
                perception: object,
                memory: object,
            ) -> ActionProposal:
                del perception, memory
                return ActionProposal(action=17, source="invalid")

        policy = BaselinePolicy(7)
        policy.capabilities = (InvalidCapability(),)
        with self.assertRaisesRegex(ValueError, "invalid Action"):
            policy.act(_ZERO_OBSERVATION)

    def test_baseline_uses_seeded_nonreversing_short_macros(self) -> None:
        policy = BaselinePolicy(7)
        repeated_policy = BaselinePolicy(7)
        different_policy = BaselinePolicy(8)
        actions = tuple(policy.act(_ZERO_OBSERVATION) for _ in range(128))
        repeated = tuple(
            repeated_policy.act(_ZERO_OBSERVATION) for _ in range(128)
        )
        different = tuple(
            different_policy.act(_ZERO_OBSERVATION) for _ in range(128)
        )

        self.assertEqual(actions, repeated)
        self.assertNotEqual(actions, different)
        self.assertNotIn((5, 5), tuple(zip(actions, actions[1:], strict=False)))

        policy = BaselinePolicy(7)
        opposite = {1: 2, 2: 1, 3: 4, 4: 3}
        previous_direction: int | None = None
        seen_directions: set[int] = set()
        for _ in range(20):
            macro: list[int] = []
            while not macro or macro[-1] != 5:
                macro.append(policy.act(_ZERO_OBSERVATION))
            direction = macro[0]
            self.assertIn(direction, {1, 2, 3, 4})
            self.assertTrue(all(action == direction for action in macro[:-1]))
            self.assertIn(len(macro) - 1, range(2, 6))
            if previous_direction is not None:
                self.assertNotEqual(direction, opposite[previous_direction])
            seen_directions.add(direction)
            previous_direction = direction
        self.assertEqual(seen_directions, {1, 2, 3, 4})

    def test_symbolic_baseline_matches_rgb_starting_action_stream(self) -> None:
        rgb = BaselinePolicy(7)
        symbolic = LocalSymbolicBaselinePolicy(7)

        self.assertEqual(
            tuple(rgb.act(_ZERO_OBSERVATION) for _ in range(256)),
            tuple(
                symbolic.act(_ZERO_SYMBOLIC_OBSERVATION)
                for _ in range(256)
            ),
        )


def _record(
    achievements: tuple[str, ...],
    *,
    reward: float,
    episode: EpisodeSpec | None = None,
) -> EpisodeRecord:
    step = Step(
        observation=_ZERO_OBSERVATION,
        reward=reward,
        terminated=True,
        metrics={"achievements_unlocked": list(achievements)},
    )
    return EpisodeRecord(
        episode=(
            EpisodeSpec(environment_seed=10)
            if episode is None
            else episode
        ),
        policy_seed=20,
        initial_observation=_ZERO_OBSERVATION,
        transitions=(Transition(action=5, step=step),),
    )


def _event_record(
    *,
    steps: int,
    unlocked: tuple[str, ...],
    event_counts: dict[str, int],
    vitals: tuple[tuple[int, int, int], ...] | None = None,
) -> EpisodeRecord:
    if vitals is not None and len(vitals) != steps:
        raise ValueError("vitals must align with steps")
    transitions = tuple(
        Transition(
            action=5,
            step=Step(
                observation=_ZERO_OBSERVATION,
                reward=0.0,
                terminated=False,
                truncated=index == steps - 1,
                metrics={
                    "achievements_unlocked": (
                        list(unlocked) if index == 0 else []
                    ),
                    "achievement_event_counts": (
                        dict(event_counts) if index == 0 else {}
                    ),
                    **(
                        {}
                        if vitals is None
                        else {
                            "maintenance_vitals": {
                                "health": vitals[index][0],
                                "food": vitals[index][1],
                                "drink": vitals[index][2],
                            }
                        }
                    ),
                },
            ),
        )
        for index in range(steps)
    )
    return EpisodeRecord(
        episode=EpisodeSpec(environment_seed=10),
        policy_seed=20,
        initial_observation=_ZERO_OBSERVATION,
        transitions=transitions,
    )


def _m2_s1_record(
    *,
    steps: int,
    unlocked: tuple[str, ...] = (),
    terminated: bool = False,
    survival_credit: float = 0.05,
) -> EpisodeRecord:
    transitions: list[Transition] = []
    for index in range(steps):
        final = index == steps - 1
        upstream_reward = 1.0 if index == 0 and unlocked else 0.0
        transition_survival_credit = (
            0.0 if final and terminated else survival_credit
        )
        transitions.append(
            Transition(
                action=5,
                step=Step(
                    observation=_ZERO_OBSERVATION,
                    reward=upstream_reward + transition_survival_credit,
                    terminated=final and terminated,
                    truncated=final and not terminated,
                    metrics={
                        "achievements_unlocked": (
                            list(unlocked) if index == 0 else []
                        ),
                        "achievement_event_counts": (
                            {name: 1 for name in unlocked}
                            if index == 0
                            else {}
                        ),
                        "upstream_reward": upstream_reward,
                        "survival_credit": transition_survival_credit,
                    },
                ),
            )
        )
    return EpisodeRecord(
        episode=EpisodeSpec(environment_seed=10),
        policy_seed=20,
        initial_observation=_ZERO_OBSERVATION,
        transitions=tuple(transitions),
    )


if __name__ == "__main__":
    unittest.main()
