from __future__ import annotations

import gzip
import importlib
import io
import json
import math
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy
from evopolicygym import EvaluationConfig, EvaluationResult, Program, evaluate
from evopolicygym.artifacts import ARTIFACT_MAX_BYTES
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
from evopolicygym.results import ValidationCandidateResult, ValidationResult
from evopolicygym.skills import AgentSkill

from nle_benchmarks import (
    ACTION_MEANINGS,
    BENCHMARK_ID,
    NetHackBenchmark,
    NetHackConfig,
    baseline_program,
)
from nle_benchmarks.constants import (
    BLSTAT_NAMES,
    CONDITION_BITS,
    NETHACK_OPTIONS,
    OBSERVATION_KEYS,
    PENALTY_MODE,
    PENALTY_STEP,
    PENALTY_TIME,
    RAW_ACTIONS,
)
from nle_benchmarks.diagnostics import identical_validation_feedback_groups
from nle_benchmarks.evidence import (
    MAX_PUBLIC_FEEDBACK_EPISODES,
    OBSERVATION_CHUNK_SIZE,
)
from nle_benchmarks.observation import project_observation
from nle_benchmarks.programs.baseline.policy import BaselinePolicy


class _FakeNLE:
    def __init__(
        self,
        *,
        terminated: bool = False,
        upstream_truncated: bool = False,
        end_status: int | None = None,
    ) -> None:
        self.steps = 0
        self.close_calls = 0
        self.terminated = terminated
        self.upstream_truncated = upstream_truncated
        self.end_status = end_status

    def reset(self) -> tuple[object, object]:
        return _raw_observation(), {"end_status": 0, "is_ascended": False}

    def step(self, action: int) -> tuple[object, object, object, object, object]:
        del action
        self.steps += 1
        observation = _raw_observation(score=self.steps, turn=self.steps)
        return (
            observation,
            float(self.steps),
            self.terminated,
            self.upstream_truncated,
            {
                "end_status": (
                    self.end_status
                    if self.end_status is not None
                    else 1
                    if self.terminated
                    else -1
                    if self.upstream_truncated
                    else 0
                ),
                "is_ascended": False,
            },
        )

    def close(self) -> None:
        self.close_calls += 1


class _ConstructorFakeNLE(_FakeNLE):
    def __init__(self) -> None:
        super().__init__()
        self.actions = RAW_ACTIONS
        self.action_space = SimpleNamespace(n=len(RAW_ACTIONS))

    def seed(
        self,
        core: int,
        disp: int,
        reseed: bool,
        lgen: int,
    ) -> tuple[int, int, bool, int]:
        return core, disp, reseed, lgen


class NetHackBenchmarkTests(unittest.TestCase):
    def test_episode_planning_is_reproducible_split_scoped_and_bounded(self) -> None:
        benchmark = NetHackBenchmark()
        train = tuple(benchmark.episodes("train", seed=7, count=10))
        repeated = tuple(benchmark.episodes("train", seed=7, count=10))
        validation = tuple(benchmark.episodes("validation", seed=7, count=10))
        assessment = tuple(benchmark.episodes("test", seed=7, count=256))
        core16_train_pool = tuple(
            benchmark.episodes("train", seed=7, count=1_024)
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
                item.scenario == {"feedback_scope": "public_training"}
                for item in train
            )
        )
        self.assertTrue(
            all(
                item.scenario == {"feedback_scope": "aggregate_only"}
                for item in validation + assessment
            )
        )
        self.assertEqual(len(core16_train_pool), 1_024)

    def test_actual_environment_replays_deterministically(self) -> None:
        report = check_benchmark(
            NetHackBenchmark(NetHackConfig(max_episode_steps=32)),
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(1, 2, 3, 4, 22, 19),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_actual_observation_is_public_and_bounded(self) -> None:
        environment = NetHackBenchmark(
            NetHackConfig(max_episode_steps=8)
        ).make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, dict)
            assert isinstance(observation, dict)
            self.assertEqual(
                set(observation),
                {"screen", "stats", "message", "inventory", "input_mode"},
            )
            screen = observation["screen"]
            self.assertIsInstance(screen, dict)
            assert isinstance(screen, dict)
            for name, dtype in (
                ("glyphs", "int16"),
                ("chars", "uint8"),
                ("colors", "uint8"),
            ):
                tensor = screen[name]
                self.assertIsInstance(tensor, TensorValue)
                assert isinstance(tensor, TensorValue)
                self.assertEqual(tensor.dtype, dtype)
                self.assertEqual(tensor.shape, (21, 79))
            serialized = repr(observation)
            self.assertNotIn("environment_seed", serialized)
            self.assertNotIn("policy_seed", serialized)
            self.assertNotIn("internal", serialized)
        finally:
            environment.close()
            environment.close()

    def test_upstream_disables_rendering_ttyrec_and_saved_files(self) -> None:
        captured: dict[str, object] = {}
        fake = _ConstructorFakeNLE()

        def construct(**keywords: object) -> _ConstructorFakeNLE:
            captured.update(keywords)
            return fake

        with patch("nle_benchmarks.environment.NetHackScore", side_effect=construct):
            from nle_benchmarks.environment import _make_upstream

            environment = _make_upstream(
                NetHackConfig(max_episode_steps=8),
                seeds=(1, 2, 3),
            )
        self.assertIs(environment, fake)
        self.assertIsNone(captured["render_mode"])
        self.assertIsNone(captured["savedir"])
        self.assertEqual(captured["save_ttyrec_every"], 0)
        self.assertEqual(captured["options"], NETHACK_OPTIONS)
        self.assertEqual(captured["penalty_mode"], PENALTY_MODE)
        self.assertEqual(captured["penalty_step"], PENALTY_STEP)
        self.assertEqual(captured["penalty_time"], PENALTY_TIME)

    def test_invalid_actions_do_not_advance_upstream(self) -> None:
        fake = _FakeNLE()
        with patch("nle_benchmarks.environment._make_upstream", return_value=fake):
            environment = NetHackBenchmark(
                NetHackConfig(max_episode_steps=8)
            ).make_environment(EpisodeSpec(environment_seed=1))
            try:
                environment.reset()
                invalid: tuple[PolicyValue, ...] = (
                    True,
                    -1,
                    23,
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
        self.assertEqual(fake.close_calls, 1)

    def test_policy_failure_stops_before_upstream_step(self) -> None:
        fake = _FakeNLE()
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
                "nle_benchmarks.environment._make_upstream",
                return_value=fake,
            ):
                result = evaluate(
                    program,
                    NetHackBenchmark(NetHackConfig(max_episode_steps=8)),
                    execution=ProcessExecution.unsafe(),
                    config=EvaluationConfig(
                        split="train",
                        episodes=1,
                        seed=3,
                        episode_timeout_seconds=30,
                    ),
                )

        self.assertEqual(result.episodes[0].failure, "invalid_action")
        self.assertEqual(result.feedback.score, -8.0)
        self.assertEqual(fake.steps, 0)
        self.assertEqual(len(result.feedback.artifacts), 3)

    def test_horizon_death_and_cleanup_semantics(self) -> None:
        for fake, terminated, truncated in (
            (_FakeNLE(), False, True),
            (_FakeNLE(terminated=True, end_status=-1), False, True),
            (_FakeNLE(terminated=True), True, False),
        ):
            with patch(
                "nle_benchmarks.environment._make_upstream",
                return_value=fake,
            ):
                environment = NetHackBenchmark(
                    NetHackConfig(max_episode_steps=1)
                ).make_environment(EpisodeSpec(environment_seed=1))
                try:
                    environment.reset()
                    step = environment.step(0)
                    self.assertIs(step.terminated, terminated)
                    self.assertIs(step.truncated, truncated)
                    self.assertEqual(fake.steps, 1)
                finally:
                    environment.close()

    def test_actual_nle_horizon_is_truncated_not_terminated(self) -> None:
        environment = NetHackBenchmark(
            NetHackConfig(max_episode_steps=1)
        ).make_environment(EpisodeSpec(environment_seed=321))
        try:
            environment.reset()
            step = environment.step(19)
            self.assertIs(step.terminated, False)
            self.assertIs(step.truncated, True)
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["end_status"], -1)
            with self.assertRaisesRegex(RuntimeError, "already complete"):
                environment.step(19)
        finally:
            environment.close()

    def test_training_feedback_is_complete_reversible_and_deterministic(self) -> None:
        initial = _public_observation(
            message="You see here a food ration.",
            input_mode="normal",
            inventory_description="a food ration",
            condition_mask=0x0120,
        )
        final = _public_observation(
            score=12,
            turn=10,
            message="You finish eating the food ration.",
            input_mode="more",
            inventory_description=None,
            condition_mask=0,
        )
        record = _record(
            reward=10.0,
            game_score=12,
            depth=3,
            initial_observation=initial,
            final_observation=final,
        )
        benchmark = NetHackBenchmark(NetHackConfig(max_episode_steps=100))

        feedback = benchmark.feedback((record,))
        repeated = benchmark.feedback((record,))

        self.assertEqual(feedback.score, 10.0)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["frozen_steps"], 0)
        self.assertEqual(feedback.content["mean_frozen_steps"], 0.0)
        self.assertEqual(feedback.content["frozen_step_fraction"], 0.0)
        self.assertEqual(feedback.content["mean_frozen_penalty"], 0.0)
        self.assertEqual(
            [(item.name, item.content) for item in feedback.artifacts],
            [(item.name, item.content) for item in repeated.artifacts],
        )
        self.assertEqual(len(feedback.artifacts), 3)
        self.assertTrue(
            all(item.size <= ARTIFACT_MAX_BYTES for item in feedback.artifacts)
        )
        manifest_artifact = next(
            item for item in feedback.artifacts if item.name == "artifact-manifest.json"
        )
        manifest = json.loads(manifest_artifact.content)
        self.assertIs(manifest["complete"], True)
        self.assertIs(manifest["visualization_generated"], False)
        self.assertEqual(manifest["episodes"], 1)
        self.assertEqual(manifest["transitions"], 1)
        self.assertEqual(manifest["observations"], 2)
        self.assertNotIn(b"environment_seed", manifest_artifact.content)
        self.assertNotIn(b"policy_seed", manifest_artifact.content)
        self.assertEqual(manifest_artifact.retention, "permanent")

        observation_artifact = next(
            item for item in feedback.artifacts if item.media_type == "application/x-npz"
        )
        self.assertEqual(observation_artifact.retention, "bulk")
        with numpy.load(
            io.BytesIO(observation_artifact.content),
            allow_pickle=False,
        ) as archive:
            self.assertEqual(
                set(archive.files),
                {
                    "episode_indices",
                    "observation_indices",
                    "glyphs",
                    "chars",
                    "colors",
                    "stats",
                    "message_bytes",
                    "message_lengths",
                    "inventory_counts",
                    "inventory_letters",
                    "inventory_descriptions",
                    "inventory_description_lengths",
                    "inventory_glyphs",
                    "inventory_object_classes",
                    "input_modes",
                },
            )
            self.assertEqual(_decode_observation(archive, 0), initial)
            self.assertEqual(_decode_observation(archive, 1), final)
            numpy.testing.assert_array_equal(archive["episode_indices"], [0, 0])
            numpy.testing.assert_array_equal(
                archive["observation_indices"],
                [0, 1],
            )

        trajectory_artifact = next(
            item for item in feedback.artifacts if item.media_type == "application/gzip"
        )
        self.assertEqual(trajectory_artifact.retention, "bulk")
        lines = [
            json.loads(line)
            for line in gzip.decompress(trajectory_artifact.content).splitlines()
        ]
        self.assertEqual(lines[0]["initial_observation_index"], 0)
        self.assertEqual(lines[0]["final_observation_index"], 1)
        self.assertEqual(lines[1]["observation_index"], 0)
        self.assertEqual(lines[1]["next_observation_index"], 1)
        self.assertEqual(lines[1]["action"], 22)
        self.assertEqual(lines[1]["action_name"], "search")

    def test_feedback_reports_exact_score_penalty_diagnostics(self) -> None:
        frozen = _record(
            reward=PENALTY_STEP,
            game_score=0,
            depth=1,
            initial_observation=_public_observation(turn=10),
        )
        feedback = NetHackBenchmark(
            NetHackConfig(max_episode_steps=100)
        ).feedback((frozen,))

        self.assertEqual(feedback.score, PENALTY_STEP)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["frozen_steps"], 1)
        self.assertEqual(feedback.content["mean_frozen_steps"], 1.0)
        self.assertEqual(feedback.content["frozen_step_fraction"], 1.0)
        self.assertEqual(
            feedback.content["mean_frozen_penalty"],
            PENALTY_STEP,
        )

    def test_host_phases_score_identically_without_detailed_artifacts(self) -> None:
        training = _record(reward=10.0, game_score=12, depth=3)
        aggregate = _record(
            reward=10.0,
            game_score=12,
            depth=3,
            scenario={"feedback_scope": "aggregate_only"},
        )
        benchmark = NetHackBenchmark(NetHackConfig(max_episode_steps=100))

        public_feedback = benchmark.feedback((training,))
        aggregate_feedback = benchmark.feedback((aggregate,))

        self.assertEqual(public_feedback.score, aggregate_feedback.score)
        self.assertTrue(public_feedback.artifacts)
        self.assertEqual(aggregate_feedback.artifacts, ())
        assert isinstance(aggregate_feedback.content, dict)
        detail = aggregate_feedback.content["detailed_feedback"]
        self.assertIsInstance(detail, dict)
        assert isinstance(detail, dict)
        self.assertEqual(detail["schema"], "nle/aggregate-only-host-phase/v1")
        invalid_scope = _record(
            reward=10.0,
            game_score=12,
            depth=3,
            scenario={"feedback_scope": []},
        )
        with self.assertRaisesRegex(ValueError, "Feedback scope is invalid"):
            benchmark.feedback((invalid_scope,))

    def test_feedback_artifact_bounds_cover_core16_worst_case(self) -> None:
        maximum_observations = MAX_PUBLIC_FEEDBACK_EPISODES * (5_000 + 1)
        chunks = math.ceil(maximum_observations / OBSERVATION_CHUNK_SIZE)
        total_artifacts = chunks + MAX_PUBLIC_FEEDBACK_EPISODES + 1
        self.assertEqual(chunks, 313)
        self.assertEqual(total_artifacts, 378)
        self.assertLessEqual(total_artifacts, 1_024)

        per_observation_bytes = (
            21 * 79 * 2
            + 21 * 79
            + 21 * 79
            + 27 * 8
            + 256
            + 2
            + 1
            + 55
            + 55 * 80
            + 55
            + 55 * 2
            + 55
            + 1
            + 4
            + 4
        )
        self.assertLess(
            per_observation_bytes * OBSERVATION_CHUNK_SIZE,
            ARTIFACT_MAX_BYTES,
        )

    def test_spec_baseline_agent_skill_and_tool_dependencies_are_packaged(self) -> None:
        benchmark = NetHackBenchmark()
        self.assertEqual(benchmark.spec.id, BENCHMARK_ID)
        self.assertEqual(benchmark.spec.max_episode_steps, 5_000)
        self.assertEqual(len(ACTION_MEANINGS), 23)
        self.assertIn("policy.py", baseline_program().files)
        agent_skill = AgentSkill.from_directory(
            Path(__file__).parents[1]
            / "skills"
            / "optimize-nethack-policy"
        )
        skill_instructions = agent_skill.read_bytes("SKILL.md").decode("utf-8")
        self.assertEqual(agent_skill.name, "optimize-nethack-policy")
        self.assertIn("analysis/", skill_instructions)
        self.assertIn("frozen_step_fraction", skill_instructions)
        self.assertIn("behaviorally distinct", skill_instructions)
        self.assertNotIn("environment_seed", skill_instructions)

        project_root = Path(__file__).parents[1]
        project = tomllib.loads((project_root / "pyproject.toml").read_text())
        dependencies = project["project"]["dependencies"]
        agent_tools = project["project"]["optional-dependencies"]["agent-tools"]
        self.assertTrue(any(item.startswith("numpy") for item in dependencies))
        self.assertTrue(any(item.startswith("pillow") for item in agent_tools))
        self.assertTrue(any(item.startswith("imageio") for item in agent_tools))
        for module in ("numpy", "PIL", "imageio.v3", "imageio_ffmpeg"):
            self.assertIsNotNone(importlib.import_module(module))

        runner = (project_root / "scripts" / "run_nle_codex.py").read_text()
        self.assertIn("view_image=True", runner)
        self.assertIn("bulk_feedback_retention_bytes", runner)
        self.assertIn("identical_validation_feedback_groups", runner)
        self.assertIn("program_digest", runner)
        self.assertNotIn("showcase", runner.lower())
        self.assertNotIn("human-video", runner.lower())
        self.assertFalse((project_root / "src/nle_benchmarks/showcase.py").exists())

    def test_baseline_direct_validation_is_deterministic_and_aggregate_only(self) -> None:
        def run_evaluation() -> EvaluationResult:
            return evaluate(
                baseline_program(),
                NetHackBenchmark(NetHackConfig(max_episode_steps=64)),
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
        self.assertEqual(result.benchmark_id, BENCHMARK_ID)
        self.assertEqual(len(result.episodes), 1)
        self.assertIsNone(result.episodes[0].failure)
        self.assertTrue(math.isfinite(result.feedback.score))
        self.assertEqual(result.feedback.artifacts, ())

    def test_baseline_is_observation_aware_and_seeded(self) -> None:
        observation = _public_observation()
        first = BaselinePolicy(7)
        repeated = BaselinePolicy(7)
        different = BaselinePolicy(8)
        actions = tuple(first.act(observation) for _ in range(32))
        repeated_actions = tuple(repeated.act(observation) for _ in range(32))
        different_actions = tuple(different.act(observation) for _ in range(32))

        self.assertEqual(actions, repeated_actions)
        self.assertNotEqual(actions, different_actions)
        self.assertTrue(
            all(type(action) is int and 0 <= action <= 22 for action in actions)
        )

    def test_runner_audits_only_exact_validation_feedback_duplicates(self) -> None:
        def candidate(
            identifier: str,
            digest_character: str,
            content: dict[str, PolicyValue],
        ) -> ValidationCandidateResult:
            return ValidationCandidateResult(
                submission_id=identifier,
                program_digest=f"sha256:{digest_character * 64}",
                score=10.0,
                episodes=4,
                policy_failures=0,
                feedback_content=content,
            )

        validation = ValidationResult(
            split="validation",
            episodes_per_candidate=4,
            primary_metric="mean_return",
            score_direction="maximize",
            candidates=(
                candidate("submission-a", "a", {"mean_return": 10.0}),
                candidate("submission-b", "b", {"mean_return": 10.0}),
                candidate(
                    "submission-c",
                    "c",
                    {"mean_return": 10.0, "max_depth": 2},
                ),
            ),
            selected_submission_id="submission-a",
        )

        self.assertEqual(
            identical_validation_feedback_groups(validation),
            [["submission-a", "submission-b"]],
        )


def _raw_observation(
    *,
    score: int = 0,
    turn: int = 0,
    message: str = "",
    input_mode: str = "normal",
    inventory_description: str | None = None,
    condition_mask: int = 0,
) -> dict[str, numpy.ndarray]:
    chars = numpy.full((21, 79), ord(" "), dtype=numpy.uint8)
    chars[9:12, 39:42] = ord(".")
    chars[10, 40] = ord("@")
    colors = numpy.zeros((21, 79), dtype=numpy.uint8)
    colors[10, 40] = 15
    glyphs = numpy.zeros((21, 79), dtype=numpy.int16)
    glyphs[10, 40] = 333
    blstats = numpy.zeros((27,), dtype=numpy.int64)
    blstats[0] = 40
    blstats[1] = 10
    blstats[9] = score
    blstats[10] = 10
    blstats[11] = 10
    blstats[12] = 1
    blstats[18] = 1
    blstats[20] = turn
    blstats[24] = 1
    blstats[25] = condition_mask
    message_array = numpy.zeros((256,), dtype=numpy.uint8)
    encoded_message = message.encode("latin-1")
    message_array[: len(encoded_message)] = numpy.frombuffer(
        encoded_message,
        dtype=numpy.uint8,
    )
    inv_glyphs = numpy.zeros((55,), dtype=numpy.int16)
    inv_strs = numpy.zeros((55, 80), dtype=numpy.uint8)
    inv_letters = numpy.zeros((55,), dtype=numpy.uint8)
    inv_oclasses = numpy.zeros((55,), dtype=numpy.uint8)
    if inventory_description is not None:
        encoded_inventory = inventory_description.encode("latin-1")
        inv_glyphs[0] = 2_144
        inv_strs[0, : len(encoded_inventory)] = numpy.frombuffer(
            encoded_inventory,
            dtype=numpy.uint8,
        )
        inv_letters[0] = ord("a")
        inv_oclasses[0] = ord("%")
    misc = numpy.zeros((3,), dtype=numpy.int32)
    mode_index = {"yes_no": 0, "get_line": 1, "more": 2}.get(input_mode)
    if input_mode != "normal" and mode_index is None:
        raise AssertionError("invalid test input mode")
    if mode_index is not None:
        misc[mode_index] = 1
    raw: dict[str, numpy.ndarray] = {
        "glyphs": glyphs,
        "chars": chars,
        "colors": colors,
        "blstats": blstats,
        "message": message_array,
        "inv_glyphs": inv_glyphs,
        "inv_strs": inv_strs,
        "inv_letters": inv_letters,
        "inv_oclasses": inv_oclasses,
        "misc": misc,
    }
    if set(raw) != set(OBSERVATION_KEYS):
        raise AssertionError("test observation keys drifted")
    return raw


def _public_observation(
    *,
    score: int = 0,
    turn: int = 0,
    message: str = "",
    input_mode: str = "normal",
    inventory_description: str | None = None,
    condition_mask: int = 0,
) -> dict[str, PolicyValue]:
    return project_observation(
        _raw_observation(
            score=score,
            turn=turn,
            message=message,
            input_mode=input_mode,
            inventory_description=inventory_description,
            condition_mask=condition_mask,
        )
    )


def _record(
    *,
    reward: float,
    game_score: int,
    depth: int,
    initial_observation: dict[str, PolicyValue] | None = None,
    final_observation: dict[str, PolicyValue] | None = None,
    scenario: PolicyValue = None,
) -> EpisodeRecord:
    initial = _public_observation() if initial_observation is None else initial_observation
    final = initial if final_observation is None else final_observation
    step = Step(
        observation=final,
        reward=reward,
        terminated=True,
        metrics={
            "game_score": game_score,
            "max_game_score": game_score,
            "depth": depth,
            "max_depth": depth,
            "experience_level": 1,
            "dungeon_level": depth,
            "hit_points": 0,
            "turn": 10,
            "ascended": False,
            "end_status": 1,
        },
    )
    return EpisodeRecord(
        episode=EpisodeSpec(environment_seed=10, scenario=scenario),
        policy_seed=20,
        initial_observation=initial,
        transitions=(Transition(action=22, step=step),),
    )


def _decode_observation(
    archive: numpy.lib.npyio.NpzFile,
    index: int,
) -> dict[str, PolicyValue]:
    stats: dict[str, PolicyValue] = {
        name: int(archive["stats"][index, position])
        for position, name in enumerate(BLSTAT_NAMES)
    }
    mask_value = stats["condition_mask"]
    if type(mask_value) is not int:
        raise AssertionError("decoded condition mask is invalid")
    mask = mask_value
    stats["conditions"] = [
        name for bit, name in CONDITION_BITS if mask & bit
    ]
    message_length = int(archive["message_lengths"][index])
    inventory: list[PolicyValue] = []
    for item_index in range(int(archive["inventory_counts"][index])):
        description_length = int(
            archive["inventory_description_lengths"][index, item_index]
        )
        inventory.append(
            {
                "letter": bytes(
                    [int(archive["inventory_letters"][index, item_index])]
                ).decode("latin-1"),
                "description": bytes(
                    archive["inventory_descriptions"][
                        index,
                        item_index,
                        :description_length,
                    ]
                ).decode("latin-1"),
                "glyph": int(archive["inventory_glyphs"][index, item_index]),
                "object_class": int(
                    archive["inventory_object_classes"][index, item_index]
                ),
            }
        )
    mode = {0: "normal", 1: "yes_no", 2: "get_line", 3: "more"}[
        int(archive["input_modes"][index])
    ]
    return {
        "screen": {
            "glyphs": _archive_tensor(archive["glyphs"][index], "int16"),
            "chars": _archive_tensor(archive["chars"][index], "uint8"),
            "colors": _archive_tensor(archive["colors"][index], "uint8"),
        },
        "stats": stats,
        "message": bytes(
            archive["message_bytes"][index, :message_length]
        ).decode("latin-1"),
        "inventory": inventory,
        "input_mode": mode,
    }


def _archive_tensor(value: numpy.ndarray, dtype: str) -> TensorValue:
    return TensorValue(
        dtype=dtype,
        shape=tuple(value.shape),
        data=numpy.ascontiguousarray(value).tobytes(order="C"),
    )


if __name__ == "__main__":
    unittest.main()
