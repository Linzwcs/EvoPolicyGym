from __future__ import annotations

import gzip
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
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

from crafter_benchmarks import (
    ACHIEVEMENTS,
    ACTIONS,
    CrafterBenchmark,
    CrafterConfig,
    CrafterLongHorizonBenchmark,
    baseline_program,
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

_ZERO_OBSERVATION = TensorValue(
    dtype="uint8",
    shape=(64, 64, 3),
    data=bytes(64 * 64 * 3),
)


class _CountingCrafter:
    action_names = ACTIONS

    def __init__(
        self,
        *,
        done: bool = False,
        discount: float = 1.0,
        achievement_name: str | None = None,
    ) -> None:
        self.steps = 0
        self.done = done
        self.discount = discount
        self.achievement_name = achievement_name
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
            0.0,
            self.done,
            {
                "discount": self.discount,
                "achievements": dict(self.achievements),
                "inventory": {"health": 9, "food": 9, "drink": 9},
            },
        )


class CrafterBenchmarkTests(unittest.TestCase):
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
        self.assertTrue(all(item.scenario is None for item in train))

    def test_environment_replays_deterministically(self) -> None:
        report = check_benchmark(
            CrafterBenchmark(CrafterConfig(max_episode_steps=32)),
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=123),
                    actions=(0, 1, 3, 5, 2, 4),
                ),
            ),
        )
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
                "bulk/episodes/episode-000000/trajectory-000000.jsonl.gz",
                "bulk/episodes/episode-000001/trajectory-000000.jsonl.gz",
                "bulk/observations-000000.npz",
                "artifact-manifest.json",
            ),
        )
        self.assertEqual(
            tuple(artifact.retention for artifact in feedback.artifacts),
            ("bulk", "bulk", "bulk", "permanent"),
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
        with numpy.load(
            io.BytesIO(feedback.artifacts[2].read_bytes()),
            allow_pickle=False,
        ) as archive:
            self.assertEqual(archive["observations"].shape, (3, 64, 64, 3))
            self.assertEqual(archive["observations"].dtype, numpy.uint8)
            self.assertEqual(
                archive["episode_indices"].tolist(),
                [0, 0, 1],
            )
            self.assertEqual(
                archive["observation_indices"].tolist(),
                [0, 1, 0],
            )
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        self.assertEqual(
            manifest["schema"],
            "crafter/complete-feedback-manifest/v1",
        )
        self.assertIs(manifest["complete"], True)
        self.assertEqual(manifest["episodes"], 2)
        self.assertEqual(manifest["transitions"], 1)
        self.assertEqual(manifest["observations"], 3)

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

        trajectory_artifacts = tuple(
            artifact
            for artifact in feedback.artifacts
            if artifact.name.endswith("trajectory-000000.jsonl.gz")
        )
        observation_artifacts = tuple(
            artifact
            for artifact in feedback.artifacts
            if artifact.name.endswith(".npz")
        )
        self.assertEqual(len(trajectory_artifacts), 1_000)
        self.assertEqual(len(observation_artifacts), 2)
        self.assertEqual(len(feedback.artifacts), 1_003)
        self.assertTrue(
            all(artifact.size <= ARTIFACT_MAX_BYTES for artifact in feedback.artifacts)
        )
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        detailed = feedback.content["detailed_feedback"]
        self.assertIsInstance(detailed, dict)
        assert isinstance(detailed, dict)
        self.assertIs(detailed["complete"], True)
        self.assertEqual(detailed["episodes"], 1_000)
        self.assertEqual(detailed["transitions"], 1_000)
        self.assertEqual(detailed["observations"], 2_000)

    def test_feedback_preserves_exact_policy_observation_bytes(self) -> None:
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
        observation_artifact = next(
            artifact
            for artifact in feedback.artifacts
            if artifact.name.endswith(".npz")
        )
        with numpy.load(
            io.BytesIO(observation_artifact.read_bytes()),
            allow_pickle=False,
        ) as archive:
            frames = archive["observations"]
            self.assertEqual(frames[0].tobytes(), initial.data)
            self.assertEqual(frames[1].tobytes(), following.data)

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

    def test_long_horizon_profile_gates_productivity_and_innovation(self) -> None:
        record = _event_record(
            steps=300,
            unlocked=("collect_drink", "collect_wood"),
            event_counts={"collect_drink": 9, "collect_wood": 9},
        )

        feedback = CrafterLongHorizonBenchmark().feedback((record,))

        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        components = feedback.content["score_components"]
        self.assertIsInstance(components, dict)
        assert isinstance(components, dict)
        survival = 100 / 7
        productivity = 10.0
        innovation = 200 / len(ACHIEVEMENTS)
        expected = survival * (
            0.70 + 0.10 * productivity / 100 + 0.05 * innovation / 100
        )
        survival_component = components["survival_score_percent"]
        productivity_component = components["productivity_score_percent"]
        innovation_component = components["innovation_score_percent"]
        self.assertIsInstance(survival_component, float)
        self.assertIsInstance(productivity_component, float)
        self.assertIsInstance(innovation_component, float)
        assert isinstance(survival_component, float)
        assert isinstance(productivity_component, float)
        assert isinstance(innovation_component, float)
        self.assertAlmostEqual(feedback.score, expected)
        self.assertAlmostEqual(survival_component, survival)
        self.assertAlmostEqual(
            productivity_component,
            productivity,
        )
        self.assertAlmostEqual(
            innovation_component,
            innovation,
        )
        survival_at_steps = feedback.content["survival_at_steps"]
        self.assertIsInstance(survival_at_steps, dict)
        assert isinstance(survival_at_steps, dict)
        self.assertEqual(
            survival_at_steps["300"],
            {"count": 1, "percent": 100.0},
        )
        self.assertEqual(
            survival_at_steps["600"],
            {"count": 0, "percent": 0.0},
        )
        self.assertLessEqual(feedback.score, survival)

    def test_long_horizon_productivity_is_capped_and_excludes_maintenance_spam(
        self,
    ) -> None:
        capped = CrafterLongHorizonBenchmark().feedback(
            (
                _event_record(
                    steps=1,
                    unlocked=("collect_wood",),
                    event_counts={"collect_wood": 9},
                ),
            )
        )
        excessive = CrafterLongHorizonBenchmark().feedback(
            (
                _event_record(
                    steps=1,
                    unlocked=("collect_wood",),
                    event_counts={"collect_wood": 100},
                ),
            )
        )
        tool_spam = CrafterLongHorizonBenchmark().feedback(
            (
                _event_record(
                    steps=1,
                    unlocked=("make_wood_pickaxe",),
                    event_counts={"make_wood_pickaxe": 100},
                ),
            )
        )
        maintenance_spam = CrafterLongHorizonBenchmark().feedback(
            (
                _event_record(
                    steps=1,
                    unlocked=("collect_drink",),
                    event_counts={"collect_drink": 100},
                ),
            )
        )

        for feedback in (capped, excessive, tool_spam, maintenance_spam):
            self.assertIsInstance(feedback.content, dict)
        assert isinstance(capped.content, dict)
        assert isinstance(excessive.content, dict)
        assert isinstance(tool_spam.content, dict)
        assert isinstance(maintenance_spam.content, dict)
        capped_components = capped.content["score_components"]
        excessive_components = excessive.content["score_components"]
        tool_components = tool_spam.content["score_components"]
        maintenance_components = maintenance_spam.content["score_components"]
        assert isinstance(capped_components, dict)
        assert isinstance(excessive_components, dict)
        assert isinstance(tool_components, dict)
        assert isinstance(maintenance_components, dict)
        self.assertEqual(
            capped_components["productivity_score_percent"],
            excessive_components["productivity_score_percent"],
        )
        self.assertEqual(tool_components["productivity_score_percent"], 0.0)
        self.assertEqual(
            maintenance_components["productivity_score_percent"],
            0.0,
        )

    def test_long_horizon_maintenance_credits_low_state_recovery(self) -> None:
        record = _event_record(
            steps=4,
            unlocked=(),
            event_counts={},
            vitals=(
                (4, 5, 5),
                (5, 9, 9),
                (7, 9, 9),
                (7, 9, 9),
            ),
        )

        feedback = CrafterLongHorizonBenchmark().feedback((record,))

        assert isinstance(feedback.content, dict)
        components = feedback.content["score_components"]
        assert isinstance(components, dict)
        maintenance = 100.0 * (
            math.log1p(2) / math.log1p(3)
            + math.log1p(1) / math.log1p(3)
            + math.log1p(1) / math.log1p(3)
        ) / 3
        survival = 100.0 * 4 / 2100
        maintenance_component = components["maintenance_score_percent"]
        self.assertIsInstance(maintenance_component, float)
        assert isinstance(maintenance_component, float)
        self.assertAlmostEqual(
            maintenance_component,
            maintenance,
        )
        self.assertAlmostEqual(
            feedback.score,
            survival * (0.70 + 0.15 * maintenance / 100),
        )
        self.assertEqual(
            feedback.content["maintenance_recovery_counts"],
            {"health": 2, "food": 1, "drink": 1},
        )

    def test_long_horizon_policy_failure_receives_zero_components(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_ZERO_OBSERVATION,
            transitions=(),
            policy_failure="invalid_action",
        )

        feedback = CrafterLongHorizonBenchmark().feedback((failed,))

        self.assertEqual(feedback.score, 0.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        components = feedback.content["score_components"]
        self.assertIsInstance(components, dict)
        assert isinstance(components, dict)
        self.assertEqual(components["survival_score_percent"], 0.0)
        self.assertEqual(components["maintenance_score_percent"], 0.0)
        self.assertEqual(components["productivity_score_percent"], 0.0)
        self.assertEqual(components["innovation_score_percent"], 0.0)

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
        self.assertIsNotNone(benchmark.spec.agent_skill)
        assert benchmark.spec.agent_skill is not None
        self.assertIn("optimize-crafter-policy", benchmark.spec.agent_skill)
        self.assertIn(
            "verifiable resource-facility-craft state machine",
            benchmark.spec.agent_skill,
        )
        self.assertIn(
            "Do not mark an achievement complete merely because",
            benchmark.spec.agent_skill,
        )
        self.assertIn(
            "Audit inherited controller bias",
            benchmark.spec.agent_skill,
        )
        self.assertIn("expanding-square", benchmark.spec.agent_skill)
        self.assertNotIn("environment_seed", benchmark.spec.agent_skill)

        long_horizon = CrafterLongHorizonBenchmark()
        self.assertEqual(
            long_horizon.spec.id,
            "crafter/CrafterReward-v1/long-horizon-development-v2",
        )
        self.assertEqual(
            long_horizon.spec.primary_metric,
            "long_horizon_development_score",
        )
        self.assertIn("episode_score_formula", long_horizon.spec.metadata)
        self.assertIn("productivity", long_horizon.spec.metadata)
        self.assertIn("maintenance", long_horizon.spec.metadata)
        self.assertIsNone(long_horizon.spec.agent_skill)
        with self.assertRaisesRegex(ValueError, "at least 900"):
            CrafterLongHorizonBenchmark(CrafterConfig(max_episode_steps=899))

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


def _record(
    achievements: tuple[str, ...],
    *,
    reward: float,
) -> EpisodeRecord:
    step = Step(
        observation=_ZERO_OBSERVATION,
        reward=reward,
        terminated=True,
        metrics={"achievements_unlocked": list(achievements)},
    )
    return EpisodeRecord(
        episode=EpisodeSpec(environment_seed=10),
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


if __name__ == "__main__":
    unittest.main()
