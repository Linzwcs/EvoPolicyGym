from __future__ import annotations

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
from PIL import Image

from crafter_benchmarks import (
    ACHIEVEMENTS,
    ACTIONS,
    CrafterBenchmark,
    CrafterConfig,
    baseline_program,
)

_ZERO_OBSERVATION = TensorValue(
    dtype="uint8",
    shape=(64, 64, 3),
    data=bytes(64 * 64 * 3),
)


class _CountingCrafter:
    action_names = ACTIONS

    def __init__(self, *, done: bool = False, discount: float = 1.0) -> None:
        self.steps = 0
        self.done = done
        self.discount = discount

    def reset(self) -> numpy.ndarray:
        return numpy.zeros((64, 64, 3), dtype=numpy.uint8)

    def step(self, action: int) -> tuple[object, float, bool, dict[str, object]]:
        del action
        self.steps += 1
        return (
            numpy.zeros((64, 64, 3), dtype=numpy.uint8),
            0.0,
            self.done,
            {
                "discount": self.discount,
                "achievements": {name: 0 for name in ACHIEVEMENTS},
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
                "trace.jsonl",
                "episode-0-frames.png",
                "episode-0-replay.mp4",
                "episode-1-frames.png",
                "artifact-manifest.json",
            ),
        )
        self.assertTrue(
            feedback.artifacts[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
        self.assertEqual(
            feedback.artifacts[2].read_bytes()[4:8],
            b"ftyp",
        )
        with Image.open(io.BytesIO(feedback.artifacts[1].read_bytes())) as image:
            self.assertEqual(image.size, (1_024, 1_024))
        manifest = json.loads(feedback.artifacts[-1].read_bytes())
        self.assertEqual(manifest["schema"], "crafter/artifact-manifest/v1")
        self.assertEqual(manifest["traced_episodes"], 2)
        self.assertEqual(len(manifest["episodes"]), 2)
        self.assertEqual(
            manifest["episodes"][0]["replay"]["frame_size"],
            [512, 512],
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

        self.assertEqual(len(feedback.artifacts), 7)
        self.assertTrue(
            all(artifact.size <= ARTIFACT_MAX_BYTES for artifact in feedback.artifacts)
        )
        documents = tuple(
            json.loads(line)
            for line in feedback.artifacts[0].read_bytes().splitlines()
        )
        episodes = tuple(
            document for document in documents if document["type"] == "episode"
        )
        self.assertEqual(len(episodes), 4)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["trace_episodes_omitted"], 996)

    def test_spec_baseline_and_agent_skill_are_packaged(self) -> None:
        benchmark = CrafterBenchmark()
        self.assertEqual(
            benchmark.spec.id,
            "crafter/CrafterReward-v1/achievement-score-v1",
        )
        self.assertEqual(benchmark.spec.max_episode_steps, 10_000)
        self.assertIn("policy.py", baseline_program().files)
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
        self.assertNotIn("environment_seed", benchmark.spec.agent_skill)

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


if __name__ == "__main__":
    unittest.main()
