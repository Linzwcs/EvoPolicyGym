from __future__ import annotations

import io
import json
import unittest

import numpy
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    Transition,
    check_benchmark,
)
from evopolicygym.policy import TensorValue
from PIL import Image, ImageSequence

from atari_benchmarks import AtariBenchmark, AtariConfig, baseline_program


class AtariBenchmarkTests(unittest.TestCase):
    def test_tetris_resets_and_steps(self) -> None:
        benchmark = AtariBenchmark()
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, TensorValue)
            assert isinstance(observation, TensorValue)
            self.assertEqual(observation.shape, (210, 160, 3))
            step = environment.step(0)
            self.assertIsInstance(step.reward, float)
        finally:
            environment.close()
            environment.close()

    def test_non_portable_game_and_invalid_action_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AtariConfig(game="Breakout")
        environment = AtariBenchmark().make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            AtariBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    (0,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_feedback_publishes_bounded_lossless_visual_trace(self) -> None:
        reward_steps = {
            20,
            35,
            50,
            65,
            80,
            95,
            110,
            125,
            140,
            155,
            170,
            185,
        }
        transitions = tuple(
            Transition(
                action=step_index % 5,
                step=Step(
                    observation=_frame(step_index + 1),
                    reward=1.0 if step_index in reward_steps else 0.0,
                    terminated=False,
                    truncated=step_index == 199,
                ),
            )
            for step_index in range(200)
        )
        record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_frame(0),
            transitions=transitions,
        )

        feedback = AtariBenchmark().feedback((record,))

        self.assertEqual(feedback.score, 12.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["traced_steps"], 48)
        self.assertEqual(feedback.content["trace_steps_omitted"], 152)
        summaries = feedback.content["episode_summaries"]
        self.assertIsInstance(summaries, list)
        assert isinstance(summaries, list)
        first_summary = summaries[0]
        self.assertIsInstance(first_summary, dict)
        assert isinstance(first_summary, dict)
        self.assertEqual(first_summary["positive_reward_events"], 12)
        self.assertEqual(first_summary["trace_steps_omitted"], 152)

        artifacts = {artifact.name: artifact for artifact in feedback.artifacts}
        self.assertEqual(
            set(artifacts),
            {
                "trace.jsonl",
                "episode-000/observations.npz",
                "episode-000/contact-sheet.png",
                "episode-000/replay.gif",
            },
        )
        trace = tuple(json.loads(line) for line in artifacts["trace.jsonl"].content.splitlines())
        transition_trace = tuple(document for document in trace if document["type"] == "transition")
        self.assertEqual(len(transition_trace), 48)
        self.assertIn(50, {item["step_index"] for item in transition_trace})
        self.assertEqual(
            transition_trace[0]["decision_observation"],
            {
                "artifact": "episode-000/observations.npz",
                "array": "decision_frames",
                "index": 0,
            },
        )

        with numpy.load(
            io.BytesIO(artifacts["episode-000/observations.npz"].content),
            allow_pickle=False,
        ) as frames:
            self.assertEqual(frames["initial_frame"].shape, (210, 160, 3))
            self.assertEqual(
                frames["decision_frames"].shape,
                (48, 210, 160, 3),
            )
            self.assertEqual(
                frames["result_frames"].shape,
                (48, 210, 160, 3),
            )
            self.assertIn(50, frames["step_indices"].tolist())
            numpy.testing.assert_array_equal(
                frames["decision_frames"][0],
                frames["initial_frame"],
            )
        self.assertTrue(
            artifacts["episode-000/contact-sheet.png"].content.startswith(b"\x89PNG\r\n\x1a\n")
        )
        replay_content = artifacts["episode-000/replay.gif"].content
        self.assertTrue(replay_content.startswith(b"GIF89a"))
        with Image.open(io.BytesIO(replay_content)) as replay:
            self.assertEqual(replay.format, "GIF")
            self.assertEqual(
                sum(1 for _ in ImageSequence.Iterator(replay)),
                24,
            )
            self.assertEqual(replay.size, (320, 448))
        frame_manifests = feedback.content["frame_artifacts"]
        self.assertIsInstance(frame_manifests, list)
        assert isinstance(frame_manifests, list)
        frame_manifest = frame_manifests[0]
        self.assertIsInstance(frame_manifest, dict)
        assert isinstance(frame_manifest, dict)
        self.assertEqual(frame_manifest["replay_frames"], 24)
        self.assertEqual(frame_manifest["replay_frames_omitted"], 25)
        replay_timeline = frame_manifest["replay_timeline"]
        self.assertIsInstance(replay_timeline, list)
        assert isinstance(replay_timeline, list)
        self.assertEqual(
            {
                item["step_index"]
                for item in replay_timeline
                if (
                    isinstance(item, dict)
                    and item["reward"] is not None
                    and item["reward"] != 0.0
                )
            },
            reward_steps,
        )
        self.assertEqual(
            frame_manifest["replay_artifact"],
            "episode-000/replay.gif",
        )
        self.assertLess(
            artifacts["episode-000/observations.npz"].size,
            16 * 1024 * 1024,
        )
        self.assertLess(
            artifacts["episode-000/replay.gif"].size,
            3 * 1024 * 1024,
        )

        public_bytes = json.dumps(feedback.content).encode("utf-8") + b"".join(
            artifact.content for artifact in feedback.artifacts
        )
        for private_name in (
            b"environment_seed",
            b"policy_seed",
            b"scenario",
        ):
            self.assertNotIn(private_name, public_bytes)


def _frame(value: int) -> TensorValue:
    color = bytes((value % 256, (value * 3) % 256, (value * 7) % 256))
    return TensorValue(
        dtype="uint8",
        shape=(210, 160, 3),
        data=color * (210 * 160),
    )


if __name__ == "__main__":
    unittest.main()
