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
from evopolicygym.policy import PolicyValue, TensorValue
from PIL import Image, ImageSequence

from vizdoom_benchmarks import (
    VIZDOOM_PROFILES,
    ViZDoomBenchmark,
    ViZDoomConfig,
    baseline_program,
)


class ViZDoomBenchmarkTests(unittest.TestCase):
    def test_all_bundled_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(VIZDOOM_PROFILES), 12)
        for profile in VIZDOOM_PROFILES:
            with self.subTest(profile=profile):
                config = ViZDoomConfig(profile=profile)
                self.assertEqual(
                    len(config.action_meanings),
                    config.action_size,
                )
                self.assertEqual(
                    len(config.game_variable_names),
                    config.game_variables,
                )
                environment = ViZDoomBenchmark(
                    config
                ).make_environment(EpisodeSpec(environment_seed=123))
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    action: PolicyValue = (
                        {
                            "binary": 0,
                            "continuous": [0.0, 0.0, 0.0],
                        }
                        if config.hybrid_action
                        else 0
                    )
                    step = environment.step(action)
                    self.assertIsInstance(step.reward, float)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_environment_identity(self) -> None:
        basic = ViZDoomBenchmark()
        audio = ViZDoomBenchmark(ViZDoomConfig(profile="basic-audio"))
        self.assertNotEqual(
            basic.spec.environment_digest,
            audio.spec.environment_digest,
        )
        self.assertEqual(audio.spec.max_episode_steps, 300)

    def test_invalid_actions_are_rejected(self) -> None:
        environment = ViZDoomBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

        environment = ViZDoomBenchmark(
            ViZDoomConfig(profile="deathmatch")
        ).make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(
                    {"binary": 0, "continuous": [0, 0, 0]}
                )
        finally:
            environment.close()

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            ViZDoomBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    (0,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_feedback_publishes_bounded_multimodal_visual_trace(self) -> None:
        config = ViZDoomConfig(profile="basic")
        reward_steps = {20, 50, 80, 110, 140, 170}
        transitions = tuple(
            Transition(
                action=step_index % 4,
                step=Step(
                    observation=_observation(
                        step_index + 1,
                        game_variables=(50.0 - step_index // 10,),
                    ),
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
            initial_observation=_observation(
                0,
                game_variables=(50.0,),
            ),
            transitions=transitions,
        )

        feedback = ViZDoomBenchmark(config).feedback((record,))

        self.assertEqual(feedback.score, 6.0)
        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["traced_steps"], 32)
        self.assertEqual(feedback.content["trace_steps_omitted"], 168)
        summaries = feedback.content["episode_summaries"]
        self.assertIsInstance(summaries, list)
        assert isinstance(summaries, list)
        summary = summaries[0]
        self.assertIsInstance(summary, dict)
        assert isinstance(summary, dict)
        self.assertEqual(summary["positive_reward_events"], 6)
        self.assertEqual(summary["action_counts"], {"0": 50, "1": 50, "2": 50, "3": 50})
        self.assertEqual(
            summary["game_variable_ranges"]["AMMO2"],
            {
                "initial": 50.0,
                "final": 31.0,
                "minimum": 31.0,
                "maximum": 50.0,
            },
        )

        artifacts = {
            artifact.name: artifact for artifact in feedback.artifacts
        }
        self.assertEqual(
            set(artifacts),
            {
                "trace.jsonl",
                "episode-000/observations.npz",
                "episode-000/contact-sheet.png",
                "episode-000/replay.gif",
            },
        )
        trace = tuple(
            json.loads(line)
            for line in artifacts["trace.jsonl"].content.splitlines()
        )
        transition_trace = tuple(
            document
            for document in trace
            if document["type"] == "transition"
        )
        self.assertEqual(len(transition_trace), 32)
        self.assertTrue(reward_steps.issubset({
            item["step_index"] for item in transition_trace
        }))
        self.assertEqual(transition_trace[0]["action_meaning"], "noop")
        self.assertEqual(transition_trace[1]["action_meaning"], "attack")
        self.assertEqual(
            transition_trace[0]["decision_observation"]["screen_array"],
            "decision_screens",
        )
        self.assertEqual(
            transition_trace[0]["result_observation"]["semantics"]["game_variables"],
            {"AMMO2": 50.0},
        )

        with numpy.load(
            io.BytesIO(artifacts["episode-000/observations.npz"].content),
            allow_pickle=False,
        ) as observations:
            self.assertEqual(
                observations["initial_screen"].shape,
                (240, 320, 3),
            )
            self.assertEqual(
                observations["decision_screens"].shape,
                (32, 240, 320, 3),
            )
            self.assertEqual(
                observations["result_screens"].shape,
                (32, 240, 320, 3),
            )
            self.assertEqual(
                observations["decision_game_variables"].shape,
                (32, 1),
            )
            self.assertIn(50, observations["step_indices"].tolist())
            numpy.testing.assert_array_equal(
                observations["decision_screens"][0],
                observations["initial_screen"],
            )

        self.assertTrue(
            artifacts["episode-000/contact-sheet.png"].content.startswith(
                b"\x89PNG\r\n\x1a\n"
            )
        )
        replay_content = artifacts["episode-000/replay.gif"].content
        self.assertTrue(replay_content.startswith(b"GIF89a"))
        with Image.open(io.BytesIO(replay_content)) as replay:
            self.assertEqual(replay.format, "GIF")
            self.assertEqual(
                sum(1 for _ in ImageSequence.Iterator(replay)),
                24,
            )
            self.assertEqual(replay.size, (640, 508))
        manifest = feedback.content["observation_artifacts"][0]
        self.assertEqual(manifest["stored_channels"], ["screen", "gamevariables"])
        self.assertEqual(manifest["replay_frames"], 24)
        self.assertEqual(manifest["replay_frames_omitted"], 9)
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

    def test_feedback_preserves_audio_and_notifications(self) -> None:
        audio_config = ViZDoomConfig(profile="basic-audio")
        audio_record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=1),
            policy_seed=2,
            initial_observation=_observation(
                0,
                game_variables=(50.0,),
                audio_value=3,
            ),
            transitions=(
                Transition(
                    action=1,
                    step=Step(
                        observation=_observation(
                            1,
                            game_variables=(49.0,),
                            audio_value=7,
                        ),
                        reward=1.0,
                        terminated=True,
                    ),
                ),
            ),
        )
        audio_feedback = ViZDoomBenchmark(audio_config).feedback((audio_record,))
        audio_artifact = {
            artifact.name: artifact for artifact in audio_feedback.artifacts
        }["episode-000/observations.npz"]
        with numpy.load(io.BytesIO(audio_artifact.content), allow_pickle=False) as arrays:
            self.assertEqual(arrays["initial_audio"].shape, (1260, 2))
            self.assertEqual(arrays["result_audio"].shape, (1, 1260, 2))
            self.assertEqual(int(arrays["result_audio"][0, 0, 0]), 7)

        notification_config = ViZDoomConfig(profile="basic-notifications")
        notification_record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=1),
            policy_seed=2,
            initial_observation=_observation(
                0,
                game_variables=(50.0,),
                notifications="",
            ),
            transitions=(
                Transition(
                    action=1,
                    step=Step(
                        observation=_observation(
                            1,
                            game_variables=(49.0,),
                            notifications="picked up ammo",
                        ),
                        reward=1.0,
                        terminated=True,
                    ),
                ),
            ),
        )
        notification_feedback = ViZDoomBenchmark(notification_config).feedback(
            (notification_record,)
        )
        notification_trace = tuple(
            json.loads(line)
            for line in notification_feedback.artifacts[0].content.splitlines()
        )
        self.assertEqual(
            notification_trace[1]["result_observation"]["notifications"],
            "picked up ammo",
        )
        self.assertEqual(
            notification_feedback.content["episode_summaries"][0]["notification_events"],
            1,
        )


def _observation(
    value: int,
    *,
    game_variables: tuple[float, ...] = (),
    audio_value: int | None = None,
    notifications: str | None = None,
) -> dict[str, PolicyValue]:
    color = bytes((value % 256, (value * 3) % 256, (value * 7) % 256))
    observation: dict[str, PolicyValue] = {
        "screen": TensorValue(
            dtype="uint8",
            shape=(240, 320, 3),
            data=color * (240 * 320),
        )
    }
    if game_variables:
        variables = numpy.asarray(game_variables, dtype=numpy.float32)
        observation["gamevariables"] = TensorValue(
            dtype="float32",
            shape=variables.shape,
            data=variables.tobytes(order="C"),
        )
    if audio_value is not None:
        audio = numpy.full((1260, 2), audio_value, dtype=numpy.int16)
        observation["audio"] = TensorValue(
            dtype="int16",
            shape=audio.shape,
            data=audio.tobytes(order="C"),
        )
    if notifications is not None:
        observation["notifications"] = notifications
    return observation


if __name__ == "__main__":
    unittest.main()
