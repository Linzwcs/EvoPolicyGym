from __future__ import annotations

import io
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import numpy
from dm_control import suite  # type: ignore[import-untyped]
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue, TensorValue

from dm_control_benchmarks import (
    DM_CONTROL_PROFILES,
    DmControlBenchmark,
    DmControlConfig,
    baseline_program,
)


class DmControlBenchmarkTests(unittest.TestCase):
    def test_all_official_benchmarking_profiles_reset_and_step(self) -> None:
        upstream = tuple(sorted(suite.BENCHMARKING))
        configured = tuple(
            sorted(
                (DmControlConfig(profile=profile).domain, DmControlConfig(profile=profile).task)
                for profile in DM_CONTROL_PROFILES
            )
        )
        self.assertEqual(len(DM_CONTROL_PROFILES), 28)
        self.assertEqual(configured, upstream)
        for profile in DM_CONTROL_PROFILES:
            with self.subTest(profile=profile):
                benchmark = DmControlBenchmark(
                    DmControlConfig(profile=profile, max_episode_steps=2)
                )
                environment = benchmark.make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    observation = environment.reset()
                    _assert_observation(self, observation, benchmark=benchmark)
                    action_size = benchmark.spec.environment_parameters["action_size"]
                    assert type(action_size) is int
                    step = environment.step([0.0] * action_size)
                    _assert_observation(self, step.observation, benchmark=benchmark)
                    self.assertIsInstance(step.reward, float)
                    self.assertIsInstance(step.metrics, dict)
                finally:
                    environment.close()
                    environment.close()

    def test_same_seed_replays_initial_state_and_step(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=2)
        )

        def replay() -> tuple[PolicyValue, object]:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=7))
            try:
                initial = environment.reset()
                step = environment.step([0.0])
                return initial, step
            finally:
                environment.close()

        self.assertEqual(replay(), replay())

    def test_camera_feedback_renders_from_episode_worker_thread(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=2)
        )

        def run_episode_step() -> dict[str, PolicyValue]:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=17))
            try:
                environment.reset()
                step = environment.step([0.0])
                self.assertIsInstance(step.metrics, dict)
                assert isinstance(step.metrics, dict)
                return step.metrics
            finally:
                environment.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            metrics = executor.submit(run_episode_step).result(timeout=30)
        initial = metrics["feedback_video_initial_rgb"]
        result = metrics["feedback_video_rgb"]
        self.assertIsInstance(initial, TensorValue)
        self.assertIsInstance(result, TensorValue)
        assert isinstance(initial, TensorValue)
        assert isinstance(result, TensorValue)
        self.assertEqual(initial.shape, (128, 128, 3))
        self.assertEqual(result.shape, (128, 128, 3))
        self.assertIs(metrics["feedback_video_capture_failed"], False)

    def test_invalid_actions_do_not_advance_environment(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=2)
        )
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=9))
        try:
            environment.reset()
            invalid: tuple[PolicyValue, ...] = (
                cast(PolicyValue, []),
                cast(PolicyValue, [0]),
                cast(PolicyValue, [1.1]),
                cast(PolicyValue, [float("nan")]),
            )
            for action in invalid:
                with self.subTest(action=action):
                    with self.assertRaises(InvalidAction):
                        environment.step(action)
            step = environment.step([0.0])
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["step_count"], 1)
        finally:
            environment.close()

    def test_configured_horizon_is_a_truncation(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=1)
        )
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=11))
        try:
            environment.reset()
            step = environment.step([0.0])
            self.assertFalse(step.terminated)
            self.assertTrue(step.truncated)
        finally:
            environment.close()

    def test_episode_plans_are_split_specific_and_reproducible(self) -> None:
        benchmark = DmControlBenchmark()
        train = tuple(benchmark.episodes("train", seed=3, count=4))
        self.assertEqual(train, tuple(benchmark.episodes("train", seed=3, count=4)))
        self.assertNotEqual(
            train,
            tuple(benchmark.episodes("validation", seed=3, count=4)),
        )
        self.assertTrue(all(item.environment_seed <= 2**32 - 1 for item in train))

    def test_spec_records_task_shapes_and_runtime_pin(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="humanoid-walk", max_episode_steps=12)
        )
        spec = benchmark.spec
        self.assertEqual(spec.max_episode_steps, 12)
        self.assertEqual(spec.environment_parameters["action_size"], 21)
        self.assertEqual(spec.environment_parameters["domain"], "humanoid")
        self.assertEqual(spec.environment_parameters["task"], "walk")
        self.assertEqual(spec.metadata["mujoco_version"], ">=3.10.0,<3.11")
        tensor_encoding = spec.environment_parameters["tensor_encoding"]
        self.assertIsInstance(tensor_encoding, str)
        assert isinstance(tensor_encoding, str)
        self.assertIn("not indexable", tensor_encoding)
        self.assertIn("struct.iter_unpack", tensor_encoding)

    def test_feedback_reports_trace_without_private_seed(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=2)
        )
        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        transitions: list[Transition] = []
        try:
            initial = environment.reset()
            for _ in range(2):
                action: PolicyValue = [0.0]
                step = environment.step(action)
                transitions.append(Transition(action=action, step=step))
                if step.done:
                    break
        finally:
            environment.close()
        feedback = benchmark.feedback(
            (
                EpisodeRecord(
                    episode=episode,
                    policy_seed=456,
                    initial_observation=initial,
                    transitions=tuple(transitions),
                ),
            )
        )
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["episodes"], 1)
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.artifacts[0].retention, "bulk")
        self.assertNotIn(b'"environment_seed"', trace)
        self.assertNotIn(b'"policy_seed"', trace)
        self.assertNotIn(b"feedback_video_initial_rgb", trace)
        self.assertNotIn(b"feedback_video_rgb", trace)
        documents = [json.loads(line) for line in trace.splitlines()]
        self.assertEqual(documents[0]["type"], "episode")
        self.assertEqual(documents[1]["type"], "transition")
        self.assertEqual(feedback.content["video_episodes"], 1)
        self.assertEqual(feedback.content["video_capture_unavailable_episodes"], 0)
        self.assertEqual(feedback.content["video_frame_shape"], [128, 128, 3])
        self.assertEqual(feedback.content["rendered_frame_evidence_episodes"], 1)
        self.assertEqual(len(feedback.artifacts), 3)
        evidence = feedback.artifacts[1]
        self.assertEqual(evidence.name, "episode-000/rendered-frames.npz")
        self.assertEqual(evidence.media_type, "application/x-npz")
        self.assertEqual(evidence.retention, "bulk")
        with numpy.load(io.BytesIO(evidence.read_bytes()), allow_pickle=False) as archive:
            self.assertEqual(archive["frames"].shape, (3, 128, 128, 3))
            self.assertEqual(archive["step_indices"].tolist(), [-1, 1, 2])
            self.assertEqual(
                archive["reward_present"].tolist(),
                [False, True, True],
            )
            metrics = transitions[0].step.metrics
            assert isinstance(metrics, dict)
            initial_frame = metrics["feedback_video_initial_rgb"]
            assert isinstance(initial_frame, TensorValue)
            self.assertEqual(archive["frames"][0].tobytes(), initial_frame.data)
        replay = feedback.artifacts[2]
        self.assertEqual(replay.name, "episode-000/default-camera.gif")
        self.assertEqual(replay.media_type, "image/gif")
        self.assertEqual(replay.retention, "bulk")
        self.assertTrue(replay.read_bytes().startswith(b"GIF89a"))

    def test_feedback_saves_frame_evidence_and_preview_per_episode(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=1)
        )
        records: list[EpisodeRecord] = []
        for episode_index in range(3):
            episode = EpisodeSpec(environment_seed=100 + episode_index)
            environment = benchmark.make_environment(episode)
            try:
                initial = environment.reset()
                action: PolicyValue = [0.0]
                step = environment.step(action)
            finally:
                environment.close()
            records.append(
                EpisodeRecord(
                    episode=episode,
                    policy_seed=200 + episode_index,
                    initial_observation=initial,
                    transitions=(Transition(action=action, step=step),),
                )
            )

        feedback = benchmark.feedback(tuple(records))

        self.assertEqual(
            [artifact.name for artifact in feedback.artifacts],
            [
                "trace.jsonl",
                "episode-000/rendered-frames.npz",
                "episode-000/default-camera.gif",
                "episode-001/rendered-frames.npz",
                "episode-001/default-camera.gif",
                "episode-002/rendered-frames.npz",
                "episode-002/default-camera.gif",
            ],
        )
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["video_episodes"], 3)
        self.assertEqual(feedback.content["video_episode_results"], 3)
        self.assertEqual(feedback.content["video_episodes_without_gif"], 0)
        self.assertEqual(feedback.content["rendered_frame_evidence_episodes"], 3)
        manifests = feedback.content["video_artifacts"]
        self.assertIsInstance(manifests, list)
        assert isinstance(manifests, list)
        self.assertEqual(
            [manifest["status"] for manifest in manifests if isinstance(manifest, dict)],
            ["available", "available", "available"],
        )
        first_manifest = manifests[0]
        assert isinstance(first_manifest, dict)
        self.assertEqual(
            first_manifest["evidence_artifact"],
            "episode-000/rendered-frames.npz",
        )
        self.assertEqual(
            first_manifest["preview_artifact"],
            "episode-000/default-camera.gif",
        )

    def test_zero_step_failure_has_explicit_unavailable_video_result(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=1)
        )
        episode = EpisodeSpec(environment_seed=301)
        feedback = benchmark.feedback(
            (
                EpisodeRecord(
                    episode=episode,
                    policy_seed=401,
                    initial_observation={},
                    transitions=(),
                    policy_failure="invalid_action",
                ),
            )
        )
        self.assertEqual([item.name for item in feedback.artifacts], ["trace.jsonl"])
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["video_episode_results"], 1)
        self.assertEqual(feedback.content["video_episodes_without_gif"], 1)
        manifests = feedback.content["video_artifacts"]
        assert isinstance(manifests, list)
        manifest = manifests[0]
        self.assertIsInstance(manifest, dict)
        assert isinstance(manifest, dict)
        self.assertEqual(manifest["status"], "unavailable")
        self.assertEqual(manifest["reason"], "no_recorded_frames")
        self.assertIsNone(manifest["frames_artifact"])
        self.assertIsNone(manifest["evidence_artifact"])
        self.assertIsNone(manifest["preview_artifact"])

    def test_conformance_and_packaged_baseline(self) -> None:
        benchmark = DmControlBenchmark(
            DmControlConfig(profile="cartpole-swingup", max_episode_steps=2)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=5),
                    actions=([0.0], [0.0]),
                ),
            ),
        )
        report.raise_for_errors()
        self.assertIn("policy.py", baseline_program().files)


def _assert_observation(
    testcase: unittest.TestCase,
    observation: PolicyValue,
    *,
    benchmark: DmControlBenchmark,
) -> None:
    testcase.assertIsInstance(observation, dict)
    assert isinstance(observation, dict)
    observation_space = benchmark.spec.observation_space
    assert isinstance(observation_space, dict)
    fields = observation_space["fields"]
    assert isinstance(fields, dict)
    testcase.assertEqual(set(observation), set(fields))
    for name, value in observation.items():
        testcase.assertIsInstance(value, TensorValue)
        assert isinstance(value, TensorValue)
        field_spec = fields[name]
        assert isinstance(field_spec, dict)
        shape = field_spec["shape"]
        assert isinstance(shape, list)
        testcase.assertEqual(list(value.shape), shape)


if __name__ == "__main__":
    unittest.main()
