from __future__ import annotations

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue, TensorValue

from robosuite_benchmarks import (
    ROBOSUITE_PROFILES,
    RobosuiteBenchmark,
    RobosuiteConfig,
    baseline_program,
)


class RobosuiteBenchmarkTests(unittest.TestCase):
    def test_all_nineteen_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(ROBOSUITE_PROFILES), 19)
        for profile in ROBOSUITE_PROFILES:
            with self.subTest(profile=profile):
                benchmark = RobosuiteBenchmark(
                    RobosuiteConfig(profile=profile, max_episode_steps=2)
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
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=2)
        )

        def replay() -> tuple[PolicyValue, object]:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=7))
            try:
                initial = environment.reset()
                step = environment.step([0.0] * 7)
                return initial, step
            finally:
                environment.close()

        self.assertEqual(replay(), replay())

    def test_camera_feedback_renders_from_episode_worker_thread(self) -> None:
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=2)
        )

        def run_episode_step() -> dict[str, PolicyValue]:
            environment = benchmark.make_environment(EpisodeSpec(environment_seed=17))
            try:
                environment.reset()
                step = environment.step([0.0] * 7)
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
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=2)
        )
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=9))
        try:
            environment.reset()
            short_action = cast(PolicyValue, [0.0] * 6)
            integer_action = cast(PolicyValue, [0] * 7)
            out_of_bounds_action = cast(PolicyValue, [0.0] * 6 + [1.1])
            non_finite_action = cast(
                PolicyValue,
                [0.0] * 6 + [float("nan")],
            )
            invalid: tuple[PolicyValue, ...] = (
                short_action,
                integer_action,
                out_of_bounds_action,
                non_finite_action,
            )
            for action in invalid:
                with self.subTest(action=action):
                    with self.assertRaises(InvalidAction):
                        environment.step(action)
            step = environment.step([0.0] * 7)
            assert isinstance(step.metrics, dict)
            self.assertEqual(step.metrics["step_count"], 1)
        finally:
            environment.close()

    def test_episode_plans_are_split_specific_and_reproducible(self) -> None:
        benchmark = RobosuiteBenchmark()
        train = tuple(benchmark.episodes("train", seed=3, count=4))
        self.assertEqual(train, tuple(benchmark.episodes("train", seed=3, count=4)))
        self.assertNotEqual(
            train,
            tuple(benchmark.episodes("validation", seed=3, count=4)),
        )

    def test_spec_records_profile_shapes_and_runtime_pin(self) -> None:
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="two-arm-peg-in-hole", max_episode_steps=12)
        )
        spec = benchmark.spec
        self.assertEqual(spec.max_episode_steps, 12)
        self.assertEqual(spec.environment_parameters["action_size"], 12)
        self.assertEqual(spec.environment_parameters["robot_count"], 2)
        self.assertEqual(spec.metadata["mujoco_version"], ">=3.3.0,<3.4")
        self.assertEqual(spec.environment_parameters["controller"], "BASIC/OSC_POSE")

    def test_feedback_reports_trace_without_private_seed(self) -> None:
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=2)
        )
        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        transitions: list[Transition] = []
        try:
            initial = environment.reset()
            for _ in range(2):
                action: PolicyValue = [0.0] * 7
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
        self.assertEqual(len(feedback.artifacts), 2)
        replay = feedback.artifacts[1]
        self.assertEqual(replay.name, "episode-000/agentview.gif")
        self.assertEqual(replay.media_type, "image/gif")
        self.assertEqual(replay.retention, "bulk")
        self.assertTrue(replay.read_bytes().startswith(b"GIF89a"))

    def test_feedback_saves_one_gif_for_every_completed_episode(self) -> None:
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=1)
        )
        records: list[EpisodeRecord] = []
        for episode_index in range(3):
            episode = EpisodeSpec(environment_seed=100 + episode_index)
            environment = benchmark.make_environment(episode)
            try:
                initial = environment.reset()
                action: PolicyValue = [0.0] * 7
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
                "episode-000/agentview.gif",
                "episode-001/agentview.gif",
                "episode-002/agentview.gif",
            ],
        )
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["video_episodes"], 3)
        self.assertEqual(feedback.content["video_episode_results"], 3)
        self.assertEqual(feedback.content["video_episodes_without_gif"], 0)
        manifests = feedback.content["video_artifacts"]
        self.assertIsInstance(manifests, list)
        assert isinstance(manifests, list)
        self.assertEqual(
            [manifest["status"] for manifest in manifests if isinstance(manifest, dict)],
            ["available", "available", "available"],
        )

    def test_zero_step_failure_has_explicit_unavailable_video_result(self) -> None:
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=1)
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

    def test_conformance_and_packaged_baseline(self) -> None:
        benchmark = RobosuiteBenchmark(
            RobosuiteConfig(profile="lift", max_episode_steps=2)
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(environment_seed=5),
                    actions=([0.0] * 7, [0.0] * 7),
                ),
            ),
        )
        report.raise_for_errors()
        self.assertIn("policy.py", baseline_program().files)


def _assert_observation(
    testcase: unittest.TestCase,
    observation: PolicyValue,
    *,
    benchmark: RobosuiteBenchmark,
) -> None:
    testcase.assertIsInstance(observation, dict)
    assert isinstance(observation, dict)
    testcase.assertEqual(set(observation), {"proprioception", "objects"})
    proprioception = observation["proprioception"]
    objects = observation["objects"]
    testcase.assertIsInstance(proprioception, TensorValue)
    testcase.assertIsInstance(objects, TensorValue)
    assert isinstance(proprioception, TensorValue)
    assert isinstance(objects, TensorValue)
    observation_space = benchmark.spec.observation_space
    assert isinstance(observation_space, dict)
    fields = observation_space["fields"]
    assert isinstance(fields, dict)
    proprioception_spec = fields["proprioception"]
    object_spec = fields["objects"]
    assert isinstance(proprioception_spec, dict)
    assert isinstance(object_spec, dict)
    proprioception_shape = proprioception_spec["shape"]
    object_shape = object_spec["shape"]
    assert isinstance(proprioception_shape, list)
    assert isinstance(object_shape, list)
    testcase.assertEqual(list(proprioception.shape), proprioception_shape)
    testcase.assertEqual(list(objects.shape), object_shape)


if __name__ == "__main__":
    unittest.main()
