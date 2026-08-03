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
from numpy.typing import NDArray

from highway_benchmarks import (
    HIGHWAY_PROFILES,
    HighwayBenchmark,
    HighwayConfig,
    baseline_program,
)


class HighwayBenchmarkTests(unittest.TestCase):
    def test_all_profiles_reset_and_take_one_strict_action(self) -> None:
        self.assertEqual(len(HIGHWAY_PROFILES), 10)
        semantic_kinds = {
            "highway": "kinematics",
            "merge": "kinematics",
            "roundabout": "kinematics",
            "intersection": "kinematics",
            "two-way": "time_to_collision",
            "exit": "kinematics",
            "u-turn": "time_to_collision",
            "parking": "goal_kinematics",
            "racetrack": "occupancy_grid",
            "lane-keeping": "vehicle_attributes",
        }
        for profile in HIGHWAY_PROFILES:
            with self.subTest(profile=profile):
                config = HighwayConfig(profile=profile)
                benchmark = HighwayBenchmark(config)
                episode = EpisodeSpec(environment_seed=123)
                environment = benchmark.make_environment(
                    episode
                )
                try:
                    observation = environment.reset()
                    self.assertTrue(
                        type(observation) in {TensorValue, dict}
                    )
                    action: PolicyValue = (
                        [0.0] * config.action_size
                        if config.continuous
                        else 1
                    )
                    step = environment.step(action)
                    self.assertIsInstance(step.reward, float)
                    record = EpisodeRecord(
                        episode=episode,
                        policy_seed=7,
                        initial_observation=observation,
                        transitions=(Transition(action=action, step=step),),
                    )
                    feedback = benchmark.feedback((record,))
                    self.assertEqual(len(feedback.artifacts), 2)
                    artifacts = {
                        artifact.name: artifact
                        for artifact in feedback.artifacts
                    }
                    trace = tuple(
                        json.loads(line)
                        for line in artifacts["trace.jsonl"].content.splitlines()
                    )
                    self.assertEqual(
                        trace[0]["initial_observation"]["semantics"]["kind"],
                        semantic_kinds[profile],
                    )
                    self.assertEqual(
                        trace[1]["result_observation"]["semantics"]["kind"],
                        semantic_kinds[profile],
                    )
                    observation_artifact = artifacts[
                        "episode-000/observations.npz"
                    ]
                    with numpy.load(
                        io.BytesIO(observation_artifact.content),
                        allow_pickle=False,
                    ) as arrays:
                        self.assertEqual(arrays["step_indices"].tolist(), [0])
                    public_bytes = json.dumps(feedback.content).encode(
                        "utf-8"
                    ) + b"".join(
                        artifact.content for artifact in feedback.artifacts
                    )
                    self.assertNotIn(b"environment_seed", public_bytes)
                    self.assertNotIn(b"policy_seed", public_bytes)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_public_identity(self) -> None:
        highway = HighwayBenchmark()
        parking = HighwayBenchmark(HighwayConfig(profile="parking"))
        self.assertNotEqual(
            highway.spec.environment_digest,
            parking.spec.environment_digest,
        )
        self.assertEqual(
            parking.spec.environment_parameters["profile"],
            "parking",
        )
        self.assertEqual(parking.spec.max_episode_steps, 500)
        self.assertEqual(
            highway.spec.action_space["meaning"]["0"],
            "lane_left",
        )
        intersection = HighwayBenchmark(
            HighwayConfig(profile="intersection")
        )
        self.assertEqual(
            intersection.spec.action_space["meaning"],
            {"0": "slower", "1": "idle", "2": "faster"},
        )
        self.assertEqual(
            parking.spec.observation_space["fields"]["achieved_goal"][
                "features"
            ],
            ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        )

    def test_invalid_profile_and_actions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HighwayConfig(profile="unknown")
        with self.assertRaises(TypeError):
            HighwayConfig(profile=1)  # type: ignore[arg-type]

        environment = HighwayBenchmark().make_environment(
            EpisodeSpec(environment_seed=1)
        )
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

        environment = HighwayBenchmark(
            HighwayConfig(profile="parking")
        ).make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([0, 0])
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_profile(self) -> None:
        with self.assertRaises(ValueError):
            HighwayBenchmark().make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "parking"},
                )
            )

    def test_baseline_is_packaged(self) -> None:
        program = baseline_program()
        self.assertIn("policy.py", program.files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            HighwayBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    (1,),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_parking_feedback_is_bounded_and_preserves_goal_state(self) -> None:
        config = HighwayConfig(profile="parking")
        transitions = tuple(
            Transition(
                action=[0.1, 0.0],
                step=Step(
                    observation=_parking_observation(step_index + 1),
                    reward=-1.0 + step_index / 100.0,
                    terminated=False,
                    truncated=step_index == 99,
                    metrics={
                        "crashed": False,
                        "is_success": step_index == 70,
                        "speed": float(step_index) / 10.0,
                    },
                ),
            )
            for step_index in range(100)
        )
        record = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_parking_observation(0),
            transitions=transitions,
        )

        feedback = HighwayBenchmark(config).feedback((record,))

        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["traced_steps"], 48)
        self.assertEqual(feedback.content["trace_steps_omitted"], 52)
        self.assertEqual(feedback.content["successful_episodes"], 1)
        summaries = feedback.content["episode_summaries"]
        self.assertIsInstance(summaries, list)
        assert isinstance(summaries, list)
        summary = summaries[0]
        self.assertIsInstance(summary, dict)
        assert isinstance(summary, dict)
        self.assertEqual(summary["success_step"], 70)
        controls = summary["control_summary"]
        self.assertIsInstance(controls, dict)
        assert isinstance(controls, dict)
        acceleration = controls["acceleration"]
        steering = controls["steering"]
        self.assertIsInstance(acceleration, dict)
        self.assertIsInstance(steering, dict)
        assert isinstance(acceleration, dict)
        assert isinstance(steering, dict)
        self.assertAlmostEqual(acceleration["mean"], 0.1)
        self.assertAlmostEqual(acceleration["mean_absolute"], 0.1)
        self.assertEqual(
            steering,
            {"mean": 0.0, "mean_absolute": 0.0},
        )
        self.assertEqual(
            summary["speed_summary"],
            {"minimum": 0.0, "mean": 4.95, "maximum": 9.9, "final": 9.9},
        )

        artifacts = {
            artifact.name: artifact for artifact in feedback.artifacts
        }
        trace = tuple(
            json.loads(line)
            for line in artifacts["trace.jsonl"].content.splitlines()
        )
        transition_trace = tuple(
            document
            for document in trace
            if document["type"] == "transition"
        )
        self.assertEqual(len(transition_trace), 48)
        success_trace = next(
            document
            for document in transition_trace
            if document["step_index"] == 70
        )
        self.assertTrue(success_trace["event"])
        self.assertEqual(
            success_trace["action_meaning"],
            "acceleration=0.1,steering=0",
        )
        semantics = success_trace["result_observation"]["semantics"]
        self.assertEqual(semantics["kind"], "goal_kinematics")
        self.assertIn("position_error", semantics)

        with numpy.load(
            io.BytesIO(artifacts["episode-000/observations.npz"].content),
            allow_pickle=False,
        ) as arrays:
            self.assertEqual(arrays["initial__achieved_goal"].shape, (6,))
            self.assertEqual(
                arrays["decision__achieved_goal"].shape,
                (48, 6),
            )
            self.assertEqual(
                arrays["result__desired_goal"].shape,
                (48, 6),
            )
            self.assertIn(70, arrays["step_indices"].tolist())


def _parking_observation(step_index: int) -> dict[str, PolicyValue]:
    progress = min(1.0, step_index / 100.0)
    achieved = numpy.asarray(
        [0.18 * progress, 0.14 * progress, 0.0, 0.0, 0.0, -1.0],
        dtype=numpy.float64,
    )
    desired = numpy.asarray(
        [0.18, 0.14, 0.0, 0.0, 0.0, -1.0],
        dtype=numpy.float64,
    )
    return {
        "observation": _tensor(achieved),
        "achieved_goal": _tensor(achieved),
        "desired_goal": _tensor(desired),
    }


def _tensor(array: NDArray[numpy.float64]) -> TensorValue:
    return TensorValue(
        dtype="float64",
        shape=tuple(int(size) for size in array.shape),
        data=array.tobytes(order="C"),
    )


if __name__ == "__main__":
    unittest.main()
