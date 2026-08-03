from __future__ import annotations

import json
import struct
import unittest

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

from metaworld_benchmarks import (
    METAWORLD_MT1_PROFILES,
    MetaWorldBenchmark,
    MetaWorldConfig,
    baseline_program,
)


class MetaWorldBenchmarkTests(unittest.TestCase):
    def test_all_fifty_mt1_profiles_reset_and_step(self) -> None:
        self.assertEqual(len(METAWORLD_MT1_PROFILES), 50)
        for profile in METAWORLD_MT1_PROFILES:
            with self.subTest(profile=profile):
                benchmark = MetaWorldBenchmark(MetaWorldConfig(profile=profile))
                environment = benchmark.make_environment(EpisodeSpec(environment_seed=123))
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, TensorValue)
                    step = environment.step([0.0, 0.0, 0.0, 0.0])
                    self.assertIsInstance(step.reward, float)
                    self.assertIsInstance(step.metrics, dict)
                    metrics = _step_metrics(step)
                    self.assertEqual(metrics["task_name"], profile)
                    self.assertIn("best_reward", metrics)
                    self.assertIn("obj_to_target_improvement_this_step", metrics)
                    self.assertIn("action_l2_norm", metrics)
                    self.assertIn("state_motion_l2", metrics)
                finally:
                    environment.close()
                    environment.close()

    def test_collection_profiles_have_public_one_hot_tasks(self) -> None:
        configs = (
            MetaWorldConfig(profile="mt10"),
            MetaWorldConfig(profile="mt50"),
            MetaWorldConfig(
                profile="custom",
                custom_tasks=("reach-v3", "push-v3", "door-open-v3"),
            ),
        )
        for config in configs:
            with self.subTest(profile=config.profile):
                benchmark = MetaWorldBenchmark(config)
                episode = benchmark.episodes("train", seed=7, count=1)[0]
                environment = benchmark.make_environment(episode)
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    assert isinstance(observation, dict)
                    self.assertEqual(set(observation), {"state", "task"})
                    task = observation["task"]
                    self.assertIsInstance(task, TensorValue)
                    assert isinstance(task, TensorValue)
                    self.assertEqual(task.dtype, "bool")
                    self.assertEqual(sum(task.data), 1)
                    self.assertEqual(task.shape, (len(config.task_names),))
                finally:
                    environment.close()

    def test_plans_are_reproducible_and_balanced(self) -> None:
        benchmark = MetaWorldBenchmark(MetaWorldConfig(profile="mt10"))
        first = tuple(benchmark.episodes("train", seed=7, count=20))
        repeated = tuple(benchmark.episodes("train", seed=7, count=20))
        self.assertEqual(first, repeated)
        indexes = [
            episode.scenario["task_index"] for episode in first if type(episode.scenario) is dict
        ]
        self.assertEqual(len(set(indexes)), 10)
        self.assertTrue(all(indexes.count(index) == 2 for index in set(indexes)))

    def test_invalid_configuration_and_action_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetaWorldConfig(profile="unknown")
        with self.assertRaises(ValueError):
            MetaWorldConfig(profile="custom")
        with self.assertRaises(ValueError):
            MetaWorldConfig(
                profile="custom",
                custom_tasks=("reach-v3", "reach-v3"),
            )
        with self.assertRaises(TypeError):
            MetaWorldConfig(custom_tasks=["reach-v3"])  # type: ignore[arg-type]

        environment = MetaWorldBenchmark().make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([0, 0, 0, 0])
        finally:
            environment.close()

    def test_spec_documents_dense_feedback_and_nonterminating_success(self) -> None:
        spec = MetaWorldBenchmark().spec
        reward_mode = spec.metadata["reward_mode"]
        success_persistence = spec.metadata["success_persistence"]
        action_handling = spec.environment_parameters["action_handling"]
        horizon = spec.environment_parameters["horizon"]
        feedback_diagnostics = spec.environment_parameters["feedback_diagnostics"]
        assert isinstance(reward_mode, str)
        assert isinstance(success_persistence, str)
        assert isinstance(action_handling, str)
        assert isinstance(horizon, str)
        assert isinstance(feedback_diagnostics, str)

        self.assertIn("dense shaped", reward_mode)
        self.assertIn("does not terminate", success_persistence)
        self.assertIn("rejected", action_handling)
        self.assertIn("500 steps", horizon)
        self.assertIn("obj_to_target", feedback_diagnostics)

    def test_collection_requires_host_task_scenario(self) -> None:
        with self.assertRaises(ValueError):
            MetaWorldBenchmark(MetaWorldConfig(profile="mt10")).make_environment(
                EpisodeSpec(environment_seed=1)
            )

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_real_reach_success_and_later_loss_are_distinct(self) -> None:
        environment = MetaWorldBenchmark().make_environment(EpisodeSpec(environment_seed=5))
        try:
            observation = environment.reset()
            self.assertIsInstance(observation, TensorValue)
            assert isinstance(observation, TensorValue)
            steps = []
            for _ in range(26):
                values = _tensor_values(observation)
                action: PolicyValue = [
                    *(
                        max(
                            -1.0,
                            min(1.0, 10.0 * (values[-3 + index] - values[index])),
                        )
                        for index in range(3)
                    ),
                    0.0,
                ]
                step = environment.step(action)
                steps.append(step)
                observation = step.observation
                assert isinstance(observation, TensorValue)

            success = steps[-1]
            success_metrics = _step_metrics(success)
            self.assertEqual(success_metrics["success"], True)
            self.assertEqual(success_metrics["success_ever"], True)
            self.assertEqual(success_metrics["first_success_step"], 26)
            self.assertEqual(success_metrics["success_first_reached_this_step"], True)
            self.assertEqual(success.reward, 10.0)
            self.assertEqual(success_metrics["best_reward"], 10.0)
            self.assertLess(_number_metric(success_metrics, "best_obj_to_target"), 0.05)

            still_success = environment.step([-1.0, -1.0, -1.0, 0.0])
            lost = environment.step([-1.0, -1.0, -1.0, 0.0])
            still_success_metrics = _step_metrics(still_success)
            lost_metrics = _step_metrics(lost)
            self.assertEqual(still_success_metrics["success"], True)
            self.assertEqual(lost_metrics["success"], False)
            self.assertEqual(lost_metrics["success_ever"], True)
            self.assertEqual(lost_metrics["success_lost_this_step"], True)
            self.assertEqual(lost_metrics["success_lost_count"], 1)
            self.assertEqual(lost_metrics["task_stage"], "success_lost")
        finally:
            environment.close()

    def test_feedback_traces_public_state_with_explicit_sampling(self) -> None:
        benchmark = MetaWorldBenchmark()
        episode = EpisodeSpec(environment_seed=123)
        environment = benchmark.make_environment(episode)
        transitions: list[Transition] = []
        try:
            initial = environment.reset()
            for _ in range(benchmark.spec.max_episode_steps):
                action: PolicyValue = [0.0, 0.0, 0.0, 0.0]
                step = environment.step(action)
                transitions.append(Transition(action=action, step=step))
                if step.done:
                    break
        finally:
            environment.close()
        record = EpisodeRecord(
            episode=episode,
            policy_seed=456,
            initial_observation=initial,
            transitions=tuple(transitions),
        )

        feedback = benchmark.feedback((record,))

        self.assertIsInstance(feedback.content, dict)
        assert isinstance(feedback.content, dict)
        self.assertEqual(feedback.content["traced_transitions"], 160)
        self.assertEqual(feedback.content["trace_transitions_omitted"], 340)
        documents = [json.loads(line) for line in feedback.artifacts[0].read_bytes().splitlines()]
        self.assertEqual(documents[0]["traced_steps"], 160)
        self.assertEqual(documents[0]["omitted_steps"], 340)
        transitions_json = [document for document in documents if document["type"] == "transition"]
        self.assertEqual(transitions_json[0]["step_index"], 0)
        self.assertEqual(transitions_json[127]["step_index"], 127)
        self.assertEqual(transitions_json[128]["step_index"], 468)
        self.assertEqual(transitions_json[-1]["step_index"], 499)
        self.assertEqual(
            transitions_json[0]["observation"]["$type"],
            "tensor",
        )
        self.assertIn("next_observation", transitions_json[0])
        self.assertEqual(feedback.content["mean_zero_action_fraction"], 1.0)
        self.assertEqual(feedback.content["task_episode_counts"], {"reach-v3": 1})
        self.assertEqual(feedback.content["task_success_rates"], {"reach-v3": 0.0})
        self.assertGreater(_number_metric(feedback.content, "mean_best_reward"), 0.0)
        self.assertGreaterEqual(_number_metric(feedback.content, "mean_state_motion_l2"), 0.0)
        self.assertIn("best_in_place_reward", transitions_json[0]["metrics"])
        self.assertEqual(
            transitions_json[-1]["metrics"]["terminal_reason"],
            "time_limit",
        )

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            MetaWorldBenchmark(),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    ([0.0, 0.0, 0.0, 0.0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)


def _tensor_values(value: TensorValue) -> tuple[float, ...]:
    return tuple(item[0] for item in struct.iter_unpack("<d", value.data))


def _step_metrics(step: Step) -> dict[str, PolicyValue]:
    metrics = step.metrics
    assert isinstance(metrics, dict)
    return metrics


def _number_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


if __name__ == "__main__":
    unittest.main()
