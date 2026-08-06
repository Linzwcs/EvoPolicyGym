from __future__ import annotations

import json
import unittest

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import EpisodeRecord, EpisodeSpec, InvalidAction, Step
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_babyai import BabyAIBenchmark, BabyAIConfig, baseline_program

_PROFILES = (
    "GoToRedBallGrey",
    "GoToRedBall",
    "GoToRedBallNoDists",
    "GoToObj",
    "GoToLocal",
    "GoTo",
    "GoToImpUnlock",
    "GoToSeq",
    "GoToRedBlueBall",
    "GoToDoor",
    "GoToObjDoor",
    "Open",
    "OpenRedDoor",
    "OpenDoor",
    "OpenTwoDoors",
    "OpenDoorsOrder",
    "Pickup",
    "UnblockPickup",
    "PickupLoc",
    "PickupDist",
    "PickupAbove",
    "PutNextLocal",
    "PutNext",
    "Unlock",
    "UnlockLocal",
    "KeyInBox",
    "UnlockPickup",
    "BlockedUnlockPickup",
    "UnlockToUnlock",
    "ActionObjDoor",
    "FindObj",
    "KeyCorridor",
    "OneRoom",
    "MoveTwoAcross",
    "Synth",
    "SynthLoc",
    "SynthSeq",
    "MiniBossLevel",
    "BossLevel",
    "BossLevelNoUnlock",
)
_GOTO_SUCCESS = (2, 0, 2, 2)
_OPEN_SUCCESS = (0, 2, 5)
_PICKUP_SUCCESS = (1, 2, 2, 1, 3)
_PUT_NEXT_SUCCESS = (0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 3, 1, 2, 1, 2, 4)
_OPEN_OUT_OF_ORDER_SUCCESS = (
    0,
    0,
    2,
    0,
    2,
    2,
    5,
    1,
    2,
    2,
    5,
    0,
    0,
    2,
    2,
    1,
    5,
    5,
)


class BabyAITests(unittest.TestCase):
    def test_families_have_distinct_ids_and_profiles_define_identity(
        self,
    ) -> None:
        goto = BabyAIBenchmark(BabyAIConfig(profile="GoTo"))
        opened = BabyAIBenchmark(BabyAIConfig(profile="Open"))
        local = BabyAIBenchmark(BabyAIConfig(profile="GoToLocal"))
        self.assertEqual(
            goto.spec.id,
            "minigrid/BabyAI-GoTo-v0/mean-return-v1",
        )
        self.assertEqual(
            opened.spec.id,
            "minigrid/BabyAI-Open-v0/mean-return-v1",
        )
        self.assertEqual(goto.spec.max_episode_steps, 576)
        self.assertNotEqual(
            goto.spec.environment_digest,
            local.spec.environment_digest,
        )
        with self.assertRaises(ValueError):
            BabyAIConfig(profile="GoToAnything")

    def test_spec_documents_generated_tasks_and_exact_action_semantics(self) -> None:
        parameters = BabyAIBenchmark(
            BabyAIConfig(profile="BossLevelNoUnlock")
        ).spec.environment_parameters
        horizon = parameters["task_conditioned_horizon"]
        natural_termination = parameters["natural_termination"]
        action_notes = parameters["action_notes"]
        assert isinstance(horizon, str)
        assert isinstance(natural_termination, str)
        assert isinstance(action_notes, dict)
        pickup_note = action_notes["pick_up"]
        done_note = action_notes["done"]
        assert isinstance(pickup_note, str)
        assert isinstance(done_note, str)

        self.assertEqual(parameters["profile_max_episode_steps"], 1152)
        self.assertIn("actual upstream horizon", horizon)
        self.assertIn("non-debug profiles", natural_termination)
        self.assertIn("does not terminate", pickup_note)
        self.assertIn("No-op", done_note)

    def test_every_requested_profile_resets_with_strict_public_values(
        self,
    ) -> None:
        for profile in _PROFILES:
            with self.subTest(profile=profile):
                benchmark = BabyAIBenchmark(BabyAIConfig(profile=profile))
                episode = benchmark.episodes("validation", seed=7, count=1)[0]
                environment = benchmark.make_environment(episode)
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    assert isinstance(observation, dict)
                    self.assertEqual(
                        set(observation),
                        {"image", "direction", "mission"},
                    )
                    image = observation["image"]
                    mission = observation["mission"]
                    self.assertIsInstance(image, TensorValue)
                    assert isinstance(image, TensorValue)
                    self.assertEqual(image.shape, (7, 7, 3))
                    self.assertIsInstance(mission, str)
                    self.assertTrue(mission)
                    step = environment.step(1)
                    self.assertIsInstance(step.metrics, dict)
                    assert isinstance(step.metrics, dict)
                    self.assertEqual(len(step.metrics), 60)
                    self.assertEqual(step.metrics["step_count"], 1)
                    self.assertIn("front_object", step.metrics)
                    self.assertIn("visible_object_labels", step.metrics)
                    self.assertIn("terminal_reason", step.metrics)
                finally:
                    environment.close()

    def test_real_navigation_success_has_exact_reward_and_target_in_front(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="GoToRedBallGrey"))
        final = _run_actions(benchmark, _GOTO_SUCCESS)[-1]
        metrics = _step_metrics(final)

        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertAlmostEqual(final.reward, 1.0 - 0.9 * 4 / 64)
        self.assertEqual(metrics["front_object"], "red_ball")
        self.assertEqual(metrics["success"], True)
        self.assertEqual(metrics["terminal_reason"], "success")

    def test_real_open_and_pickup_successes_report_effective_events(self) -> None:
        opened = _run_actions(
            BabyAIBenchmark(BabyAIConfig(profile="OpenRedDoor")),
            _OPEN_SUCCESS,
        )[-1]
        picked_up = _run_actions(
            BabyAIBenchmark(BabyAIConfig(profile="PickupLoc")),
            _PICKUP_SUCCESS,
        )[-1]
        opened_metrics = _step_metrics(opened)
        pickup_metrics = _step_metrics(picked_up)

        self.assertTrue(opened.terminated)
        self.assertEqual(opened_metrics["front_object_before_action"], "red_closed_door")
        self.assertEqual(opened_metrics["front_object"], "red_open_door")
        self.assertEqual(opened_metrics["door_opened_this_step"], True)
        self.assertEqual(opened_metrics["door_open_event_count"], 1)
        self.assertTrue(picked_up.terminated)
        self.assertEqual(pickup_metrics["object_picked_up_this_step"], True)
        self.assertEqual(pickup_metrics["carried_object"], "blue_box")
        self.assertEqual(pickup_metrics["first_pickup_step"], 5)

    def test_real_put_next_reports_pickup_drop_and_completion(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="PutNextLocal"))
        steps = _run_actions(benchmark, _PUT_NEXT_SUCCESS)
        pickup_metrics = _step_metrics(steps[10])
        final_metrics = _step_metrics(steps[-1])

        self.assertEqual(pickup_metrics["carried_object"], "grey_key")
        self.assertEqual(pickup_metrics["object_picked_up_this_step"], True)
        self.assertTrue(steps[-1].terminated)
        self.assertEqual(final_metrics["object_dropped_this_step"], True)
        self.assertEqual(final_metrics["drop_event_count"], 1)
        self.assertEqual(final_metrics["task_stage"], "completed")

    def test_out_of_order_open_continues_and_must_be_repeated(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="OpenDoorsOrder"))
        steps = _run_actions(benchmark, _OPEN_OUT_OF_ORDER_SUCCESS)
        first_open_metrics = _step_metrics(steps[6])
        second_open_metrics = _step_metrics(steps[10])
        close_metrics = _step_metrics(steps[-2])
        final_metrics = _step_metrics(steps[-1])

        self.assertFalse(steps[6].terminated)
        self.assertEqual(first_open_metrics["front_object"], "purple_open_door")
        self.assertEqual(first_open_metrics["door_opened_this_step"], True)
        self.assertFalse(steps[10].terminated)
        self.assertEqual(second_open_metrics["front_object"], "green_open_door")
        self.assertTrue(steps[-1].terminated)
        self.assertEqual(close_metrics["door_closed_this_step"], True)
        self.assertEqual(final_metrics["door_opened_this_step"], True)
        self.assertEqual(final_metrics["door_open_event_count"], 3)

    def test_failed_interactions_and_timeout_are_actionable(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="GoToRedBallGrey"))
        attempts = _run_actions(benchmark, (3, 5))
        pickup_metrics = _step_metrics(attempts[0])
        toggle_metrics = _step_metrics(attempts[1])

        self.assertEqual(pickup_metrics["failed_pickup"], True)
        self.assertEqual(toggle_metrics["failed_toggle"], True)
        self.assertEqual(toggle_metrics["ineffective_action_fraction"], 1.0)

        timeout = _run_actions(benchmark, (6,) * 64)[-1]
        timeout_metrics = _step_metrics(timeout)
        self.assertFalse(timeout.terminated)
        self.assertTrue(timeout.truncated)
        self.assertEqual(timeout_metrics["remaining_steps"], 0)
        self.assertEqual(timeout_metrics["done_action_count"], 64)
        self.assertEqual(timeout_metrics["terminal_reason"], "time_limit")

    def test_scenario_and_invalid_actions_are_rejected(self) -> None:
        benchmark = BabyAIBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "Open"},
                )
            )
        environment = benchmark.make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(7)
        finally:
            environment.close()

    def test_feedback_keeps_identity_private(self) -> None:
        failed = EpisodeRecord(
            episode=EpisodeSpec(environment_seed=11),
            policy_seed=21,
            initial_observation=_empty_observation(),
            transitions=(),
            policy_failure="invalid_action",
        )
        feedback = BabyAIBenchmark().feedback((failed,))
        trace = feedback.artifacts[0].read_bytes()
        self.assertEqual(feedback.score, 0.0)
        self.assertNotIn(b"environment_seed", trace)
        self.assertNotIn(b"policy_seed", trace)
        self.assertNotIn(b'"profile"', trace)

    def test_baseline_feedback_aggregates_rich_real_traces(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="OpenRedDoor"))
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=3,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )

        content = result.feedback.content
        assert isinstance(content, dict)
        self.assertEqual(content["success_rate"], 1.0)
        self.assertEqual(result.feedback.score, content["mean_return"])
        self.assertEqual(content["door_opened_this_step_rate"], 1.0)
        self.assertGreater(_number_metric(content, "mean_unique_observation_count"), 0)
        documents = tuple(
            json.loads(line) for line in result.feedback.artifacts[0].read_bytes().splitlines()
        )
        transitions = tuple(document for document in documents if document["type"] == "transition")
        self.assertTrue(transitions)
        self.assertTrue(all("terminal_reason" in item["metrics"] for item in transitions))


def _run_actions(
    benchmark: BabyAIBenchmark,
    actions: tuple[int, ...],
) -> tuple[Step, ...]:
    environment = benchmark.make_environment(EpisodeSpec(environment_seed=5))
    steps: list[Step] = []
    try:
        environment.reset()
        for action in actions:
            steps.append(environment.step(action))
    finally:
        environment.close()
    return tuple(steps)


def _empty_observation() -> dict[str, PolicyValue]:
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=(7, 7, 3),
            data=bytes(147),
        ),
        "direction": 0,
        "mission": "go to a blue ball",
    }


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
