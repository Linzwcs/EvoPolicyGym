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
            "minigrid/BabyAI-GoTo-v0/success-rate-v1",
        )
        self.assertEqual(
            opened.spec.id,
            "minigrid/BabyAI-Open-v0/success-rate-v1",
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

        self.assertEqual(parameters["profile_max_episode_steps"], 1152)
        self.assertIn("actual upstream horizon", parameters["task_conditioned_horizon"])
        self.assertIn("non-debug profiles", parameters["natural_termination"])
        self.assertIn("does not terminate", parameters["action_notes"]["pick_up"])
        self.assertIn("No-op", parameters["action_notes"]["done"])

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

        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertAlmostEqual(final.reward, 1.0 - 0.9 * 4 / 64)
        self.assertEqual(final.metrics["front_object"], "red_ball")
        self.assertEqual(final.metrics["success"], True)
        self.assertEqual(final.metrics["terminal_reason"], "success")

    def test_real_open_and_pickup_successes_report_effective_events(self) -> None:
        opened = _run_actions(
            BabyAIBenchmark(BabyAIConfig(profile="OpenRedDoor")),
            _OPEN_SUCCESS,
        )[-1]
        picked_up = _run_actions(
            BabyAIBenchmark(BabyAIConfig(profile="PickupLoc")),
            _PICKUP_SUCCESS,
        )[-1]

        self.assertTrue(opened.terminated)
        self.assertEqual(opened.metrics["front_object_before_action"], "red_closed_door")
        self.assertEqual(opened.metrics["front_object"], "red_open_door")
        self.assertEqual(opened.metrics["door_opened_this_step"], True)
        self.assertEqual(opened.metrics["door_open_event_count"], 1)
        self.assertTrue(picked_up.terminated)
        self.assertEqual(picked_up.metrics["object_picked_up_this_step"], True)
        self.assertEqual(picked_up.metrics["carried_object"], "blue_box")
        self.assertEqual(picked_up.metrics["first_pickup_step"], 5)

    def test_real_put_next_reports_pickup_drop_and_completion(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="PutNextLocal"))
        steps = _run_actions(benchmark, _PUT_NEXT_SUCCESS)

        self.assertEqual(steps[10].metrics["carried_object"], "grey_key")
        self.assertEqual(steps[10].metrics["object_picked_up_this_step"], True)
        self.assertTrue(steps[-1].terminated)
        self.assertEqual(steps[-1].metrics["object_dropped_this_step"], True)
        self.assertEqual(steps[-1].metrics["drop_event_count"], 1)
        self.assertEqual(steps[-1].metrics["task_stage"], "completed")

    def test_out_of_order_open_continues_and_must_be_repeated(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="OpenDoorsOrder"))
        steps = _run_actions(benchmark, _OPEN_OUT_OF_ORDER_SUCCESS)

        self.assertFalse(steps[6].terminated)
        self.assertEqual(steps[6].metrics["front_object"], "purple_open_door")
        self.assertEqual(steps[6].metrics["door_opened_this_step"], True)
        self.assertFalse(steps[10].terminated)
        self.assertEqual(steps[10].metrics["front_object"], "green_open_door")
        self.assertTrue(steps[-1].terminated)
        self.assertEqual(steps[-2].metrics["door_closed_this_step"], True)
        self.assertEqual(steps[-1].metrics["door_opened_this_step"], True)
        self.assertEqual(steps[-1].metrics["door_open_event_count"], 3)

    def test_failed_interactions_and_timeout_are_actionable(self) -> None:
        benchmark = BabyAIBenchmark(BabyAIConfig(profile="GoToRedBallGrey"))
        attempts = _run_actions(benchmark, (3, 5))

        self.assertEqual(attempts[0].metrics["failed_pickup"], True)
        self.assertEqual(attempts[1].metrics["failed_toggle"], True)
        self.assertEqual(attempts[1].metrics["ineffective_action_fraction"], 1.0)

        timeout = _run_actions(benchmark, (6,) * 64)[-1]
        self.assertFalse(timeout.terminated)
        self.assertTrue(timeout.truncated)
        self.assertEqual(timeout.metrics["remaining_steps"], 0)
        self.assertEqual(timeout.metrics["done_action_count"], 64)
        self.assertEqual(timeout.metrics["terminal_reason"], "time_limit")

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

        self.assertEqual(result.feedback.score, 1.0)
        self.assertEqual(result.feedback.content["door_opened_this_step_rate"], 1.0)
        self.assertGreater(result.feedback.content["mean_unique_observation_count"], 0)
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


if __name__ == "__main__":
    unittest.main()
