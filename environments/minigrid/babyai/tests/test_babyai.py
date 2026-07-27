from __future__ import annotations

import unittest

from evopolicygym.authoring import EpisodeRecord, EpisodeSpec, InvalidAction
from evopolicygym.policy import PolicyValue, TensorValue

from minigrid_babyai import BabyAIBenchmark, BabyAIConfig

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
                    self.assertEqual(set(step.metrics), {"success"})
                finally:
                    environment.close()

    def test_scenario_and_invalid_actions_are_rejected(self) -> None:
        benchmark = BabyAIBenchmark()
        with self.assertRaises(ValueError):
            benchmark.make_environment(
                EpisodeSpec(
                    environment_seed=1,
                    scenario={"profile": "Open"},
                )
            )
        environment = benchmark.make_environment(
            EpisodeSpec(environment_seed=1)
        )
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
