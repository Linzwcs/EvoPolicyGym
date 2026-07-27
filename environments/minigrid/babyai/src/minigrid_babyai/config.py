"""Typed, public BabyAI task-family configuration."""

from __future__ import annotations

from dataclasses import dataclass

type Profile = tuple[str, str, int]

_PROFILES: dict[str, Profile] = {
    # GoTo family
    "GoToRedBallGrey": ("GoTo", "BabyAI-GoToRedBallGrey-v0", 64),
    "GoToRedBall": ("GoTo", "BabyAI-GoToRedBall-v0", 64),
    "GoToRedBallNoDists": (
        "GoTo",
        "BabyAI-GoToRedBallNoDists-v0",
        64,
    ),
    "GoToObj": ("GoTo", "BabyAI-GoToObj-v0", 64),
    "GoToLocal": ("GoTo", "BabyAI-GoToLocal-v0", 64),
    "GoTo": ("GoTo", "BabyAI-GoTo-v0", 576),
    "GoToImpUnlock": ("GoTo", "BabyAI-GoToImpUnlock-v0", 576),
    "GoToSeq": ("GoTo", "BabyAI-GoToSeq-v0", 1152),
    "GoToRedBlueBall": ("GoTo", "BabyAI-GoToRedBlueBall-v0", 64),
    "GoToDoor": ("GoTo", "BabyAI-GoToDoor-v0", 441),
    "GoToObjDoor": ("GoTo", "BabyAI-GoToObjDoor-v0", 576),
    # Open family
    "Open": ("Open", "BabyAI-Open-v0", 576),
    "OpenRedDoor": ("Open", "BabyAI-OpenRedDoor-v0", 50),
    "OpenDoor": ("Open", "BabyAI-OpenDoor-v0", 576),
    "OpenTwoDoors": ("Open", "BabyAI-OpenTwoDoors-v0", 720),
    "OpenDoorsOrder": ("Open", "BabyAI-OpenDoorsOrderN4-v0", 720),
    # Pickup and placement family
    "Pickup": ("PickupPut", "BabyAI-Pickup-v0", 576),
    "UnblockPickup": (
        "PickupPut",
        "BabyAI-UnblockPickup-v0",
        576,
    ),
    "PickupLoc": ("PickupPut", "BabyAI-PickupLoc-v0", 64),
    "PickupDist": ("PickupPut", "BabyAI-PickupDist-v0", 49),
    "PickupAbove": ("PickupPut", "BabyAI-PickupAbove-v0", 288),
    "PutNextLocal": ("PickupPut", "BabyAI-PutNextLocal-v0", 128),
    "PutNext": ("PickupPut", "BabyAI-PutNextS7N4-v0", 392),
    # Unlock family
    "Unlock": ("Unlock", "BabyAI-Unlock-v0", 576),
    "UnlockLocal": ("Unlock", "BabyAI-UnlockLocal-v0", 576),
    "KeyInBox": ("Unlock", "BabyAI-KeyInBox-v0", 576),
    "UnlockPickup": ("Unlock", "BabyAI-UnlockPickup-v0", 72),
    "BlockedUnlockPickup": (
        "Unlock",
        "BabyAI-BlockedUnlockPickup-v0",
        576,
    ),
    "UnlockToUnlock": ("Unlock", "BabyAI-UnlockToUnlock-v0", 1080),
    # Composite family
    "ActionObjDoor": (
        "Composite",
        "BabyAI-ActionObjDoor-v0",
        441,
    ),
    "FindObj": ("Composite", "BabyAI-FindObjS7-v0", 980),
    "KeyCorridor": ("Composite", "BabyAI-KeyCorridor-v0", 1080),
    "OneRoom": ("Composite", "BabyAI-OneRoomS8-v0", 64),
    "MoveTwoAcross": (
        "Composite",
        "BabyAI-MoveTwoAcrossS8N9-v0",
        1024,
    ),
    "Synth": ("Composite", "BabyAI-Synth-v0", 576),
    "SynthLoc": ("Composite", "BabyAI-SynthLoc-v0", 576),
    "SynthSeq": ("Composite", "BabyAI-SynthSeq-v0", 1152),
    "MiniBossLevel": (
        "Composite",
        "BabyAI-MiniBossLevel-v0",
        600,
    ),
    "BossLevel": ("Composite", "BabyAI-BossLevel-v0", 1152),
    "BossLevelNoUnlock": (
        "Composite",
        "BabyAI-BossLevelNoUnlock-v0",
        1152,
    ),
}


@dataclass(frozen=True, slots=True)
class BabyAIConfig:
    """Parameters defining one BabyAI family/profile Benchmark identity."""

    profile: str = "GoTo"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in _PROFILES:
            choices = "', '".join(_PROFILES)
            raise ValueError(f"profile must be one of '{choices}'")

    @property
    def family(self) -> str:
        return _PROFILES[self.profile][0]

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile][1]

    @property
    def max_episode_steps(self) -> int:
        return _PROFILES[self.profile][2]


__all__ = ["BabyAIConfig"]
