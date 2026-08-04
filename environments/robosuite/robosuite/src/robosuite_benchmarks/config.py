"""Typed Host-selected robosuite manipulation profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Profile:
    environment_id: str
    robot_count: int
    action_size: int
    proprioception_size: int
    object_state_size: int


_PROFILES = {
    "lift": _Profile("Lift", 1, 7, 50, 10),
    "stack": _Profile("Stack", 1, 7, 50, 23),
    "nut-assembly": _Profile("NutAssembly", 1, 7, 50, 28),
    "nut-assembly-single": _Profile("NutAssemblySingle", 1, 7, 50, 15),
    "nut-assembly-square": _Profile("NutAssemblySquare", 1, 7, 50, 14),
    "nut-assembly-round": _Profile("NutAssemblyRound", 1, 7, 50, 14),
    "pick-place": _Profile("PickPlace", 1, 7, 50, 56),
    "pick-place-single": _Profile("PickPlaceSingle", 1, 7, 50, 15),
    "pick-place-milk": _Profile("PickPlaceMilk", 1, 7, 50, 14),
    "pick-place-bread": _Profile("PickPlaceBread", 1, 7, 50, 14),
    "pick-place-cereal": _Profile("PickPlaceCereal", 1, 7, 50, 14),
    "pick-place-can": _Profile("PickPlaceCan", 1, 7, 50, 14),
    "door": _Profile("Door", 1, 7, 50, 14),
    "wipe": _Profile("Wipe", 1, 6, 47, 8),
    "tool-hang": _Profile("ToolHang", 1, 7, 50, 44),
    "two-arm-lift": _Profile("TwoArmLift", 2, 14, 100, 19),
    "two-arm-peg-in-hole": _Profile("TwoArmPegInHole", 2, 12, 92, 17),
    "two-arm-handover": _Profile("TwoArmHandover", 2, 14, 100, 16),
    "two-arm-transport": _Profile("TwoArmTransport", 2, 14, 100, 41),
}

ROBOSUITE_PROFILES = tuple(_PROFILES)


@dataclass(frozen=True, slots=True)
class RobosuiteConfig:
    """Configuration that fixes one robosuite Benchmark identity."""

    profile: str = "lift"
    max_episode_steps: int = 500

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in _PROFILES:
            raise ValueError(
                "profile must be one of: " + ", ".join(ROBOSUITE_PROFILES)
            )
        if type(self.max_episode_steps) is not int:
            raise TypeError("max_episode_steps must be an exact integer")
        if not 1 <= self.max_episode_steps <= 1_000:
            raise ValueError("max_episode_steps must be between 1 and 1000")

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile].environment_id

    @property
    def robot_count(self) -> int:
        return _PROFILES[self.profile].robot_count

    @property
    def action_size(self) -> int:
        return _PROFILES[self.profile].action_size

    @property
    def proprioception_size(self) -> int:
        return _PROFILES[self.profile].proprioception_size

    @property
    def object_state_size(self) -> int:
        return _PROFILES[self.profile].object_state_size


__all__ = ["ROBOSUITE_PROFILES", "RobosuiteConfig"]
