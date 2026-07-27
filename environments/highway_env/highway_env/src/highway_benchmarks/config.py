"""Typed Host-selected HighwayEnv task profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Profile:
    environment_id: str
    max_episode_steps: int
    observation_kind: str
    action_size: int
    continuous: bool = False


_PROFILES = {
    "highway": _Profile("highway-v0", 40, "kinematics", 5),
    "merge": _Profile("merge-v1", 40, "kinematics", 5),
    "roundabout": _Profile("roundabout-v1", 11, "kinematics", 5),
    "intersection": _Profile("intersection-v2", 13, "kinematics", 3),
    "two-way": _Profile("two-way-v0", 15, "time_to_collision", 5),
    "exit": _Profile("exit-v1", 18, "kinematics", 5),
    "u-turn": _Profile("u-turn-v1", 10, "time_to_collision", 5),
    "parking": _Profile("parking-v0", 500, "goal_kinematics", 2, True),
    "racetrack": _Profile("racetrack-v1", 1_500, "occupancy_grid", 1, True),
    "lane-keeping": _Profile(
        "lane-keeping-v0",
        200,
        "vehicle_attributes",
        1,
        True,
    ),
}

HIGHWAY_PROFILES = tuple(_PROFILES)


@dataclass(frozen=True, slots=True)
class HighwayConfig:
    """Configuration that fixes one HighwayEnv Benchmark identity."""

    profile: str = "highway"

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in _PROFILES:
            raise ValueError(
                "profile must be one of: " + ", ".join(HIGHWAY_PROFILES)
            )

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile].environment_id

    @property
    def max_episode_steps(self) -> int:
        return _PROFILES[self.profile].max_episode_steps

    @property
    def observation_kind(self) -> str:
        return _PROFILES[self.profile].observation_kind

    @property
    def action_size(self) -> int:
        return _PROFILES[self.profile].action_size

    @property
    def continuous(self) -> bool:
        return _PROFILES[self.profile].continuous


__all__ = ["HIGHWAY_PROFILES", "HighwayConfig"]
