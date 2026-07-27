"""Typed Host-selected Gymnasium-Robotics profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Profile:
    environment_id: str
    family: str
    max_episode_steps: int
    action_size: int
    observation_size: int
    goal_size: int | None
    action_dtype: str = "float32"


_PROFILES = {
    "fetch-reach": _Profile("FetchReach-v4", "fetch", 50, 4, 10, 3),
    "fetch-push": _Profile("FetchPush-v4", "fetch", 50, 4, 25, 3),
    "fetch-slide": _Profile("FetchSlide-v4", "fetch", 50, 4, 25, 3),
    "fetch-pick-and-place": _Profile(
        "FetchPickAndPlace-v4",
        "fetch",
        50,
        4,
        25,
        3,
    ),
    "point-maze": _Profile(
        "PointMaze_UMaze-v3",
        "maze",
        300,
        2,
        4,
        2,
    ),
    "ant-maze": _Profile(
        "AntMaze_UMaze-v5",
        "maze",
        700,
        8,
        105,
        2,
    ),
    "adroit-hand-door": _Profile(
        "AdroitHandDoor-v1",
        "adroit",
        200,
        28,
        39,
        None,
    ),
    "adroit-hand-hammer": _Profile(
        "AdroitHandHammer-v1",
        "adroit",
        200,
        26,
        46,
        None,
    ),
    "adroit-hand-pen": _Profile(
        "AdroitHandPen-v1",
        "adroit",
        200,
        24,
        45,
        None,
    ),
    "adroit-hand-relocate": _Profile(
        "AdroitHandRelocate-v1",
        "adroit",
        200,
        30,
        39,
        None,
    ),
    "hand-reach": _Profile(
        "HandReach-v3",
        "shadow-hand",
        50,
        20,
        63,
        15,
    ),
    "hand-manipulate-block": _Profile(
        "HandManipulateBlock-v1",
        "shadow-hand",
        100,
        20,
        61,
        7,
    ),
    "hand-manipulate-block-boolean-touch": _Profile(
        "HandManipulateBlock_BooleanTouchSensors-v1",
        "shadow-hand-touch",
        100,
        20,
        153,
        7,
    ),
    "hand-manipulate-block-continuous-touch": _Profile(
        "HandManipulateBlock_ContinuousTouchSensors-v1",
        "shadow-hand-touch",
        100,
        20,
        153,
        7,
    ),
    "hand-manipulate-egg": _Profile(
        "HandManipulateEgg-v1",
        "shadow-hand",
        100,
        20,
        61,
        7,
    ),
    "hand-manipulate-egg-boolean-touch": _Profile(
        "HandManipulateEgg_BooleanTouchSensors-v1",
        "shadow-hand-touch",
        100,
        20,
        153,
        7,
    ),
    "hand-manipulate-egg-continuous-touch": _Profile(
        "HandManipulateEgg_ContinuousTouchSensors-v1",
        "shadow-hand-touch",
        100,
        20,
        153,
        7,
    ),
    "hand-manipulate-pen": _Profile(
        "HandManipulatePen-v1",
        "shadow-hand",
        100,
        20,
        61,
        7,
    ),
    "hand-manipulate-pen-boolean-touch": _Profile(
        "HandManipulatePen_BooleanTouchSensors-v1",
        "shadow-hand-touch",
        100,
        20,
        153,
        7,
    ),
    "hand-manipulate-pen-continuous-touch": _Profile(
        "HandManipulatePen_ContinuousTouchSensors-v1",
        "shadow-hand-touch",
        100,
        20,
        153,
        7,
    ),
    "franka-kitchen": _Profile(
        "FrankaKitchen-v1",
        "franka-kitchen",
        280,
        9,
        59,
        -1,
        "float64",
    ),
}

ROBOTICS_PROFILES = tuple(_PROFILES)


@dataclass(frozen=True, slots=True)
class RoboticsConfig:
    """Configuration that fixes one Robotics Benchmark identity."""

    profile: str = "fetch-reach"

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in _PROFILES:
            raise ValueError(
                "profile must be one of: " + ", ".join(ROBOTICS_PROFILES)
            )

    @property
    def environment_id(self) -> str:
        return _PROFILES[self.profile].environment_id

    @property
    def family(self) -> str:
        return _PROFILES[self.profile].family

    @property
    def max_episode_steps(self) -> int:
        return _PROFILES[self.profile].max_episode_steps

    @property
    def action_size(self) -> int:
        return _PROFILES[self.profile].action_size

    @property
    def action_dtype(self) -> str:
        return _PROFILES[self.profile].action_dtype

    @property
    def observation_size(self) -> int:
        return _PROFILES[self.profile].observation_size

    @property
    def goal_size(self) -> int | None:
        return _PROFILES[self.profile].goal_size


__all__ = ["ROBOTICS_PROFILES", "RoboticsConfig"]

