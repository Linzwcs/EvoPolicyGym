"""Typed Host-selected DeepMind Control Suite profiles."""

from __future__ import annotations

from dataclasses import dataclass

type ObservationField = tuple[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _Profile:
    domain: str
    task: str
    action_size: int
    observation_fields: tuple[ObservationField, ...]


_ACROBOT = (("orientations", (4,)), ("velocity", (2,)))
_BALL_IN_CUP = (("position", (4,)), ("velocity", (4,)))
_CARTPOLE = (("position", (3,)), ("velocity", (2,)))
_CHEETAH = (("position", (8,)), ("velocity", (9,)))
_FINGER_SPIN = (("position", (4,)), ("velocity", (3,)), ("touch", (2,)))
_FINGER_TURN = _FINGER_SPIN + (
    ("target_position", (2,)),
    ("dist_to_target", ()),
)
_FISH_SWIM = (
    ("joint_angles", (7,)),
    ("upright", ()),
    ("target", (3,)),
    ("velocity", (13,)),
)
_FISH_UPRIGHT = (
    ("joint_angles", (7,)),
    ("upright", ()),
    ("velocity", (13,)),
)
_HOPPER = (("position", (6,)), ("velocity", (7,)), ("touch", (2,)))
_HUMANOID = (
    ("joint_angles", (21,)),
    ("head_height", ()),
    ("extremities", (12,)),
    ("torso_vertical", (3,)),
    ("com_velocity", (3,)),
    ("velocity", (27,)),
)
_MANIPULATOR = (
    ("arm_pos", (8, 2)),
    ("arm_vel", (8,)),
    ("touch", (5,)),
    ("hand_pos", (4,)),
    ("object_pos", (4,)),
    ("object_vel", (3,)),
    ("target_pos", (4,)),
)
_PENDULUM = (("orientation", (2,)), ("velocity", (1,)))
_POINT_MASS = (("position", (2,)), ("velocity", (2,)))
_REACHER = (("position", (2,)), ("to_target", (2,)), ("velocity", (2,)))
_SWIMMER_15 = (
    ("joints", (14,)),
    ("to_target", (2,)),
    ("body_velocities", (45,)),
)
_SWIMMER_6 = (
    ("joints", (5,)),
    ("to_target", (2,)),
    ("body_velocities", (18,)),
)
_WALKER = (("height", ()), ("orientations", (14,)), ("velocity", (9,)))


_PROFILES = {
    "acrobot-swingup": _Profile("acrobot", "swingup", 1, _ACROBOT),
    "acrobot-swingup-sparse": _Profile("acrobot", "swingup_sparse", 1, _ACROBOT),
    "ball-in-cup-catch": _Profile("ball_in_cup", "catch", 2, _BALL_IN_CUP),
    "cartpole-balance": _Profile("cartpole", "balance", 1, _CARTPOLE),
    "cartpole-balance-sparse": _Profile("cartpole", "balance_sparse", 1, _CARTPOLE),
    "cartpole-swingup": _Profile("cartpole", "swingup", 1, _CARTPOLE),
    "cartpole-swingup-sparse": _Profile("cartpole", "swingup_sparse", 1, _CARTPOLE),
    "cheetah-run": _Profile("cheetah", "run", 6, _CHEETAH),
    "finger-spin": _Profile("finger", "spin", 2, _FINGER_SPIN),
    "finger-turn-easy": _Profile("finger", "turn_easy", 2, _FINGER_TURN),
    "finger-turn-hard": _Profile("finger", "turn_hard", 2, _FINGER_TURN),
    "fish-swim": _Profile("fish", "swim", 5, _FISH_SWIM),
    "fish-upright": _Profile("fish", "upright", 5, _FISH_UPRIGHT),
    "hopper-hop": _Profile("hopper", "hop", 4, _HOPPER),
    "hopper-stand": _Profile("hopper", "stand", 4, _HOPPER),
    "humanoid-run": _Profile("humanoid", "run", 21, _HUMANOID),
    "humanoid-stand": _Profile("humanoid", "stand", 21, _HUMANOID),
    "humanoid-walk": _Profile("humanoid", "walk", 21, _HUMANOID),
    "manipulator-bring-ball": _Profile("manipulator", "bring_ball", 5, _MANIPULATOR),
    "pendulum-swingup": _Profile("pendulum", "swingup", 1, _PENDULUM),
    "point-mass-easy": _Profile("point_mass", "easy", 2, _POINT_MASS),
    "reacher-easy": _Profile("reacher", "easy", 2, _REACHER),
    "reacher-hard": _Profile("reacher", "hard", 2, _REACHER),
    "swimmer-swimmer15": _Profile("swimmer", "swimmer15", 14, _SWIMMER_15),
    "swimmer-swimmer6": _Profile("swimmer", "swimmer6", 5, _SWIMMER_6),
    "walker-run": _Profile("walker", "run", 6, _WALKER),
    "walker-stand": _Profile("walker", "stand", 6, _WALKER),
    "walker-walk": _Profile("walker", "walk", 6, _WALKER),
}

DM_CONTROL_PROFILES = tuple(_PROFILES)


@dataclass(frozen=True, slots=True)
class DmControlConfig:
    """Configuration that fixes one Control Suite Benchmark identity."""

    profile: str = "cartpole-swingup"
    max_episode_steps: int = 1_000

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if self.profile not in _PROFILES:
            raise ValueError(
                "profile must be one of: " + ", ".join(DM_CONTROL_PROFILES)
            )
        if type(self.max_episode_steps) is not int:
            raise TypeError("max_episode_steps must be an exact integer")
        if not 1 <= self.max_episode_steps <= 1_000:
            raise ValueError("max_episode_steps must be between 1 and 1000")

    @property
    def domain(self) -> str:
        return _PROFILES[self.profile].domain

    @property
    def task(self) -> str:
        return _PROFILES[self.profile].task

    @property
    def action_size(self) -> int:
        return _PROFILES[self.profile].action_size

    @property
    def observation_fields(self) -> tuple[ObservationField, ...]:
        return _PROFILES[self.profile].observation_fields


__all__ = ["DM_CONTROL_PROFILES", "DmControlConfig", "ObservationField"]
