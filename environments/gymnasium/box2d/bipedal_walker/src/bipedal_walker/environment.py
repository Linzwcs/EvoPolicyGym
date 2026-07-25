"""One fresh Gymnasium BipedalWalker-v3 Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import BipedalWalkerConfig

_SCALAR_OBSERVATION_NAMES = (
    "hull_angle",
    "hull_angular_velocity",
    "horizontal_velocity",
    "vertical_velocity",
    "left_hip_angle",
    "left_hip_angular_velocity",
    "left_knee_angle",
    "left_knee_angular_velocity",
    "left_foot_contact",
    "right_hip_angle",
    "right_hip_angular_velocity",
    "right_knee_angle",
    "right_knee_angular_velocity",
    "right_foot_contact",
)
_OBSERVATION_SIZE = 24
_ACTION_SIZE = 4


class BipedalWalkerEnvironment:
    """The seeded strict adapter around configured BipedalWalker-v3."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: BipedalWalkerConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BipedalWalkerConfig:
            raise TypeError("config must be BipedalWalkerConfig")
        if episode.scenario is not None:
            raise ValueError(
                "BipedalWalker configuration belongs in BipedalWalkerConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "BipedalWalker-v3",
                hardcore=config.hardcore,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(observation)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        observation, reward, terminated, truncated, _ = self._environment.step(
            _action(action)
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "BipedalWalker returned invalid termination flags"
            )
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation),
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(value: PolicyValue) -> list[float]:
    if type(value) is not list or len(value) != _ACTION_SIZE:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -1.0 <= item <= 1.0
        ):
            raise InvalidAction()
        action.append(item)
    return action


def _observation(value: object) -> dict[str, PolicyValue]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise RuntimeError("BipedalWalker returned an invalid observation")
    items = tuple(value)
    if len(items) != _OBSERVATION_SIZE:
        raise RuntimeError(
            "BipedalWalker returned an invalid observation shape"
        )

    observation: dict[str, PolicyValue] = {}
    for index, name in enumerate(_SCALAR_OBSERVATION_NAMES):
        if name in {"left_foot_contact", "right_foot_contact"}:
            observation[name] = _contact(items[index], name=name)
        else:
            observation[name] = _number(items[index], name=name)
    observation["lidar_ranges"] = [
        _number(item, name=f"lidar range {index}")
        for index, item in enumerate(items[14:])
    ]
    return observation


def _contact(value: object, *, name: str) -> bool:
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(f"BipedalWalker returned an invalid {name}")
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"BipedalWalker returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"BipedalWalker returned a non-finite {name}")
    return number


__all__ = ["BipedalWalkerEnvironment"]
