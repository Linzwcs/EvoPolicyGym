"""One fresh Gymnasium LunarLander-v3 Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import LunarLanderConfig

_OBSERVATION_NAMES = (
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)


class LunarLanderEnvironment:
    """The seeded strict adapter around configured LunarLander-v3."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: LunarLanderConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not LunarLanderConfig:
            raise TypeError("config must be LunarLanderConfig")
        if episode.scenario is not None:
            raise ValueError(
                "LunarLander configuration belongs in LunarLanderConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._continuous = config.continuous
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "LunarLander-v3",
                continuous=config.continuous,
                gravity=config.gravity,
                enable_wind=config.enable_wind,
                wind_power=config.wind_power,
                turbulence_power=config.turbulence_power,
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

        applied = (
            _continuous_action(action)
            if self._continuous
            else _discrete_action(action)
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            applied
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "LunarLander returned invalid termination flags"
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


def _discrete_action(value: PolicyValue) -> int:
    if type(value) is not int or value not in {0, 1, 2, 3}:
        raise InvalidAction()
    return value


def _continuous_action(value: PolicyValue) -> list[float]:
    if type(value) is not list or len(value) != 2:
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
        raise RuntimeError("LunarLander returned an invalid observation")
    items = tuple(value)
    if len(items) != len(_OBSERVATION_NAMES):
        raise RuntimeError(
            "LunarLander returned an invalid observation shape"
        )
    return {
        "x_position": _number(items[0], name="x position"),
        "y_position": _number(items[1], name="y position"),
        "x_velocity": _number(items[2], name="x velocity"),
        "y_velocity": _number(items[3], name="y velocity"),
        "angle": _number(items[4], name="angle"),
        "angular_velocity": _number(
            items[5],
            name="angular velocity",
        ),
        "left_leg_contact": _contact(items[6], name="left leg contact"),
        "right_leg_contact": _contact(
            items[7],
            name="right leg contact",
        ),
    }


def _contact(value: object, *, name: str) -> bool:
    number = _number(value, name=name)
    if number not in {0.0, 1.0}:
        raise RuntimeError(f"LunarLander returned an invalid {name}")
    return number == 1.0


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"LunarLander returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"LunarLander returned a non-finite {name}")
    return number


__all__ = ["LunarLanderEnvironment"]
