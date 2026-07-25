"""One fresh Gymnasium Pendulum Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

_OBSERVATION_NAMES = (
    "cos_theta",
    "sin_theta",
    "theta_angular_velocity",
)


class PendulumEnvironment:
    """The seeded strict adapter around Gymnasium Pendulum-v1."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, tuple[float]],
            gymnasium.make("Pendulum-v1"),
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
        if (
            type(action) is not float
            or not math.isfinite(action)
            or not -2.0 <= action <= 2.0
        ):
            raise InvalidAction()

        observation, reward, terminated, truncated, _ = self._environment.step(
            (action,)
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Pendulum returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation),
            reward=_number(reward),
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(value: object) -> dict[str, PolicyValue]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise RuntimeError("Pendulum returned an invalid observation")
    items = tuple(value)
    if len(items) != len(_OBSERVATION_NAMES):
        raise RuntimeError("Pendulum returned an invalid observation shape")
    return {
        name: _number(item)
        for name, item in zip(_OBSERVATION_NAMES, items, strict=True)
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("Pendulum returned a non-numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("Pendulum returned a non-finite value")
    return number


__all__ = ["PendulumEnvironment"]
