"""One fresh strict Stable-Retro Airstriker instance per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
import stable_retro  # type: ignore[import-untyped]
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from gymnasium.spaces import Discrete

from .config import AirstrikerConfig

_OBSERVATION_SHAPE = (224, 320, 3)
_ACTION_COUNT = 126
_MAX_EPISODE_STEPS = 18_000


class AirstrikerEnvironment:
    """Seeded adapter around the bundled Airstriker ROM and Level 1 state."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: AirstrikerConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not AirstrikerConfig:
            raise TypeError("config must be AirstrikerConfig")
        if episode.scenario is not None:
            raise ValueError("Airstriker configuration belongs in its config")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            stable_retro.make(
                game=config.game,
                state=config.state,
                render_mode="rgb_array",
                use_restricted_actions=stable_retro.Actions.DISCRETE,
            ),
        )
        action_space = self._environment.action_space
        if (
            not isinstance(action_space, Discrete)
            or action_space.n != _ACTION_COUNT
            or action_space.start != 0
        ):
            self._environment.close()
            raise RuntimeError(
                "Stable-Retro restricted action space changed incompatibly"
            )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0

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
        if type(action) is not int or not 0 <= action < _ACTION_COUNT:
            raise InvalidAction()
        observation, reward, terminated, truncated, _ = (
            self._environment.step(action)
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Stable-Retro returned invalid termination flags")
        self._steps += 1
        if self._steps >= _MAX_EPISODE_STEPS and not terminated:
            truncated = True
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


def _observation(value: object) -> TensorValue:
    if (
        type(value) is not numpy.ndarray
        or value.dtype != numpy.dtype("uint8")
        or value.shape != _OBSERVATION_SHAPE
    ):
        raise RuntimeError("Stable-Retro returned an invalid RGB observation")
    return TensorValue(
        dtype="uint8",
        shape=_OBSERVATION_SHAPE,
        data=numpy.ascontiguousarray(value).tobytes(order="C"),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("Stable-Retro returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("Stable-Retro returned non-finite reward")
    return number


__all__ = ["AirstrikerEnvironment"]
