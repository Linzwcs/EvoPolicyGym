"""One fresh InvertedDoublePendulum-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import InvertedDoublePendulumConfig

_OBSERVATION_NAMES = (
    "cart_position",
    "pole1_sin",
    "pole2_sin",
    "pole1_cos",
    "pole2_cos",
    "cart_velocity",
    "pole1_angular_velocity",
    "pole2_angular_velocity",
    "cart_constraint_force",
)


class InvertedDoublePendulumEnvironment:
    """Strict adapter around configured InvertedDoublePendulum-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: InvertedDoublePendulumConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not InvertedDoublePendulumConfig:
            raise TypeError(
                "config must be InvertedDoublePendulumConfig"
            )
        if episode.scenario is not None:
            raise ValueError(
                "InvertedDoublePendulum configuration belongs in "
                "InvertedDoublePendulumConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "InvertedDoublePendulum-v5",
                frame_skip=config.frame_skip,
                healthy_reward=config.healthy_reward,
                reset_noise_scale=config.reset_noise_scale,
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

        observation, reward, terminated, truncated, info = (
            self._environment.step(_action(action))
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "InvertedDoublePendulum returned invalid termination flags"
            )
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation),
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
            metrics=_reward_metrics(info),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 1:
        raise InvalidAction()
    item = value[0]
    if (
        type(item) is not float
        or not math.isfinite(item)
        or not -1.0 <= item <= 1.0
    ):
        raise InvalidAction()
    return numpy.asarray([item], dtype=numpy.float32)


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (9,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError(
            "InvertedDoublePendulum returned an invalid observation"
        )
    return {
        name: _number(item, name=name)
        for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _reward_metrics(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise RuntimeError(
            "InvertedDoublePendulum returned invalid reward metrics"
        )
    required = {
        "reward_survive",
        "distance_penalty",
        "velocity_penalty",
    }
    if not required.issubset(value):
        raise RuntimeError(
            "InvertedDoublePendulum omitted reward metrics"
        )
    return {
        key: _number(value[key], name=key.replace("_", " "))
        for key in sorted(required)
    }


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(
            f"InvertedDoublePendulum returned an invalid {name}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            f"InvertedDoublePendulum returned a non-finite {name}"
        )
    return number


__all__ = ["InvertedDoublePendulumEnvironment"]
