"""One fresh Gymnasium CarRacing-v3 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import CarRacingConfig

_FRAME_SHAPE = (96, 96, 3)


class CarRacingEnvironment:
    """The seeded strict adapter around configured CarRacing-v3."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: CarRacingConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not CarRacingConfig:
            raise TypeError("config must be CarRacingConfig")
        if episode.scenario is not None:
            raise ValueError(
                "CarRacing configuration belongs in CarRacingConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._continuous = config.continuous
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "CarRacing-v3",
                continuous=config.continuous,
                lap_complete_percent=config.lap_complete_percent,
                domain_randomize=config.domain_randomize,
                verbose=False,
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
        return _frame(observation)

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
            raise RuntimeError("CarRacing returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_frame(observation),
            reward=_number(reward, name="reward"),
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _continuous_action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 3:
        raise InvalidAction()
    bounds = ((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    action: list[float] = []
    for item, (minimum, maximum) in zip(value, bounds, strict=True):
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not minimum <= item <= maximum
        ):
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _discrete_action(value: PolicyValue) -> int:
    if type(value) is not int or value not in {0, 1, 2, 3, 4}:
        raise InvalidAction()
    return value


def _frame(value: object) -> TensorValue:
    if (
        type(value) is not numpy.ndarray
        or value.shape != _FRAME_SHAPE
        or value.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("CarRacing returned an invalid RGB frame")
    return TensorValue(
        dtype="uint8",
        shape=_FRAME_SHAPE,
        data=value.tobytes(order="C"),
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"CarRacing returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"CarRacing returned a non-finite {name}")
    return number


__all__ = ["CarRacingEnvironment"]
