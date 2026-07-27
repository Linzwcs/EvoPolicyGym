"""One fresh strict BabyAI Environment per Episode."""

from __future__ import annotations

import math
import operator
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import BabyAIConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))


class BabyAIEnvironment:
    """Strict seeded adapter around one Host-selected BabyAI profile."""

    def __init__(self, episode: EpisodeSpec, *, config: BabyAIConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BabyAIConfig:
            raise TypeError("config must be BabyAIConfig")
        if episode.scenario is not None:
            raise ValueError("BabyAI configuration belongs in BabyAIConfig")
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._mission: str | None = None
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, mission = _observation(observation)
        horizon = self._environment.get_wrapper_attr("max_steps")
        if (
            type(horizon) is not int
            or not 0 < horizon <= self._config.max_episode_steps
        ):
            raise RuntimeError("BabyAI returned an unexpected horizon")
        self._mission = mission
        self._started = True
        return public

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("BabyAI returned invalid termination flags")
        number = _number(reward)
        public, mission = _observation(observation)
        if mission != self._mission:
            raise RuntimeError("BabyAI changed its mission during an Episode")
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={"success": success},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], str]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("BabyAI returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
        or numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("BabyAI returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("BabyAI returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("BabyAI returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or not mission:
        raise RuntimeError("BabyAI returned invalid mission")
    return (
        {
            "image": TensorValue(
                dtype="uint8",
                shape=_IMAGE_SHAPE,
                data=image.tobytes(order="C"),
            ),
            "direction": direction,
            "mission": mission,
        },
        mission,
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("BabyAI returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("BabyAI returned non-finite reward")
    return number


__all__ = ["BabyAIEnvironment"]
