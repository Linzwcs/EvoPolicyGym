"""One fresh MiniGrid DynamicObstacles Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401  # Import registers MiniGrid environments.
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import DynamicObstaclesConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(3))
_MISSION = "get to the green goal square"
_GOAL = 8


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    goal_visible: bool


class DynamicObstaclesEnvironment:
    """Strict seeded adapter around MiniGrid DynamicObstacles."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: DynamicObstaclesConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not DynamicObstaclesConfig:
            raise TypeError("config must be DynamicObstaclesConfig")
        if episode.scenario is not None:
            raise ValueError(
                "DynamicObstacles configuration belongs in "
                "DynamicObstaclesConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._goal_found = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._goal_found = facts.goal_visible
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
            raise RuntimeError(
                "MiniGrid DynamicObstacles returned invalid termination flags"
            )
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        self._goal_found = self._goal_found or facts.goal_visible
        collided = bool(terminated and number < 0.0)
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "goal_found": self._goal_found,
                "collision": collided,
                "success": success,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError(
            "MiniGrid DynamicObstacles returned an invalid observation"
        )

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError(
            "MiniGrid DynamicObstacles returned an invalid image"
        )
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError(
            "MiniGrid DynamicObstacles returned out-of-range image codes"
        )

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid DynamicObstacles returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError(
            "MiniGrid DynamicObstacles returned an invalid direction"
        )

    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError(
            "MiniGrid DynamicObstacles returned an invalid mission"
        )

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
        _ObservationFacts(
            goal_visible=bool(numpy.any(image[:, :, 0] == _GOAL)),
        ),
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(
            f"MiniGrid DynamicObstacles returned an invalid {name}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            f"MiniGrid DynamicObstacles returned a non-finite {name}"
        )
    return number


__all__ = ["DynamicObstaclesEnvironment"]
