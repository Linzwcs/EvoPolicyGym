"""One fresh MiniGrid DoorKey Environment per Episode."""

from __future__ import annotations

import math
import operator
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401  # Import registers MiniGrid environments.
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import DoorKeyConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "use the key to open the door and then get to the goal"
_DOOR = 4
_KEY = 5
_OPEN = 0


class DoorKeyEnvironment:
    """The seeded strict adapter around a configured MiniGrid DoorKey task."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: DoorKeyConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not DoorKeyConfig:
            raise TypeError("config must be DoorKeyConfig")
        if episode.scenario is not None:
            raise ValueError(
                "DoorKey configuration belongs in DoorKeyConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._picked_up_key = False
        self._opened_door = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, _, _ = _observation(observation)
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
            raise RuntimeError("MiniGrid DoorKey returned invalid termination flags")
        number = _number(reward, name="reward")
        public, carrying_key, open_door_visible = _observation(observation)
        self._picked_up_key = self._picked_up_key or carrying_key
        self._opened_door = self._opened_door or open_door_visible
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "carrying_key": carrying_key,
                "picked_up_key": self._picked_up_key,
                "opened_door": self._opened_door,
                "success": bool(terminated and number > 0.0),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], bool, bool]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid DoorKey returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid DoorKey returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid DoorKey returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid DoorKey returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid DoorKey returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid DoorKey returned an invalid mission")

    carrying_key = bool(image[3, 6, 0] == _KEY)
    open_door_visible = bool(
        numpy.any(
            (image[:, :, 0] == _DOOR) & (image[:, :, 2] == _OPEN)
        )
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
        carrying_key,
        open_door_visible,
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid DoorKey returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid DoorKey returned a non-finite {name}")
    return number


__all__ = ["DoorKeyEnvironment"]
