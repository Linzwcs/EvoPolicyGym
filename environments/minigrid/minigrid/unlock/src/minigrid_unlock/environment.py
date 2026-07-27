"""One fresh MiniGrid Unlock Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

_ENVIRONMENT_ID = "MiniGrid-Unlock-v0"
_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "open the door"
_DOOR = 4
_KEY = 5
_LOCKED = 2


@dataclass(frozen=True, slots=True)
class _Facts:
    key_visible: bool
    carried_key_color: int | None
    front_locked_door_color: int | None


class UnlockEnvironment:
    """Strict seeded adapter around MiniGrid Unlock."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Unlock has no Episode scenario overrides")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(_ENVIRONMENT_ID),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._carried_key_color: int | None = None
        self._front_locked_door_color: int | None = None
        self._key_found = False
        self._key_picked_up = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._update(facts)
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
        opened_door = bool(
            action == 5
            and self._front_locked_door_color is not None
            and self._carried_key_color
            == self._front_locked_door_color
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid Unlock returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        self._update(facts)
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "key_found": self._key_found,
                "key_picked_up": self._key_picked_up,
                "door_opened": opened_door,
                "success": success,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _update(self, facts: _Facts) -> None:
        self._carried_key_color = facts.carried_key_color
        self._front_locked_door_color = facts.front_locked_door_color
        self._key_found = self._key_found or facts.key_visible
        self._key_picked_up = (
            self._key_picked_up or facts.carried_key_color is not None
        )


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid Unlock returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Unlock returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid Unlock returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Unlock returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid Unlock returned invalid mission")
    carried_key_color = (
        int(image[3, 6, 1])
        if image[3, 6, 0] == _KEY
        else None
    )
    front_locked_door_color = (
        int(image[3, 5, 1])
        if image[3, 5, 0] == _DOOR
        and image[3, 5, 2] == _LOCKED
        else None
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
        _Facts(
            key_visible=bool(numpy.any(image[:, :, 0] == _KEY)),
            carried_key_color=carried_key_color,
            front_locked_door_color=front_locked_door_color,
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid Unlock returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid Unlock returned non-finite reward")
    return number


__all__ = ["UnlockEnvironment"]
