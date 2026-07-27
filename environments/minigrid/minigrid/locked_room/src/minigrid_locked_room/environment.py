"""One fresh MiniGrid LockedRoom Environment per Episode."""

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

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_DOOR = 4
_KEY = 5
_GOAL = 8
_CLOSED = 1
_LOCKED = 2


@dataclass(frozen=True, slots=True)
class _Facts:
    target_color: int
    target_key_visible: bool
    carried_key_color: int | None
    goal_visible: bool
    front_door: tuple[int, int] | None


class LockedRoomEnvironment:
    """Strict seeded adapter around MiniGrid LockedRoom."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("LockedRoom has no Episode scenario overrides")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make("MiniGrid-LockedRoom-v0"),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target_color: int | None = None
        self._carried_key_color: int | None = None
        self._front_door: tuple[int, int] | None = None
        self._key_found = False
        self._key_picked_up = False
        self._target_door_opened = False
        self._goal_found = False
        self._opened_doors = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._target_color = facts.target_color
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
            and self._front_door is not None
            and (
                self._front_door[1] == _CLOSED
                or (
                    self._front_door[1] == _LOCKED
                    and self._carried_key_color == self._front_door[0]
                )
            )
        )
        opened_target = bool(
            opened_door
            and self._front_door is not None
            and self._front_door[0] == self._target_color
            and self._front_door[1] == _LOCKED
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid LockedRoom returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        if facts.target_color != self._target_color:
            raise RuntimeError("MiniGrid LockedRoom changed mission target")
        if opened_door:
            self._opened_doors += 1
        self._target_door_opened = (
            self._target_door_opened or opened_target
        )
        self._update(facts)
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "opened_doors": self._opened_doors,
                "key_found": self._key_found,
                "key_picked_up": self._key_picked_up,
                "target_door_opened": self._target_door_opened,
                "goal_found": self._goal_found,
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
        self._front_door = facts.front_door
        self._key_found = self._key_found or facts.target_key_visible
        self._key_picked_up = (
            self._key_picked_up
            or facts.carried_key_color == self._target_color
        )
        self._goal_found = self._goal_found or facts.goal_visible


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid LockedRoom returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid LockedRoom returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid LockedRoom returned invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid LockedRoom returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    target_color = _target_color(mission)
    carried_key_color = (
        int(image[3, 6, 1])
        if image[3, 6, 0] == _KEY
        else None
    )
    front_door = (
        (int(image[3, 5, 1]), int(image[3, 5, 2]))
        if image[3, 5, 0] == _DOOR
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
            target_color=target_color,
            target_key_visible=bool(
                numpy.any(
                    (image[:, :, 0] == _KEY)
                    & (image[:, :, 1] == target_color)
                )
            ),
            carried_key_color=carried_key_color,
            goal_visible=bool(numpy.any(image[:, :, 0] == _GOAL)),
            front_door=front_door,
        ),
    )


def _target_color(mission: str) -> int:
    prefix = "get the "
    suffix = " key from the "
    if not mission.startswith(prefix) or suffix not in mission:
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    color = mission.removeprefix(prefix).split(suffix, maxsplit=1)[0]
    expected = f"unlock the {color} door and go to the goal"
    if color not in _COLORS or expected not in mission:
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    return _COLORS.index(color)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid LockedRoom returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid LockedRoom returned non-finite reward")
    return number


__all__ = ["LockedRoomEnvironment"]
