"""One fresh ObstructedMaze Environment per Episode."""

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

from .config import ObstructedMazeConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_DOOR = 4
_KEY = 5
_BALL = 6
_BOX = 7
_LOCKED = 2


@dataclass(frozen=True, slots=True)
class _Facts:
    target: tuple[int, int]
    carried: tuple[int, int] | None
    key_visible: bool
    blocker_visible: bool
    target_visible: bool
    front_object: int
    front_locked_door_color: int | None


class ObstructedMazeEnvironment:
    """Strict seeded adapter around the upstream environment."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: ObstructedMazeConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not ObstructedMazeConfig:
            raise TypeError("config must be ObstructedMazeConfig")
        if episode.scenario is not None:
            raise ValueError(
                "ObstructedMaze configuration belongs in "
                "ObstructedMazeConfig"
            )
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target: tuple[int, int] | None = None
        self._front_object = 0
        self._front_locked_door_color: int | None = None
        self._carried: tuple[int, int] | None = None
        self._key_found = False
        self._blocker_found = False
        self._target_found = False
        self._blocker_moved = False
        self._key_picked_up = False
        self._door_opened = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._target = facts.target
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
        moved_blocker = bool(
            action == 3
            and self._carried is None
            and self._front_object_is_blocker()
        )
        picked_key = bool(
            action == 3
            and self._carried is None
            and self._front_object_is_key()
        )
        opened_door = bool(
            action == 5
            and self._front_locked_door_color is not None
            and self._carried
            == (_KEY, self._front_locked_door_color)
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "MiniGrid ObstructedMaze returned invalid flags"
            )
        number = _number(reward)
        public, facts = _observation(observation)
        if facts.target != self._target:
            raise RuntimeError(
                "MiniGrid ObstructedMaze changed its mission"
            )
        self._blocker_moved = self._blocker_moved or moved_blocker
        self._key_picked_up = self._key_picked_up or picked_key
        self._door_opened = self._door_opened or opened_door
        self._update(facts)
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "blocker_found": self._blocker_found,
                "blocker_moved": self._blocker_moved,
                "key_found": self._key_found,
                "key_picked_up": self._key_picked_up,
                "door_opened": self._door_opened,
                "target_found": self._target_found,
                "success": success,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _update(self, facts: _Facts) -> None:
        self._carried = facts.carried
        self._front_object = facts.front_object
        self._front_locked_door_color = facts.front_locked_door_color
        self._key_found = self._key_found or facts.key_visible
        self._blocker_found = self._blocker_found or facts.blocker_visible
        self._target_found = self._target_found or facts.target_visible

    def _front_object_is_blocker(self) -> bool:
        return self._front_object == _BALL

    def _front_object_is_key(self) -> bool:
        return self._front_object == _KEY


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid observation"
        )
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid image"
        )
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid direction"
        )
    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid mission"
        )
    target = _target(mission)
    carried_code = int(image[3, 6, 0])
    carried = (
        (carried_code, int(image[3, 6, 1]))
        if carried_code in {_KEY, _BALL, _BOX}
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
            target=target,
            carried=carried,
            key_visible=bool(numpy.any(image[:, :, 0] == _KEY)),
            blocker_visible=bool(numpy.any(image[:, :, 0] == _BALL)),
            target_visible=bool(
                numpy.any(
                    (image[:, :, 0] == target[0])
                    & (image[:, :, 1] == target[1])
                )
            ),
            front_object=int(image[3, 5, 0]),
            front_locked_door_color=front_locked_door_color,
        ),
    )


def _target(mission: str) -> tuple[int, int]:
    prefix = "pick up the "
    if not mission.startswith(prefix):
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid mission"
        )
    parts = mission.removeprefix(prefix).split(" ")
    kinds = {"ball": _BALL}
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] not in kinds:
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid mission"
        )
    return kinds[parts[1]], _COLORS.index(parts[0])


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned invalid reward"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            "MiniGrid ObstructedMaze returned non-finite reward"
        )
    return number


__all__ = ["ObstructedMazeEnvironment"]
