"""One fresh MiniGrid RedBlueDoors Environment per Episode."""

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

from .config import RedBlueDoorsConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "open the red door then the blue door"
_DOOR = 4
_RED = 0
_BLUE = 2
_CLOSED = 1


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    red_visible: bool
    blue_visible: bool
    front_door: tuple[int, int] | None


class RedBlueDoorsEnvironment:
    """Strict seeded adapter around MiniGrid RedBlueDoors."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: RedBlueDoorsConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not RedBlueDoorsConfig:
            raise TypeError("config must be RedBlueDoorsConfig")
        if episode.scenario is not None:
            raise ValueError(
                "RedBlueDoors configuration belongs in RedBlueDoorsConfig, "
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
        self._front_door: tuple[int, int] | None = None
        self._red_found = False
        self._blue_found = False
        self._red_opened = False

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
        opened_red = bool(
            action == 5
            and self._front_door == (_RED, _CLOSED)
        )
        opened_blue_before_red = bool(
            action == 5
            and self._front_door == (_BLUE, _CLOSED)
            and not self._red_opened
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "MiniGrid RedBlueDoors returned invalid termination flags"
            )
        number = _number(reward)
        public, facts = _observation(observation)
        self._red_opened = self._red_opened or opened_red
        self._update(facts)
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "red_door_found": self._red_found,
                "blue_door_found": self._blue_found,
                "red_door_opened": self._red_opened,
                "blue_opened_before_red": opened_blue_before_red,
                "success": success,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _update(self, facts: _ObservationFacts) -> None:
        self._front_door = facts.front_door
        self._red_found = self._red_found or facts.red_visible
        self._blue_found = self._blue_found or facts.blue_visible


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError(
            "MiniGrid RedBlueDoors returned an invalid observation"
        )
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid RedBlueDoors returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError(
            "MiniGrid RedBlueDoors returned out-of-range image codes"
        )
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid RedBlueDoors returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError(
            "MiniGrid RedBlueDoors returned an invalid direction"
        )
    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError(
            "MiniGrid RedBlueDoors returned an invalid mission"
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
        _ObservationFacts(
            red_visible=bool(
                numpy.any(
                    (image[:, :, 0] == _DOOR)
                    & (image[:, :, 1] == _RED)
                )
            ),
            blue_visible=bool(
                numpy.any(
                    (image[:, :, 0] == _DOOR)
                    & (image[:, :, 1] == _BLUE)
                )
            ),
            front_door=front_door,
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid RedBlueDoors returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid RedBlueDoors returned non-finite reward")
    return number


__all__ = ["RedBlueDoorsEnvironment"]
