"""One fresh MiniGrid KeyCorridor Environment per Episode."""

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

from .config import KeyCorridorConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_DOOR = 4
_KEY = 5
_BALL = 6
_LOCKED = 2


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    target_color: int
    carrying_key_color: int | None
    key_visible: bool
    target_object_visible: bool
    front_locked_door_color: int | None


class KeyCorridorEnvironment:
    """The seeded strict adapter around configured MiniGrid KeyCorridor."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: KeyCorridorConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not KeyCorridorConfig:
            raise TypeError("config must be KeyCorridorConfig")
        if episode.scenario is not None:
            raise ValueError(
                "KeyCorridor configuration belongs in KeyCorridorConfig, "
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
        self._target_color: int | None = None
        self._carrying_key_color: int | None = None
        self._front_locked_door_color: int | None = None
        self._found_key = False
        self._picked_up_key = False
        self._opened_target_door = False
        self._found_target_object = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._target_color = facts.target_color
        self._update_facts(facts)
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

        opened_target_door = bool(
            action == 5
            and self._front_locked_door_color is not None
            and self._carrying_key_color
            == self._front_locked_door_color
        )
        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "MiniGrid KeyCorridor returned invalid termination flags"
            )
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if facts.target_color != self._target_color:
            raise RuntimeError(
                "MiniGrid KeyCorridor changed target color during an Episode"
            )
        self._opened_target_door = (
            self._opened_target_door or opened_target_door
        )
        self._update_facts(facts)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "found_key": self._found_key,
                "carrying_key": self._carrying_key_color is not None,
                "picked_up_key": self._picked_up_key,
                "opened_target_door": self._opened_target_door,
                "found_target_object": self._found_target_object,
                "success": bool(terminated and number > 0.0),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _update_facts(self, facts: _ObservationFacts) -> None:
        self._carrying_key_color = facts.carrying_key_color
        self._front_locked_door_color = facts.front_locked_door_color
        self._found_key = self._found_key or facts.key_visible
        self._picked_up_key = (
            self._picked_up_key or facts.carrying_key_color is not None
        )
        self._found_target_object = (
            self._found_target_object or facts.target_object_visible
        )


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError(
            "MiniGrid KeyCorridor returned out-of-range image codes"
        )

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid KeyCorridor returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError(
            "MiniGrid KeyCorridor returned an invalid direction"
        )

    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid mission")
    target_color = _target_color(mission)

    carrying_key_color = (
        int(image[3, 6, 1])
        if image[3, 6, 0] == _KEY
        else None
    )
    key_visible = bool(numpy.any(image[:, :, 0] == _KEY))
    target_object_visible = bool(
        numpy.any(
            (image[:, :, 0] == _BALL)
            & (image[:, :, 1] == target_color)
        )
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
        _ObservationFacts(
            target_color=target_color,
            carrying_key_color=carrying_key_color,
            key_visible=key_visible,
            target_object_visible=target_object_visible,
            front_locked_door_color=front_locked_door_color,
        ),
    )


def _target_color(mission: str) -> int:
    prefix = "pick up the "
    suffix = " ball"
    if not mission.startswith(prefix) or not mission.endswith(suffix):
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid mission")
    color = mission[len(prefix) : -len(suffix)]
    if color not in _COLORS:
        raise RuntimeError("MiniGrid KeyCorridor returned an invalid mission")
    return _COLORS.index(color)


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid KeyCorridor returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            f"MiniGrid KeyCorridor returned a non-finite {name}"
        )
    return number


__all__ = ["KeyCorridorEnvironment"]
