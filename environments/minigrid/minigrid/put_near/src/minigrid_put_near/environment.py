"""One fresh MiniGrid PutNear Environment per Episode."""

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

from .config import PutNearConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_OBJECT_CODES = {"key": 5, "ball": 6, "box": 7}


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    move_object: tuple[int, int]
    target_object: tuple[int, int]
    move_visible: bool
    target_visible: bool
    carried_object: tuple[int, int] | None


class PutNearEnvironment:
    """The seeded strict adapter around configured MiniGrid PutNear."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: PutNearConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not PutNearConfig:
            raise TypeError("config must be PutNearConfig")
        if episode.scenario is not None:
            raise ValueError(
                "PutNear configuration belongs in PutNearConfig, "
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
        self._mission_objects: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._move_found = False
        self._target_found = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._mission_objects = (facts.move_object, facts.target_object)
        self._move_found = facts.move_visible
        self._target_found = facts.target_visible
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
            raise RuntimeError("MiniGrid PutNear returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if (facts.move_object, facts.target_object) != self._mission_objects:
            raise RuntimeError(
                "MiniGrid PutNear changed mission objects during an Episode"
            )
        self._move_found = self._move_found or facts.move_visible
        self._target_found = self._target_found or facts.target_visible
        success = bool(terminated and number > 0.0)
        wrong_object = bool(
            terminated
            and action == 3
            and facts.carried_object is not None
            and facts.carried_object != facts.move_object
        )
        misplaced_object = bool(
            terminated and action == 4 and not success
        )
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "move_object_found": self._move_found,
                "target_object_found": self._target_found,
                "carrying_move_object": (
                    facts.carried_object == facts.move_object
                ),
                "wrong_object": wrong_object,
                "misplaced_object": misplaced_object,
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
        raise RuntimeError("MiniGrid PutNear returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid PutNear returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid PutNear returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid PutNear returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid PutNear returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid PutNear returned an invalid mission")
    move_object, target_object = _mission_objects(mission)
    carried_type = int(image[3, 6, 0])
    carried_object = (
        (carried_type, int(image[3, 6, 1]))
        if carried_type in _OBJECT_CODES.values()
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
            move_object=move_object,
            target_object=target_object,
            move_visible=_visible(image, move_object),
            target_visible=_visible(image, target_object),
            carried_object=carried_object,
        ),
    )


def _mission_objects(
    mission: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    prefix = "put the "
    separator = " near the "
    if not mission.startswith(prefix) or separator not in mission:
        raise RuntimeError("MiniGrid PutNear returned an invalid mission")
    move_text, target_text = mission.removeprefix(prefix).split(separator)
    return _named_object(move_text), _named_object(target_text)


def _named_object(text: str) -> tuple[int, int]:
    parts = text.split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] not in _OBJECT_CODES:
        raise RuntimeError("MiniGrid PutNear returned an invalid mission")
    return _OBJECT_CODES[parts[1]], _COLORS.index(parts[0])


def _visible(image: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]], target: tuple[int, int]) -> bool:
    return bool(
        numpy.any(
            (image[:, :, 0] == target[0])
            & (image[:, :, 1] == target[1])
        )
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid PutNear returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid PutNear returned a non-finite {name}")
    return number


__all__ = ["PutNearEnvironment"]
