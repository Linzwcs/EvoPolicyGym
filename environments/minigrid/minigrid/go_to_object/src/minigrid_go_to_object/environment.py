"""One fresh MiniGrid GoToObject Environment per Episode."""

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

from .config import GoToObjectConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_OBJECT_CODES = {"key": 5, "ball": 6, "box": 7}


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    target_color: int
    target_type: str
    target_visible: bool


class GoToObjectEnvironment:
    """The seeded strict adapter around configured MiniGrid GoToObject."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: GoToObjectConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not GoToObjectConfig:
            raise TypeError("config must be GoToObjectConfig")
        if episode.scenario is not None:
            raise ValueError(
                "GoToObject configuration belongs in GoToObjectConfig, "
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
        self._target: tuple[int, str] | None = None
        self._target_found = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        self._target = (facts.target_color, facts.target_type)
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
            raise RuntimeError("MiniGrid GoToObject returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if (facts.target_color, facts.target_type) != self._target:
            raise RuntimeError(
                "MiniGrid GoToObject changed target during an Episode"
            )
        self._target_found = self._target_found or facts.target_visible
        success = bool(terminated and number > 0.0)
        wrong_completion = bool(
            terminated and action in {5, 6} and not success
        )
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "target_found": self._target_found,
                "wrong_completion": wrong_completion,
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
        raise RuntimeError("MiniGrid GoToObject returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid GoToObject returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid GoToObject returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid GoToObject returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid GoToObject returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid GoToObject returned an invalid mission")
    target_color, target_type = _target(mission)
    target_code = _OBJECT_CODES[target_type]
    target_visible = bool(
        numpy.any(
            (image[:, :, 0] == target_code)
            & (image[:, :, 1] == target_color)
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
        _ObservationFacts(
            target_color=target_color,
            target_type=target_type,
            target_visible=target_visible,
        ),
    )


def _target(mission: str) -> tuple[int, str]:
    prefix = "go to the "
    if not mission.startswith(prefix):
        raise RuntimeError("MiniGrid GoToObject returned an invalid mission")
    remainder = mission.removeprefix(prefix)
    parts = remainder.split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] not in _OBJECT_CODES:
        raise RuntimeError("MiniGrid GoToObject returned an invalid mission")
    return _COLORS.index(parts[0]), parts[1]


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid GoToObject returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid GoToObject returned a non-finite {name}")
    return number


__all__ = ["GoToObjectEnvironment"]
