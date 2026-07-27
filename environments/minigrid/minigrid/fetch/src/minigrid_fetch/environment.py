"""One fresh MiniGrid Fetch Environment per Episode."""

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

from .config import FetchConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_OBJECT_CODES = {"key": 5, "ball": 6}
_MISSION_PREFIXES = (
    "get a ",
    "go get a ",
    "fetch a ",
    "go fetch a ",
    "you must fetch a ",
)


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    target_color: int
    target_type: str
    target_visible: bool
    carried_color: int | None
    carried_type: str | None


class FetchEnvironment:
    """The seeded strict adapter around configured MiniGrid Fetch."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: FetchConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not FetchConfig:
            raise TypeError("config must be FetchConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Fetch configuration belongs in FetchConfig, "
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
            raise RuntimeError("MiniGrid Fetch returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if (facts.target_color, facts.target_type) != self._target:
            raise RuntimeError(
                "MiniGrid Fetch changed target during an Episode"
            )
        self._target_found = self._target_found or facts.target_visible
        picked_up_object = facts.carried_type is not None
        success = bool(terminated and number > 0.0)
        wrong_object = bool(terminated and picked_up_object and not success)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "target_found": self._target_found,
                "picked_up_object": picked_up_object,
                "wrong_object": wrong_object,
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
        raise RuntimeError("MiniGrid Fetch returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Fetch returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid Fetch returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid Fetch returned an invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Fetch returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid Fetch returned an invalid mission")
    target_color, target_type = _target(mission)
    target_code = _OBJECT_CODES[target_type]
    target_visible = bool(
        numpy.any(
            (image[:, :, 0] == target_code)
            & (image[:, :, 1] == target_color)
        )
    )
    carried_code = int(image[3, 6, 0])
    carried_type = next(
        (
            name
            for name, code in _OBJECT_CODES.items()
            if code == carried_code
        ),
        None,
    )
    carried_color = int(image[3, 6, 1]) if carried_type is not None else None
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
            carried_color=carried_color,
            carried_type=carried_type,
        ),
    )


def _target(mission: str) -> tuple[int, str]:
    remainder = next(
        (
            mission.removeprefix(prefix)
            for prefix in _MISSION_PREFIXES
            if mission.startswith(prefix)
        ),
        None,
    )
    if remainder is None:
        raise RuntimeError("MiniGrid Fetch returned an invalid mission")
    parts = remainder.split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] not in _OBJECT_CODES:
        raise RuntimeError("MiniGrid Fetch returned an invalid mission")
    return _COLORS.index(parts[0]), parts[1]


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid Fetch returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid Fetch returned a non-finite {name}")
    return number


__all__ = ["FetchEnvironment"]
