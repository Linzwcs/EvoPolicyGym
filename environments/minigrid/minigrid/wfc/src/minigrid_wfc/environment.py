"""One fresh MiniGrid WFC Environment per Episode."""

from __future__ import annotations

import hashlib
import math
import operator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib.resources import as_file, files
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from minigrid.envs.wfc.config import WFC_PRESETS_ALL
from minigrid.envs.wfc.wfcenv import WFCEnv

from .config import WFCConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_GOAL = 8
_GENERATION_ATTEMPTS = 8
_RETRY_DOMAIN = b"evopolicygym-minigrid-wfc/generation-retry/v1\0"


@dataclass(frozen=True, slots=True)
class _Facts:
    goal_visible: bool


class WFCEnvironment:
    """Strict seeded adapter around a configured WFC registration."""

    def __init__(self, episode: EpisodeSpec, *, config: WFCConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not WFCConfig:
            raise TypeError("config must be WFCConfig")
        if episode.scenario is not None:
            raise ValueError("WFC configuration belongs in WFCConfig")
        self._seed = episode.environment_seed
        self._resources = ExitStack()
        upstream = WFC_PRESETS_ALL[config.profile]
        resource = files("minigrid_wfc").joinpath(
            "patterns",
            upstream.pattern_path.name,
        )
        pattern_path = self._resources.enter_context(as_file(resource))
        generation = replace(upstream, pattern_path=pattern_path)
        try:
            self._environment = cast(
                gymnasium.Env[object, int],
                WFCEnv(
                    wfc_config=generation,
                    size=config.size,
                    ensure_connected=True,
                    max_steps=config.max_episode_steps,
                ),
            )
        except BaseException:
            self._resources.close()
            raise
        self._started = False
        self._done = False
        self._closed = False
        self._goal_found = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation: object | None = None
        for attempt in range(_GENERATION_ATTEMPTS):
            try:
                observation, _ = self._environment.reset(
                    seed=_generation_seed(self._seed, attempt)
                )
                break
            except RuntimeError as error:
                if "Could not generate a valid pattern" not in str(error):
                    raise
        if observation is None:
            raise RuntimeError(
                "MiniGrid WFC exhausted deterministic generation retries"
            )
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
            raise RuntimeError("MiniGrid WFC returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        self._goal_found = self._goal_found or facts.goal_visible
        success = bool(terminated and number > 0.0)
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "goal_found": self._goal_found,
                "success": success,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._environment.close()
        finally:
            self._resources.close()
            self._closed = True


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid WFC returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid WFC returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid WFC returned invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid WFC returned invalid direction")
    mission = value["mission"]
    if (
        type(mission) is not str
        or mission != "traverse the maze to get to the goal"
    ):
        raise RuntimeError("MiniGrid WFC returned invalid mission")
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
        _Facts(goal_visible=bool(numpy.any(image[:, :, 0] == _GOAL))),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid WFC returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid WFC returned non-finite reward")
    return number


def _generation_seed(seed: int, attempt: int) -> int:
    if attempt == 0:
        return seed
    digest = hashlib.sha256()
    digest.update(_RETRY_DOMAIN)
    digest.update(seed.to_bytes(8, "big"))
    digest.update(attempt.to_bytes(1, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


__all__ = ["WFCEnvironment"]
