"""One fresh Gymnasium CliffWalking-v1 Environment per Episode."""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import CliffWalkingConfig

MAX_EPISODE_STEPS = 200
_ACTIONS = frozenset({0, 1, 2, 3})
_ROWS = 4
_COLUMNS = 12


class CliffWalkingEnvironment:
    """The seeded strict adapter around configured CliffWalking-v1."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: CliffWalkingConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not CliffWalkingConfig:
            raise TypeError("config must be CliffWalkingConfig")
        if episode.scenario is not None:
            raise ValueError(
                "CliffWalking configuration belongs in CliffWalkingConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(
                "CliffWalking-v1",
                is_slippery=config.is_slippery,
            ),
        )
        self._steps = 0
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        state, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(state)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()

        state, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "CliffWalking returned invalid termination flags"
            )
        self._steps += 1
        if self._steps >= MAX_EPISODE_STEPS and not terminated:
            truncated = True
        self._done = terminated or truncated
        return Step(
            observation=_observation(state),
            reward=_reward(reward),
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(value: object) -> dict[str, PolicyValue]:
    if type(value) is bool:
        raise RuntimeError("CliffWalking returned an invalid state")
    try:
        state = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise RuntimeError(
            "CliffWalking returned an invalid state"
        ) from error
    if not 0 <= state < _ROWS * _COLUMNS:
        raise RuntimeError("CliffWalking returned an out-of-range state")
    row, column = divmod(state, _COLUMNS)
    return {
        "state": state,
        "row": row,
        "column": column,
        "tile": _tile(row, column),
    }


def _tile(row: int, column: int) -> str:
    if (row, column) == (3, 0):
        return "start"
    if (row, column) == (3, 11):
        return "goal"
    if row == 3 and 1 <= column <= 10:
        return "cliff"
    return "safe"


def _reward(value: object) -> float:
    if type(value) not in {int, float}:
        raise RuntimeError("CliffWalking returned an invalid reward")
    reward = float(cast(int | float, value))
    if not math.isfinite(reward):
        raise RuntimeError("CliffWalking returned a non-finite reward")
    return reward


__all__ = ["CliffWalkingEnvironment", "MAX_EPISODE_STEPS"]
