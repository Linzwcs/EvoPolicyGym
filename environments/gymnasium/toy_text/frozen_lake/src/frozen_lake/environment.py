"""One fresh Gymnasium FrozenLake Environment per Episode."""

from __future__ import annotations

import operator
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import FrozenLakeConfig

_ACTIONS = frozenset({0, 1, 2, 3})


class FrozenLakeEnvironment:
    """The seeded strict adapter around a configured standard FrozenLake."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: FrozenLakeConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not FrozenLakeConfig:
            raise TypeError("config must be FrozenLakeConfig")
        if episode.scenario is not None:
            raise ValueError(
                "FrozenLake configuration belongs in FrozenLakeConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(
                config.environment_id,
                desc=list(config.layout),
                is_slippery=config.is_slippery,
                success_rate=config.success_rate,
                reward_schedule=(1, 0, 0),
            ),
        )
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
        return _observation(state, self._config)

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
            raise RuntimeError("FrozenLake returned invalid termination flags")
        if type(reward) not in {int, float}:
            raise RuntimeError("FrozenLake returned an invalid reward")
        self._done = terminated or truncated
        return Step(
            observation=_observation(state, self._config),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(
    value: object,
    config: FrozenLakeConfig,
) -> dict[str, PolicyValue]:
    try:
        state = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise RuntimeError("FrozenLake returned an invalid state") from error

    width = len(config.layout[0])
    state_count = width * len(config.layout)
    if not 0 <= state < state_count:
        raise RuntimeError("FrozenLake returned an out-of-range state")
    row, column = divmod(state, width)
    return {
        "state": state,
        "row": row,
        "column": column,
        "tile": config.layout[row][column],
    }


__all__ = ["FrozenLakeEnvironment"]
