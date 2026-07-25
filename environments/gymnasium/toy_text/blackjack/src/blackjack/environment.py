"""One fresh Gymnasium Blackjack-v1 Environment per Episode."""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import BlackjackConfig

_ACTIONS = frozenset({0, 1})


class BlackjackEnvironment:
    """The seeded strict adapter around configured Blackjack-v1."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: BlackjackConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BlackjackConfig:
            raise TypeError("config must be BlackjackConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Blackjack configuration belongs in BlackjackConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(
                "Blackjack-v1",
                natural=config.natural,
                sab=config.sab,
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
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(observation)

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
            raise RuntimeError("Blackjack returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_observation(observation),
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
    if type(value) is not tuple or len(value) != 3:
        raise RuntimeError("Blackjack returned an invalid observation")
    player_sum = _integer(value[0], name="player sum")
    dealer_showing = _integer(value[1], name="dealer card")
    usable_ace = _boolean(value[2], name="usable-ace flag")
    if not 0 <= player_sum <= 31:
        raise RuntimeError("Blackjack returned an invalid player sum")
    if not 1 <= dealer_showing <= 10:
        raise RuntimeError("Blackjack returned an invalid dealer card")
    return {
        "player_sum": player_sum,
        "dealer_showing": dealer_showing,
        "usable_ace": usable_ace,
    }


def _integer(value: object, *, name: str) -> int:
    if type(value) is bool:
        raise RuntimeError(f"Blackjack returned an invalid {name}")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise RuntimeError(
            f"Blackjack returned an invalid {name}"
        ) from error


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is bool:
        return value
    integer = _integer(value, name=name)
    if integer not in {0, 1}:
        raise RuntimeError(f"Blackjack returned an invalid {name}")
    return bool(integer)


def _reward(value: object) -> float:
    if type(value) not in {int, float}:
        raise RuntimeError("Blackjack returned an invalid reward")
    reward = float(cast(int | float, value))
    if not math.isfinite(reward):
        raise RuntimeError("Blackjack returned a non-finite reward")
    return reward


__all__ = ["BlackjackEnvironment"]
