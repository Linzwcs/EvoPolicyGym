"""One fresh Gymnasium Taxi-v4 Environment per Episode."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import TaxiConfig

_ACTIONS = frozenset({0, 1, 2, 3, 4, 5})
_LANDMARK_NAMES = ("red", "green", "yellow", "blue")


class TaxiEnvironment:
    """The seeded strict adapter around a configured Taxi-v4."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: TaxiConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not TaxiConfig:
            raise TypeError("config must be TaxiConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Taxi configuration belongs in TaxiConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(
                "Taxi-v4",
                is_rainy=config.is_rainy,
                fickle_passenger=config.fickle_passenger,
                rainy_probability=config.rainy_probability,
                fickle_probability=config.fickle_probability,
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
        state, info = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(state, info)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()

        state, reward, terminated, truncated, info = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Taxi returned invalid termination flags")
        self._done = terminated or truncated
        return Step(
            observation=_observation(state, info),
            reward=_reward(reward),
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
    info: object,
) -> dict[str, PolicyValue]:
    state = _state(value)
    taxi_row, taxi_column, passenger_index, destination_index = _decode(state)
    passenger_location = (
        _LANDMARK_NAMES[passenger_index]
        if passenger_index < len(_LANDMARK_NAMES)
        else "in_taxi"
    )
    return {
        "state": state,
        "taxi_row": taxi_row,
        "taxi_column": taxi_column,
        "passenger_location": passenger_location,
        "destination": _LANDMARK_NAMES[destination_index],
        "legal_actions": _legal_actions(info),
    }


def _state(value: object) -> int:
    if type(value) is bool:
        raise RuntimeError("Taxi returned an invalid state")
    try:
        state = operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise RuntimeError("Taxi returned an invalid state") from error
    if not 0 <= state < 500:
        raise RuntimeError("Taxi returned an out-of-range state")
    return state


def _decode(state: int) -> tuple[int, int, int, int]:
    destination = state % 4
    remaining = state // 4
    passenger = remaining % 5
    remaining //= 5
    taxi_column = remaining % 5
    taxi_row = remaining // 5
    return taxi_row, taxi_column, passenger, destination


def _legal_actions(info: object) -> list[PolicyValue]:
    if not isinstance(info, Mapping):
        raise RuntimeError("Taxi returned invalid public info")
    mask = info.get("action_mask")
    if isinstance(mask, (str, bytes)) or not isinstance(mask, Iterable):
        raise RuntimeError("Taxi returned an invalid action mask")
    flags = tuple(mask)
    if len(flags) != 6:
        raise RuntimeError("Taxi returned an invalid action mask")

    actions: list[PolicyValue] = []
    for action, value in enumerate(flags):
        if type(value) is bool:
            flag = int(value)
        else:
            try:
                flag = operator.index(cast(SupportsIndex, value))
            except TypeError as error:
                raise RuntimeError(
                    "Taxi returned an invalid action mask"
                ) from error
        if flag not in {0, 1}:
            raise RuntimeError("Taxi returned an invalid action mask")
        if flag == 1:
            actions.append(action)
    return actions


def _reward(value: object) -> float:
    if type(value) not in {int, float}:
        raise RuntimeError("Taxi returned an invalid reward")
    reward = float(cast(int | float, value))
    if not math.isfinite(reward):
        raise RuntimeError("Taxi returned a non-finite reward")
    return reward


__all__ = ["TaxiEnvironment"]
