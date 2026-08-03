"""One fresh Gymnasium FrozenLake Environment per Episode."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import FrozenLakeConfig

_ACTIONS = frozenset({0, 1, 2, 3})
_ACTION_MEANINGS = ("left", "down", "right", "up")
_ACTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


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
        self._state: int | None = None
        self._steps = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        state, _ = self._environment.reset(seed=self._seed)
        observation = _observation(state, self._config)
        public_state = observation["state"]
        if type(public_state) is not int:
            raise RuntimeError("FrozenLake returned an invalid state")
        self._state = public_state
        self._started = True
        return observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()

        previous_state = self._state
        if previous_state is None:
            raise RuntimeError("FrozenLake state is unavailable")
        state, reward, terminated, truncated, info = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("FrozenLake returned invalid termination flags")
        if type(reward) not in {int, float}:
            raise RuntimeError("FrozenLake returned an invalid reward")
        observation = _observation(state, self._config)
        public_state = observation["state"]
        if type(public_state) is not int:
            raise RuntimeError("FrozenLake returned an invalid state")
        self._steps += 1
        metrics = _transition_metrics(
            previous_state,
            public_state,
            action,
            config=self._config,
            sampled_branch_probability=_sampled_branch_probability(info),
        )
        metrics["step_count"] = self._steps
        if terminated or truncated:
            metrics["terminal_reason"] = _terminal_reason(
                observation,
                terminated=terminated,
                truncated=truncated,
            )
        self._state = public_state
        self._done = terminated or truncated
        return Step(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
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


def _sampled_branch_probability(info: object) -> float:
    if not isinstance(info, Mapping) or "prob" not in info:
        raise RuntimeError("FrozenLake omitted sampled transition probability")
    value = info["prob"]
    if isinstance(value, bool):
        raise RuntimeError("FrozenLake returned an invalid transition probability")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("FrozenLake returned an invalid transition probability") from None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("FrozenLake returned an invalid transition probability")
    return probability


def _transition_metrics(
    previous_state: int,
    next_state: int,
    action: int,
    *,
    config: FrozenLakeConfig,
    sampled_branch_probability: float,
) -> dict[str, PolicyValue]:
    branches: tuple[tuple[int, float], ...]
    if config.is_slippery:
        failure_probability = (1.0 - config.success_rate) / 2.0
        branches = (
            ((action - 1) % 4, failure_probability),
            (action, config.success_rate),
            ((action + 1) % 4, failure_probability),
        )
    else:
        branches = ((action, 1.0),)
    matching = tuple(
        (direction, probability)
        for direction, probability in branches
        if probability > 0.0
        and _move(previous_state, direction, config=config) == next_state
    )
    if not matching or not any(
        math.isclose(
            sampled_branch_probability,
            probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for _, probability in matching
    ):
        raise RuntimeError("FrozenLake transition does not match configured dynamics")
    rows = len(config.layout)
    columns = len(config.layout[0])
    previous_row, previous_column = divmod(previous_state, columns)
    next_row, next_column = divmod(next_state, columns)
    delta = (next_row - previous_row, next_column - previous_column)
    if delta == (0, 0):
        observed_movement = "stayed"
    else:
        try:
            observed_movement = _ACTION_MEANINGS[_ACTION_DELTAS.index(delta)]
        except ValueError:
            raise RuntimeError("FrozenLake returned a non-adjacent transition") from None
    if not (0 <= next_row < rows and 0 <= next_column < columns):
        raise RuntimeError("FrozenLake returned an invalid coordinate transition")
    return {
        "requested_direction": _ACTION_MEANINGS[action],
        "observed_movement": observed_movement,
        "possible_sampled_directions": [
            _ACTION_MEANINGS[direction] for direction, _ in matching
        ],
        "sampled_branch_probability": sampled_branch_probability,
        "observable_outcome_probability": sum(
            probability for _, probability in matching
        ),
    }


def _move(state: int, action: int, *, config: FrozenLakeConfig) -> int:
    rows = len(config.layout)
    columns = len(config.layout[0])
    row, column = divmod(state, columns)
    row_delta, column_delta = _ACTION_DELTAS[action]
    row = min(max(row + row_delta, 0), rows - 1)
    column = min(max(column + column_delta, 0), columns - 1)
    return row * columns + column


def _terminal_reason(
    observation: dict[str, PolicyValue],
    *,
    terminated: bool,
    truncated: bool,
) -> str:
    tile = observation["tile"]
    reasons: list[str] = []
    if terminated and tile == "G":
        reasons.append("goal")
    elif terminated and tile == "H":
        reasons.append("hole")
    elif terminated:
        raise RuntimeError("FrozenLake terminated on a nonterminal tile")
    if truncated:
        reasons.append("time_limit")
    if not reasons:
        raise RuntimeError("FrozenLake completed without a public reason")
    return "+".join(reasons)


__all__ = ["FrozenLakeEnvironment"]
