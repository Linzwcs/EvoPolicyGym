"""One fresh Gymnasium CliffWalking-v1 Environment per Episode."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import SupportsIndex, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .config import CliffWalkingConfig

MAX_EPISODE_STEPS = 200
_ACTIONS = frozenset({0, 1, 2, 3})
_ROWS = 4
_COLUMNS = 12
_ACTION_MEANINGS = ("up", "right", "down", "left")
_ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_START_STATE = 36
_GOAL_STATE = 47


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
        self._config = config
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
        self._state: int | None = None

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        state, _ = self._environment.reset(seed=self._seed)
        observation = _observation(state)
        public_state = observation["state"]
        if type(public_state) is not int:
            raise RuntimeError("CliffWalking returned an invalid state")
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
            raise RuntimeError("CliffWalking state is unavailable")
        state, reward, terminated, truncated, info = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "CliffWalking returned invalid termination flags"
            )
        self._steps += 1
        if self._steps >= MAX_EPISODE_STEPS and not terminated:
            truncated = True
        observation = _observation(state)
        public_state = observation["state"]
        if type(public_state) is not int:
            raise RuntimeError("CliffWalking returned an invalid state")
        public_reward = _reward(reward)
        metrics = _transition_metrics(
            previous_state,
            public_state,
            action,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            sampled_branch_probability=_transition_probability(info),
            config=self._config,
            step_count=self._steps,
        )
        self._state = public_state
        self._done = terminated or truncated
        return Step(
            observation=observation,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
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
        raise RuntimeError("CliffWalking returned a transient cliff state")
    return "safe"


def _reward(value: object) -> float:
    if type(value) not in {int, float}:
        raise RuntimeError("CliffWalking returned an invalid reward")
    reward = float(cast(int | float, value))
    if not math.isfinite(reward):
        raise RuntimeError("CliffWalking returned a non-finite reward")
    if reward not in {-100.0, -1.0}:
        raise RuntimeError("CliffWalking returned an unknown reward")
    return reward


def _transition_probability(info: object) -> float:
    if not isinstance(info, Mapping) or "prob" not in info:
        raise RuntimeError("CliffWalking omitted transition probability")
    value = info["prob"]
    if isinstance(value, bool):
        raise RuntimeError("CliffWalking returned an invalid transition probability")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(
            "CliffWalking returned an invalid transition probability"
        ) from None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("CliffWalking returned an invalid transition probability")
    return probability


def _transition_metrics(
    previous_state: int,
    next_state: int,
    action: int,
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    sampled_branch_probability: float,
    config: CliffWalkingConfig,
    step_count: int,
) -> dict[str, PolicyValue]:
    branch_probability = 1.0 / 3.0 if config.is_slippery else 1.0
    directions = (
        ((action - 1) % 4, action, (action + 1) % 4)
        if config.is_slippery
        else (action,)
    )
    matching_directions = [
        direction
        for direction in directions
        if _outcome(previous_state, direction) == (next_state, reward, terminated)
    ]
    if (
        not matching_directions
        or not math.isclose(
            sampled_branch_probability,
            branch_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("CliffWalking transition does not match configured dynamics")

    previous_row, previous_column = divmod(previous_state, _COLUMNS)
    next_row, next_column = divmod(next_state, _COLUMNS)
    if reward == -100.0:
        event = "cliff_fall"
        observed_movement = "cliff_then_reset_to_start"
    elif terminated:
        event = "goal_reached"
        observed_movement = _movement_name(
            next_row - previous_row,
            next_column - previous_column,
        )
    elif next_state == previous_state:
        event = "boundary_noop"
        observed_movement = "stayed"
    else:
        event = "movement"
        observed_movement = _movement_name(
            next_row - previous_row,
            next_column - previous_column,
        )
    metrics: dict[str, PolicyValue] = {
        "step_count": step_count,
        "requested_direction": _ACTION_MEANINGS[action],
        "event": event,
        "observed_movement": observed_movement,
        "possible_sampled_directions": [
            _ACTION_MEANINGS[direction] for direction in matching_directions
        ],
        "sampled_branch_probability": sampled_branch_probability,
        "observable_outcome_probability": (
            branch_probability * len(matching_directions)
        ),
        "fell_from_cliff": reward == -100.0,
    }
    reasons: list[str] = []
    if terminated:
        if next_state != _GOAL_STATE:
            raise RuntimeError("CliffWalking terminated away from the goal")
        reasons.append("goal_reached")
    if truncated:
        reasons.append("time_limit")
    if reasons:
        metrics["terminal_reason"] = "+".join(reasons)
    return metrics


def _outcome(state: int, direction: int) -> tuple[int, float, bool]:
    row, column = divmod(state, _COLUMNS)
    row_delta, column_delta = _ACTION_DELTAS[direction]
    row = min(max(row + row_delta, 0), _ROWS - 1)
    column = min(max(column + column_delta, 0), _COLUMNS - 1)
    if row == 3 and 1 <= column <= 10:
        return _START_STATE, -100.0, False
    next_state = row * _COLUMNS + column
    return next_state, -1.0, next_state == _GOAL_STATE


def _movement_name(row_delta: int, column_delta: int) -> str:
    try:
        return _ACTION_MEANINGS[_ACTION_DELTAS.index((row_delta, column_delta))]
    except ValueError:
        raise RuntimeError("CliffWalking returned a non-adjacent movement") from None


__all__ = ["CliffWalkingEnvironment", "MAX_EPISODE_STEPS"]
