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
_ACTION_MEANINGS = ("south", "north", "east", "west", "pickup", "dropoff")


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
        self._config = config
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
        self._observation: dict[str, PolicyValue] | None = None
        self._steps = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        state, info = self._environment.reset(seed=self._seed)
        observation = _observation(state, info)
        self._observation = observation
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

        previous_observation = self._observation
        if previous_observation is None:
            raise RuntimeError("Taxi observation is unavailable")
        state, reward, terminated, truncated, info = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Taxi returned invalid termination flags")
        next_observation = _observation(state, info)
        public_reward = _reward(reward)
        self._steps += 1
        metrics = _transition_metrics(
            previous_observation,
            next_observation,
            action,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            sampled_branch_probability=_transition_probability(info),
            config=self._config,
            step_count=self._steps,
        )
        self._observation = next_observation
        self._done = terminated or truncated
        return Step(
            observation=next_observation,
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
    if reward not in {-10.0, -1.0, 20.0}:
        raise RuntimeError("Taxi returned an unknown reward")
    return reward


def _transition_probability(info: object) -> float:
    if not isinstance(info, Mapping) or "prob" not in info:
        raise RuntimeError("Taxi omitted transition probability")
    value = info["prob"]
    if isinstance(value, bool):
        raise RuntimeError("Taxi returned an invalid transition probability")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("Taxi returned an invalid transition probability") from None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("Taxi returned an invalid transition probability")
    return probability


def _transition_metrics(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    action: int,
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    sampled_branch_probability: float,
    config: TaxiConfig,
    step_count: int,
) -> dict[str, PolicyValue]:
    previous_row = _integer_field(previous, "taxi_row")
    previous_column = _integer_field(previous, "taxi_column")
    current_row = _integer_field(current, "taxi_row")
    current_column = _integer_field(current, "taxi_column")
    previous_passenger = _string_field(previous, "passenger_location")
    current_passenger = _string_field(current, "passenger_location")
    previous_destination = _string_field(previous, "destination")
    current_destination = _string_field(current, "destination")
    legal_actions = previous.get("legal_actions")
    if type(legal_actions) is not list:
        raise RuntimeError("Taxi returned invalid advisory Actions")

    expected_probabilities: tuple[float, ...] = (1.0,)
    if config.is_rainy and action < 4:
        expected_probabilities = (
            config.rainy_probability,
            (1.0 - config.rainy_probability) / 2.0,
        )
    if not any(
        math.isclose(
            sampled_branch_probability,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for expected in expected_probabilities
    ):
        raise RuntimeError("Taxi transition probability drifted")

    row_delta = current_row - previous_row
    column_delta = current_column - previous_column
    movement_by_delta = {
        (0, 0): "stayed",
        (1, 0): "south",
        (-1, 0): "north",
        (0, 1): "east",
        (0, -1): "west",
    }
    observed_movement = movement_by_delta.get((row_delta, column_delta))
    if observed_movement is None:
        raise RuntimeError("Taxi returned a non-adjacent movement")

    event: str
    if action < 4:
        event = "movement" if observed_movement != "stayed" else "movement_noop"
    elif action == 4:
        event = (
            "pickup"
            if previous_passenger != "in_taxi" and current_passenger == "in_taxi"
            else "illegal_pickup"
        )
    elif terminated:
        event = "successful_dropoff"
    elif previous_passenger == "in_taxi" and current_passenger != "in_taxi":
        event = "wrong_landmark_dropoff"
    else:
        event = "illegal_dropoff"

    if event.startswith("illegal_") and reward != -10.0:
        raise RuntimeError("Taxi illegal Action reward drifted")
    if event == "successful_dropoff" and reward != 20.0:
        raise RuntimeError("Taxi delivery reward drifted")
    if event not in {"illegal_pickup", "illegal_dropoff", "successful_dropoff"} and reward != -1.0:
        raise RuntimeError("Taxi ordinary reward drifted")

    metrics: dict[str, PolicyValue] = {
        "step_count": step_count,
        "requested_action": _ACTION_MEANINGS[action],
        "action_was_listed_as_state_changing": action in legal_actions,
        "event": event,
        "observed_movement": observed_movement,
        "taxi_position_changed": observed_movement != "stayed",
        "passenger_location_changed": previous_passenger != current_passenger,
        "destination_changed": previous_destination != current_destination,
        "sampled_branch_probability": sampled_branch_probability,
        "state_changed": _integer_field(previous, "state") != _integer_field(current, "state"),
    }
    if previous_destination != current_destination:
        metrics["previous_destination"] = previous_destination
        metrics["new_destination"] = current_destination
    reasons: list[str] = []
    if terminated:
        if event != "successful_dropoff":
            raise RuntimeError("Taxi terminated without successful delivery")
        reasons.append("passenger_delivered")
    if truncated:
        reasons.append("time_limit")
    if reasons:
        metrics["terminal_reason"] = "+".join(reasons)
    return metrics


def _integer_field(observation: dict[str, PolicyValue], name: str) -> int:
    value = observation.get(name)
    if type(value) is not int:
        raise RuntimeError(f"Taxi returned invalid {name}")
    return value


def _string_field(observation: dict[str, PolicyValue], name: str) -> str:
    value = observation.get(name)
    if type(value) is not str:
        raise RuntimeError(f"Taxi returned invalid {name}")
    return value


__all__ = ["TaxiEnvironment"]
