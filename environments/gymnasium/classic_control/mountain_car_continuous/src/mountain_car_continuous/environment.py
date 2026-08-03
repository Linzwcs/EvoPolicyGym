"""One fresh Gymnasium Continuous Mountain Car Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

_OBSERVATION_NAMES = ("position", "velocity")
_MAX_EPISODE_STEPS = 999
_MIN_POSITION = -1.2
_MAX_POSITION = 0.6
_MAX_SPEED = 0.07
_GOAL_POSITION = 0.45
_GOAL_VELOCITY = 0.0
_POWER = 0.0015
_GRAVITY_VELOCITY_SCALE = 0.0025
_ACTION_COST_COEFFICIENT = 0.1
_GOAL_BONUS = 100.0


class MountainCarContinuousEnvironment:
    """The seeded strict adapter around Gymnasium MountainCarContinuous-v0."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Continuous Mountain Car does not accept Episode scenarios")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, tuple[float]],
            gymnasium.make("MountainCarContinuous-v0"),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._observation: dict[str, PolicyValue] | None = None
        self._steps = 0
        self._maximum_position = -math.inf
        self._minimum_position = math.inf
        self._cumulative_control_cost = 0.0
        self._cumulative_reward = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        position = _float_field(public_observation, "position")
        velocity = _float_field(public_observation, "velocity")
        if not -0.600001 <= position <= -0.399999 or velocity != 0.0:
            raise RuntimeError("Continuous Mountain Car initial-state semantics drifted")
        self._observation = public_observation
        self._maximum_position = position
        self._minimum_position = position
        self._started = True
        return public_observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if (
            type(action) is not float
            or not math.isfinite(action)
            or not -1.0 <= action <= 1.0
        ):
            raise InvalidAction()

        previous_observation = self._observation
        if previous_observation is None:
            raise RuntimeError("Continuous Mountain Car observation is unavailable")
        observation, reward, terminated, truncated, _ = self._environment.step((action,))
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError(
                "Continuous Mountain Car returned invalid termination flags"
            )
        public_observation = _observation(observation)
        public_reward = _number(reward)
        self._steps += 1
        current_position = _float_field(public_observation, "position")
        self._maximum_position = max(self._maximum_position, current_position)
        self._minimum_position = min(self._minimum_position, current_position)
        control_cost = _ACTION_COST_COEFFICIENT * action**2
        self._cumulative_control_cost += control_cost
        self._cumulative_reward += public_reward
        metrics = _transition_metrics(
            previous_observation,
            public_observation,
            action=action,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            episode_maximum_position=self._maximum_position,
            episode_minimum_position=self._minimum_position,
            cumulative_control_cost=self._cumulative_control_cost,
            cumulative_reward=self._cumulative_reward,
        )
        self._observation = public_observation
        self._done = terminated or truncated
        return Step(
            observation=public_observation,
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
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise RuntimeError(
            "Continuous Mountain Car returned an invalid observation"
        )
    items = tuple(value)
    if len(items) != len(_OBSERVATION_NAMES):
        raise RuntimeError(
            "Continuous Mountain Car returned an invalid observation shape"
        )
    return {
        name: _number(item)
        for name, item in zip(_OBSERVATION_NAMES, items, strict=True)
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(
            "Continuous Mountain Car returned a non-numeric value"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(
            "Continuous Mountain Car returned a non-finite value"
        )
    return number


def _transition_metrics(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    *,
    action: float,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    episode_maximum_position: float,
    episode_minimum_position: float,
    cumulative_control_cost: float,
    cumulative_reward: float,
) -> dict[str, PolicyValue]:
    previous_position = _float_field(previous, "position")
    previous_velocity = _float_field(previous, "velocity")
    position = _float_field(current, "position")
    velocity = _float_field(current, "velocity")
    engine_increment = action * _POWER
    gravity_increment = -math.cos(3.0 * previous_position) * _GRAVITY_VELOCITY_SCALE
    expected_velocity = min(
        max(previous_velocity + engine_increment + gravity_increment, -_MAX_SPEED),
        _MAX_SPEED,
    )
    expected_position = min(
        max(previous_position + expected_velocity, _MIN_POSITION),
        _MAX_POSITION,
    )
    left_wall_collision = expected_position == _MIN_POSITION and expected_velocity < 0.0
    if left_wall_collision:
        expected_velocity = 0.0
    position_matches = math.isclose(
        position,
        expected_position,
        rel_tol=0.0,
        abs_tol=2e-6,
    )
    velocity_matches = math.isclose(
        velocity,
        expected_velocity,
        rel_tol=0.0,
        abs_tol=2e-6,
    )
    if not position_matches or not velocity_matches:
        raise RuntimeError("Continuous Mountain Car transition does not match public dynamics")
    goal_reached = position >= _GOAL_POSITION and velocity >= _GOAL_VELOCITY
    if terminated != goal_reached:
        raise RuntimeError("Continuous Mountain Car goal termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Continuous Mountain Car time-limit semantics drifted")
    control_cost = _ACTION_COST_COEFFICIENT * action**2
    goal_bonus = _GOAL_BONUS if terminated else 0.0
    expected_reward = goal_bonus - control_cost
    if not math.isclose(reward, expected_reward, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Continuous Mountain Car reward semantics drifted")
    direction_before = _velocity_direction(previous_velocity)
    direction_after = _velocity_direction(velocity)
    reasons: list[str] = []
    if terminated:
        reasons.append("goal_reached")
    if truncated:
        reasons.append("time_limit")
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "requested_force": action,
        "force_direction": _signed_direction(action),
        "engine_velocity_increment": engine_increment,
        "gravity_velocity_increment": gravity_increment,
        "position_before": previous_position,
        "position_after": position,
        "position_change": position - previous_position,
        "velocity_before": previous_velocity,
        "velocity_after": velocity,
        "velocity_change": velocity - previous_velocity,
        "velocity_direction_before": direction_before,
        "velocity_direction_after": direction_after,
        "direction_reversed": (
            direction_before != "stationary"
            and direction_after != "stationary"
            and direction_before != direction_after
        ),
        "left_wall_collision": left_wall_collision,
        "terrain_height": math.sin(3.0 * position) * 0.45 + 0.55,
        "control_cost": control_cost,
        "goal_bonus": goal_bonus,
        "reward_from_public_terms": expected_reward,
        "cumulative_control_cost": cumulative_control_cost,
        "cumulative_return": cumulative_reward,
        "goal_position": _GOAL_POSITION,
        "distance_to_goal_position": max(_GOAL_POSITION - position, 0.0),
        "goal_velocity": _GOAL_VELOCITY,
        "goal_reached": goal_reached,
        "episode_maximum_position": episode_maximum_position,
        "episode_minimum_position": episode_minimum_position,
        "terminal_reason": "+".join(reasons) if reasons else "none",
    }


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Continuous Mountain Car returned invalid {name}")
    return value


def _velocity_direction(velocity: float) -> str:
    return _signed_direction(velocity)


def _signed_direction(value: float) -> str:
    if value > 0.0:
        return "right"
    if value < 0.0:
        return "left"
    return "stationary"


__all__ = ["MountainCarContinuousEnvironment"]
