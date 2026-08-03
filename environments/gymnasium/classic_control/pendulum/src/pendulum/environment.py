"""One fresh Gymnasium Pendulum Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

_OBSERVATION_NAMES = (
    "cos_theta",
    "sin_theta",
    "theta_angular_velocity",
)
_MAX_EPISODE_STEPS = 200
_MAX_SPEED = 8.0
_MAX_TORQUE = 2.0
_SECONDS_PER_STEP = 0.05
_GRAVITY = 10.0
_MASS = 1.0
_LENGTH = 1.0
_ANGLE_COST_COEFFICIENT = 1.0
_ANGULAR_VELOCITY_COST_COEFFICIENT = 0.1
_TORQUE_COST_COEFFICIENT = 0.001


class PendulumEnvironment:
    """The seeded strict adapter around Gymnasium Pendulum-v1."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Pendulum does not accept Episode scenarios")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, tuple[float]],
            gymnasium.make("Pendulum-v1"),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._observation: dict[str, PolicyValue] | None = None
        self._steps = 0
        self._cumulative_angle_cost = 0.0
        self._cumulative_angular_velocity_cost = 0.0
        self._cumulative_torque_cost = 0.0
        self._cumulative_reward = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public_observation = _observation(observation)
        angle, angular_velocity = _state(public_observation)
        if not -math.pi <= angle <= math.pi or not -1.000001 <= angular_velocity <= 1.000001:
            raise RuntimeError("Pendulum initial-state semantics drifted")
        self._observation = public_observation
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
            or not -2.0 <= action <= 2.0
        ):
            raise InvalidAction()

        previous_observation = self._observation
        if previous_observation is None:
            raise RuntimeError("Pendulum observation is unavailable")
        observation, reward, terminated, truncated, _ = self._environment.step((action,))
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Pendulum returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward)
        self._steps += 1
        previous_angle, previous_angular_velocity = _state(previous_observation)
        angle_cost = _ANGLE_COST_COEFFICIENT * previous_angle**2
        angular_velocity_cost = (
            _ANGULAR_VELOCITY_COST_COEFFICIENT * previous_angular_velocity**2
        )
        torque_cost = _TORQUE_COST_COEFFICIENT * action**2
        self._cumulative_angle_cost += angle_cost
        self._cumulative_angular_velocity_cost += angular_velocity_cost
        self._cumulative_torque_cost += torque_cost
        self._cumulative_reward += public_reward
        metrics = _transition_metrics(
            previous_observation,
            public_observation,
            torque=action,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            cumulative_angle_cost=self._cumulative_angle_cost,
            cumulative_angular_velocity_cost=self._cumulative_angular_velocity_cost,
            cumulative_torque_cost=self._cumulative_torque_cost,
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
        raise RuntimeError("Pendulum returned an invalid observation")
    items = tuple(value)
    if len(items) != len(_OBSERVATION_NAMES):
        raise RuntimeError("Pendulum returned an invalid observation shape")
    return {
        name: _number(item)
        for name, item in zip(_OBSERVATION_NAMES, items, strict=True)
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("Pendulum returned a non-numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("Pendulum returned a non-finite value")
    return number


def _transition_metrics(
    previous: dict[str, PolicyValue],
    current: dict[str, PolicyValue],
    *,
    torque: float,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    cumulative_angle_cost: float,
    cumulative_angular_velocity_cost: float,
    cumulative_torque_cost: float,
    cumulative_reward: float,
) -> dict[str, PolicyValue]:
    previous_angle, previous_velocity = _state(previous)
    angle, velocity = _state(current)
    gravity_velocity_increment = (
        3.0 * _GRAVITY / (2.0 * _LENGTH) * math.sin(previous_angle)
    ) * _SECONDS_PER_STEP
    torque_velocity_increment = (
        3.0 / (_MASS * _LENGTH**2) * torque
    ) * _SECONDS_PER_STEP
    unclipped_velocity = (
        previous_velocity + gravity_velocity_increment + torque_velocity_increment
    )
    expected_velocity = min(max(unclipped_velocity, -_MAX_SPEED), _MAX_SPEED)
    expected_angle = _angle_normalize(
        previous_angle + expected_velocity * _SECONDS_PER_STEP
    )
    if not math.isclose(velocity, expected_velocity, rel_tol=0.0, abs_tol=2e-6):
        raise RuntimeError("Pendulum angular-velocity dynamics drifted")
    if abs(_angle_normalize(angle - expected_angle)) > 2e-6:
        raise RuntimeError("Pendulum angle dynamics drifted")
    if terminated:
        raise RuntimeError("Pendulum unexpectedly terminated")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Pendulum time-limit semantics drifted")
    angle_cost = _ANGLE_COST_COEFFICIENT * previous_angle**2
    angular_velocity_cost = (
        _ANGULAR_VELOCITY_COST_COEFFICIENT * previous_velocity**2
    )
    torque_cost = _TORQUE_COST_COEFFICIENT * torque**2
    total_cost = angle_cost + angular_velocity_cost + torque_cost
    if not math.isclose(reward, -total_cost, rel_tol=0.0, abs_tol=2e-5):
        raise RuntimeError("Pendulum reward semantics drifted")
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "simulated_seconds": step_count * _SECONDS_PER_STEP,
        "requested_torque_newton_meters": torque,
        "torque_direction": _signed_direction(torque),
        "angle_before_radians": previous_angle,
        "angle_after_radians": angle,
        "angle_before_degrees": math.degrees(previous_angle),
        "angle_after_degrees": math.degrees(angle),
        "absolute_angle_error_before_radians": abs(previous_angle),
        "absolute_angle_error_after_radians": abs(angle),
        "angular_velocity_before_radians_per_second": previous_velocity,
        "angular_velocity_after_radians_per_second": velocity,
        "gravity_velocity_increment_radians_per_second": gravity_velocity_increment,
        "torque_velocity_increment_radians_per_second": torque_velocity_increment,
        "unclipped_angular_velocity_radians_per_second": unclipped_velocity,
        "angular_velocity_was_clipped": unclipped_velocity != expected_velocity,
        "angle_change_radians": _angle_normalize(angle - previous_angle),
        "upright_half_before": math.cos(previous_angle) >= 0.0,
        "upright_half_after": math.cos(angle) >= 0.0,
        "angle_cost": angle_cost,
        "angular_velocity_cost": angular_velocity_cost,
        "torque_cost": torque_cost,
        "total_cost": total_cost,
        "reward_from_public_terms": -total_cost,
        "cumulative_angle_cost": cumulative_angle_cost,
        "cumulative_angular_velocity_cost": cumulative_angular_velocity_cost,
        "cumulative_torque_cost": cumulative_torque_cost,
        "cumulative_total_cost": (
            cumulative_angle_cost
            + cumulative_angular_velocity_cost
            + cumulative_torque_cost
        ),
        "cumulative_return": cumulative_reward,
        "cos_sin_unit_circle_error": abs(
            _float_field(current, "cos_theta") ** 2
            + _float_field(current, "sin_theta") ** 2
            - 1.0
        ),
        "terminal_reason": "time_limit" if truncated else "none",
    }


def _state(observation: dict[str, PolicyValue]) -> tuple[float, float]:
    cosine = _float_field(observation, "cos_theta")
    sine = _float_field(observation, "sin_theta")
    angular_velocity = _float_field(observation, "theta_angular_velocity")
    if abs(cosine**2 + sine**2 - 1.0) > 2e-6:
        raise RuntimeError("Pendulum cosine and sine are not on the unit circle")
    if not -8.000001 <= angular_velocity <= 8.000001:
        raise RuntimeError("Pendulum angular velocity is outside its public bounds")
    return math.atan2(sine, cosine), angular_velocity


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Pendulum returned invalid {name}")
    return value


def _angle_normalize(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def _signed_direction(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


__all__ = ["PendulumEnvironment"]
