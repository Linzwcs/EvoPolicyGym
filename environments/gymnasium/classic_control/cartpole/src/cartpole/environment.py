"""One fresh Gymnasium CartPole Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

_MAX_EPISODE_STEPS = 500
_CART_POSITION_LIMIT = 2.4
_POLE_ANGLE_LIMIT_RADIANS = 12.0 * math.pi / 180.0
_SECONDS_PER_STEP = 0.02
_FORCE_MAGNITUDE_NEWTONS = 10.0


class CartPoleEnvironment:
    """The minimal seeded adapter around Gymnasium CartPole-v1."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("CartPole does not accept Episode scenarios")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make("CartPole-v1"),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0

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
        if type(action) is not int or action not in {0, 1}:
            raise InvalidAction()

        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("CartPole returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward)
        if public_reward != 1.0:
            raise RuntimeError("CartPole reward semantics drifted")
        self._steps += 1
        metrics = _transition_metrics(
            public_observation,
            action=action,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
        )
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


def _observation(value: object) -> list[PolicyValue]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise RuntimeError("CartPole returned an invalid observation")
    items = tuple(value)
    if len(items) != 4:
        raise RuntimeError("CartPole returned an invalid observation shape")
    return [_number(item) for item in items]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("CartPole returned a non-numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("CartPole returned a non-finite value")
    return number


def _transition_metrics(
    observation: list[PolicyValue],
    *,
    action: int,
    terminated: bool,
    truncated: bool,
    step_count: int,
) -> dict[str, PolicyValue]:
    values: list[float] = []
    for value in observation:
        if type(value) is not float:
            raise RuntimeError("CartPole returned an invalid public observation")
        values.append(value)
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = values
    cart_margin = _CART_POSITION_LIMIT - abs(cart_position)
    pole_margin = _POLE_ANGLE_LIMIT_RADIANS - abs(pole_angle)
    cart_within_limit = cart_margin >= 0.0
    pole_within_limit = pole_margin >= 0.0
    reasons: list[str] = []
    if not cart_within_limit:
        reasons.append("cart_position_limit")
    if not pole_within_limit:
        reasons.append("pole_angle_limit")
    if terminated and not reasons:
        raise RuntimeError("CartPole terminated without crossing a public limit")
    if not terminated and reasons:
        raise RuntimeError("CartPole crossed a termination limit without terminating")
    if truncated:
        if step_count != _MAX_EPISODE_STEPS:
            raise RuntimeError("CartPole truncated before its public time limit")
        reasons.append("time_limit")
    metrics: dict[str, PolicyValue] = {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "elapsed_simulation_seconds": step_count * _SECONDS_PER_STEP,
        "requested_action": "push_left" if action == 0 else "push_right",
        "applied_force_newtons": (
            -_FORCE_MAGNITUDE_NEWTONS if action == 0 else _FORCE_MAGNITUDE_NEWTONS
        ),
        "cart_position": cart_position,
        "cart_velocity": cart_velocity,
        "pole_angle_radians": pole_angle,
        "pole_angle_degrees": math.degrees(pole_angle),
        "pole_angular_velocity_radians_per_second": pole_angular_velocity,
        "cart_position_limit_margin": cart_margin,
        "pole_angle_limit_margin_radians": pole_margin,
        "cart_within_limit": cart_within_limit,
        "pole_within_limit": pole_within_limit,
        "balanced_within_limits": cart_within_limit and pole_within_limit,
        "survival_fraction": step_count / _MAX_EPISODE_STEPS,
    }
    if reasons:
        metrics["terminal_reason"] = "+".join(reasons)
    return metrics


__all__ = ["CartPoleEnvironment"]
