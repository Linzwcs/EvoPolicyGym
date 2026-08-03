"""One fresh Gymnasium Acrobot Environment per Episode."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

_OBSERVATION_NAMES = (
    "cos_theta_1",
    "sin_theta_1",
    "cos_theta_2",
    "sin_theta_2",
    "theta_1_angular_velocity",
    "theta_2_angular_velocity",
)
_MAX_EPISODE_STEPS = 500
_SECONDS_PER_STEP = 0.2
_TARGET_HEIGHT_METERS = 1.0
_TORQUES_NEWTON_METERS = (-1.0, 0.0, 1.0)


class AcrobotEnvironment:
    """The seeded strict adapter around Gymnasium Acrobot-v1."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Acrobot does not accept Episode scenarios")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make("Acrobot-v1"),
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
        if type(action) is not int or action not in {0, 1, 2}:
            raise InvalidAction()

        observation, reward, terminated, truncated, _ = self._environment.step(
            action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Acrobot returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward)
        if public_reward != (0.0 if terminated else -1.0):
            raise RuntimeError("Acrobot reward semantics drifted")
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


def _observation(value: object) -> dict[str, PolicyValue]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise RuntimeError("Acrobot returned an invalid observation")
    items = tuple(value)
    if len(items) != len(_OBSERVATION_NAMES):
        raise RuntimeError("Acrobot returned an invalid observation shape")
    return {
        name: _number(item)
        for name, item in zip(_OBSERVATION_NAMES, items, strict=True)
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("Acrobot returned a non-numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("Acrobot returned a non-finite value")
    return number


def _transition_metrics(
    observation: dict[str, PolicyValue],
    *,
    action: int,
    terminated: bool,
    truncated: bool,
    step_count: int,
) -> dict[str, PolicyValue]:
    cos_theta_1 = _float_field(observation, "cos_theta_1")
    sin_theta_1 = _float_field(observation, "sin_theta_1")
    cos_theta_2 = _float_field(observation, "cos_theta_2")
    sin_theta_2 = _float_field(observation, "sin_theta_2")
    theta_1 = math.atan2(sin_theta_1, cos_theta_1)
    theta_2_relative = math.atan2(sin_theta_2, cos_theta_2)
    link_1_vertical_height = -cos_theta_1
    link_2_vertical_component = -(
        cos_theta_1 * cos_theta_2 - sin_theta_1 * sin_theta_2
    )
    tip_height = link_1_vertical_height + link_2_vertical_component
    target_margin = tip_height - _TARGET_HEIGHT_METERS
    if terminated and target_margin < -1e-5:
        raise RuntimeError("Acrobot terminated below its public target height")
    if not terminated and target_margin > 1e-5:
        raise RuntimeError("Acrobot crossed its target height without terminating")
    if truncated and step_count != _MAX_EPISODE_STEPS:
        raise RuntimeError("Acrobot truncated before its public time limit")
    reasons: list[str] = []
    if terminated:
        reasons.append("target_height_reached")
    if truncated:
        reasons.append("time_limit")
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "elapsed_simulation_seconds": step_count * _SECONDS_PER_STEP,
        "requested_action": (
            "negative_torque" if action == 0 else "zero_torque" if action == 1 else "positive_torque"
        ),
        "applied_torque_newton_meters": _TORQUES_NEWTON_METERS[action],
        "theta_1_radians_from_downward": theta_1,
        "theta_1_degrees_from_downward": math.degrees(theta_1),
        "theta_2_relative_radians": theta_2_relative,
        "theta_2_relative_degrees": math.degrees(theta_2_relative),
        "theta_2_absolute_radians_from_downward": theta_1 + theta_2_relative,
        "theta_1_angular_velocity_radians_per_second": _float_field(
            observation,
            "theta_1_angular_velocity",
        ),
        "theta_2_relative_angular_velocity_radians_per_second": _float_field(
            observation,
            "theta_2_angular_velocity",
        ),
        "link_1_end_vertical_height_meters": link_1_vertical_height,
        "free_end_vertical_height_meters": tip_height,
        "target_height_meters": _TARGET_HEIGHT_METERS,
        "target_height_margin_meters": target_margin,
        "height_remaining_to_target_meters": max(-target_margin, 0.0),
        "target_reached_from_public_observation": target_margin > 0.0,
        "theta_1_unit_circle_error": abs(
            cos_theta_1 * cos_theta_1 + sin_theta_1 * sin_theta_1 - 1.0
        ),
        "theta_2_unit_circle_error": abs(
            cos_theta_2 * cos_theta_2 + sin_theta_2 * sin_theta_2 - 1.0
        ),
        "terminal_reason": "+".join(reasons) if reasons else "none",
    }


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Acrobot returned invalid {name}")
    return value


__all__ = ["AcrobotEnvironment"]
