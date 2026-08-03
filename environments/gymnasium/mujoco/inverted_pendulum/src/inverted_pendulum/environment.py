"""One fresh Gymnasium InvertedPendulum-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import InvertedPendulumConfig

_OBSERVATION_NAMES = (
    "cart_position",
    "pole_angle",
    "cart_velocity",
    "pole_angular_velocity",
)
_MODEL_TIMESTEP_SECONDS = 0.02
_ACTUATOR_GEAR = 100.0
_TERMINATION_ANGLE_RADIANS = 0.2
_MAX_EPISODE_STEPS = 1_000


class InvertedPendulumEnvironment:
    """The seeded strict adapter around configured InvertedPendulum-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: InvertedPendulumConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not InvertedPendulumConfig:
            raise TypeError("config must be InvertedPendulumConfig")
        if episode.scenario is not None:
            raise ValueError(
                "InvertedPendulum configuration belongs in "
                "InvertedPendulumConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "InvertedPendulum-v5",
                frame_skip=config.frame_skip,
                reset_noise_scale=config.reset_noise_scale,
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._minimum_cart_position = math.inf
        self._maximum_cart_position = -math.inf
        self._maximum_absolute_cart_velocity = 0.0
        self._maximum_absolute_pole_angle = 0.0
        self._minimum_pole_angle_margin = math.inf
        self._maximum_absolute_pole_angular_velocity = 0.0
        self._cumulative_absolute_action = 0.0
        self._cumulative_survival_reward = 0.0
        self._cumulative_return = 0.0

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

        applied_action = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(applied_action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("InvertedPendulum returned invalid termination flags")
        public_observation = _observation(observation)
        public_reward = _number(reward, name="reward")
        reward_survive = _reward_survive(info)
        cart_position = _float_field(public_observation, "cart_position")
        pole_angle = _float_field(public_observation, "pole_angle")
        cart_velocity = _float_field(public_observation, "cart_velocity")
        pole_angular_velocity = _float_field(
            public_observation,
            "pole_angular_velocity",
        )
        self._steps += 1
        self._minimum_cart_position = min(
            self._minimum_cart_position,
            cart_position,
        )
        self._maximum_cart_position = max(
            self._maximum_cart_position,
            cart_position,
        )
        self._maximum_absolute_cart_velocity = max(
            self._maximum_absolute_cart_velocity,
            abs(cart_velocity),
        )
        self._maximum_absolute_pole_angle = max(
            self._maximum_absolute_pole_angle,
            abs(pole_angle),
        )
        angle_margin = _TERMINATION_ANGLE_RADIANS - abs(pole_angle)
        self._minimum_pole_angle_margin = min(
            self._minimum_pole_angle_margin,
            angle_margin,
        )
        self._maximum_absolute_pole_angular_velocity = max(
            self._maximum_absolute_pole_angular_velocity,
            abs(pole_angular_velocity),
        )
        requested_action = float(applied_action[0])
        self._cumulative_absolute_action += abs(requested_action)
        self._cumulative_survival_reward += reward_survive
        self._cumulative_return += public_reward
        metrics = _transition_metrics(
            public_observation,
            requested_action=requested_action,
            reward=public_reward,
            reward_survive=reward_survive,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            minimum_cart_position=self._minimum_cart_position,
            maximum_cart_position=self._maximum_cart_position,
            maximum_absolute_cart_velocity=(self._maximum_absolute_cart_velocity),
            maximum_absolute_pole_angle=(self._maximum_absolute_pole_angle),
            minimum_pole_angle_margin=self._minimum_pole_angle_margin,
            maximum_absolute_pole_angular_velocity=(self._maximum_absolute_pole_angular_velocity),
            cumulative_absolute_action=self._cumulative_absolute_action,
            cumulative_survival_reward=self._cumulative_survival_reward,
            cumulative_return=self._cumulative_return,
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


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 1:
        raise InvalidAction()
    item = value[0]
    if type(item) is not float or not math.isfinite(item) or not -3.0 <= item <= 3.0:
        raise InvalidAction()
    return numpy.asarray([item], dtype=numpy.float32)


def _observation(value: object) -> dict[str, PolicyValue]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (4,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("InvertedPendulum returned an invalid observation")
    return {
        name: _number(item, name=name) for name, item in zip(_OBSERVATION_NAMES, value, strict=True)
    }


def _reward_survive(value: object) -> float:
    if type(value) is not dict or "reward_survive" not in value:
        raise RuntimeError("InvertedPendulum omitted survival reward metrics")
    return _number(
        value["reward_survive"],
        name="survival reward",
    )


def _transition_metrics(
    observation: dict[str, PolicyValue],
    *,
    requested_action: float,
    reward: float,
    reward_survive: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: InvertedPendulumConfig,
    minimum_cart_position: float,
    maximum_cart_position: float,
    maximum_absolute_cart_velocity: float,
    maximum_absolute_pole_angle: float,
    minimum_pole_angle_margin: float,
    maximum_absolute_pole_angular_velocity: float,
    cumulative_absolute_action: float,
    cumulative_survival_reward: float,
    cumulative_return: float,
) -> dict[str, PolicyValue]:
    pole_angle = _float_field(observation, "pole_angle")
    healthy = abs(pole_angle) <= _TERMINATION_ANGLE_RADIANS
    expected_survival = 1.0 if healthy else 0.0
    if reward_survive != expected_survival or reward != reward_survive:
        raise RuntimeError("InvertedPendulum survival-reward semantics drifted")
    if terminated != (not healthy):
        raise RuntimeError("InvertedPendulum angle termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("InvertedPendulum time-limit semantics drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_survival_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("InvertedPendulum cumulative reward semantics drifted")
    terminal_reason = "none"
    if terminated and truncated:
        terminal_reason = "fallen_and_time_limit"
    elif terminated:
        terminal_reason = "fallen"
    elif truncated:
        terminal_reason = "time_limit"
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "requested_cart_control": requested_action,
        "actuator_gear_scaled_cart_force": requested_action * _ACTUATOR_GEAR,
        "cumulative_absolute_action": cumulative_absolute_action,
        "cart_position": _float_field(observation, "cart_position"),
        "minimum_cart_position": minimum_cart_position,
        "maximum_cart_position": maximum_cart_position,
        "cart_velocity": _float_field(observation, "cart_velocity"),
        "maximum_absolute_cart_velocity": maximum_absolute_cart_velocity,
        "pole_angle_radians": pole_angle,
        "pole_angle_degrees": math.degrees(pole_angle),
        "maximum_absolute_pole_angle_radians": maximum_absolute_pole_angle,
        "pole_angle_termination_threshold_radians": (_TERMINATION_ANGLE_RADIANS),
        "pole_angle_margin_radians": (_TERMINATION_ANGLE_RADIANS - abs(pole_angle)),
        "minimum_pole_angle_margin_radians": minimum_pole_angle_margin,
        "pole_angular_velocity": _float_field(
            observation,
            "pole_angular_velocity",
        ),
        "maximum_absolute_pole_angular_velocity": (maximum_absolute_pole_angular_velocity),
        "healthy": healthy,
        "reward_survive": reward_survive,
        "cumulative_reward_survive": cumulative_survival_reward,
        "cumulative_return": cumulative_return,
        "terminal_reason": terminal_reason,
    }


def _float_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"InvertedPendulum returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"InvertedPendulum returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"InvertedPendulum returned a non-finite {name}")
    return number


__all__ = ["InvertedPendulumEnvironment"]
