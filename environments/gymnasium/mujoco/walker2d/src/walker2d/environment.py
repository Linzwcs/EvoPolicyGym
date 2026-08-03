"""One fresh Gymnasium Walker2d-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import Walker2dConfig

_BODY_FIELDS = (
    "torso_z_position",
    "torso_angle",
    "right_thigh_angle",
    "right_leg_angle",
    "right_foot_angle",
    "left_thigh_angle",
    "left_leg_angle",
    "left_foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_angular_velocity",
    "right_thigh_angular_velocity",
    "right_leg_angular_velocity",
    "right_foot_angular_velocity",
    "left_thigh_angular_velocity",
    "left_leg_angular_velocity",
    "left_foot_angular_velocity",
)
_VELOCITY_FIELDS = _BODY_FIELDS[8:]
_MODEL_TIMESTEP_SECONDS = 0.002
_ACTUATOR_GEAR = 100.0
_VELOCITY_CLIP_LIMIT = 10.0
_MAX_EPISODE_STEPS = 1_000


class Walker2dEnvironment:
    """The seeded strict adapter around configured Walker2d-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: Walker2dConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not Walker2dConfig:
            raise TypeError("config must be Walker2dConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Walker2d configuration belongs in Walker2dConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._exclude_positions = config.exclude_current_positions_from_observation
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Walker2d-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                healthy_reward=config.healthy_reward,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_z_range=config.healthy_z_range,
                healthy_angle_range=config.healthy_angle_range,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._start_x_position: float | None = None
        self._minimum_x_position = math.inf
        self._maximum_x_position = -math.inf
        self._minimum_torso_z_position = math.inf
        self._maximum_torso_z_position = -math.inf
        self._minimum_torso_angle = math.inf
        self._maximum_torso_angle = -math.inf
        self._minimum_height_health_margin = math.inf
        self._minimum_angle_health_margin = math.inf
        self._minimum_x_velocity = math.inf
        self._maximum_x_velocity = -math.inf
        self._backward_steps = 0
        self._unhealthy_steps = 0
        self._maximum_observed_absolute_velocity = 0.0
        self._velocity_clip_limit_steps = 0
        self._cumulative_action_squared_norm = 0.0
        self._cumulative_reward_forward = 0.0
        self._cumulative_reward_control = 0.0
        self._cumulative_reward_survive = 0.0
        self._cumulative_return = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        self._started = True
        return _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )

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
            raise RuntimeError("Walker2d returned invalid termination flags")
        public_observation = _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        seconds_per_step = self._config.frame_skip * _MODEL_TIMESTEP_SECONDS
        x_position = provider_metrics["x_position"]
        x_velocity = provider_metrics["x_velocity"]
        torso_z = _float_field(public_observation, "torso_z_position")
        torso_angle = _float_field(public_observation, "torso_angle")
        height_margin = _range_margin(torso_z, self._config.healthy_z_range)
        angle_margin = _range_margin(
            torso_angle,
            self._config.healthy_angle_range,
        )
        healthy = height_margin > 0.0 and angle_margin > 0.0
        if self._start_x_position is None:
            self._start_x_position = x_position - x_velocity * seconds_per_step
            self._minimum_x_position = self._start_x_position
            self._maximum_x_position = self._start_x_position
        self._steps += 1
        self._minimum_x_position = min(self._minimum_x_position, x_position)
        self._maximum_x_position = max(self._maximum_x_position, x_position)
        self._minimum_torso_z_position = min(
            self._minimum_torso_z_position,
            torso_z,
        )
        self._maximum_torso_z_position = max(
            self._maximum_torso_z_position,
            torso_z,
        )
        self._minimum_torso_angle = min(self._minimum_torso_angle, torso_angle)
        self._maximum_torso_angle = max(self._maximum_torso_angle, torso_angle)
        self._minimum_height_health_margin = min(
            self._minimum_height_health_margin,
            height_margin,
        )
        self._minimum_angle_health_margin = min(
            self._minimum_angle_health_margin,
            angle_margin,
        )
        self._minimum_x_velocity = min(self._minimum_x_velocity, x_velocity)
        self._maximum_x_velocity = max(self._maximum_x_velocity, x_velocity)
        self._backward_steps += int(x_velocity < 0.0)
        self._unhealthy_steps += int(not healthy)
        observed_max_velocity = max(
            abs(_float_field(public_observation, name)) for name in _VELOCITY_FIELDS
        )
        velocity_at_clip_limit = observed_max_velocity == _VELOCITY_CLIP_LIMIT
        self._maximum_observed_absolute_velocity = max(
            self._maximum_observed_absolute_velocity,
            observed_max_velocity,
        )
        self._velocity_clip_limit_steps += int(velocity_at_clip_limit)
        action_squared_norm = float(numpy.square(applied_action).sum())
        self._cumulative_action_squared_norm += action_squared_norm
        self._cumulative_reward_forward += provider_metrics["reward_forward"]
        self._cumulative_reward_control += provider_metrics["reward_control"]
        self._cumulative_reward_survive += provider_metrics["reward_survive"]
        self._cumulative_return += public_reward
        metrics = _transition_metrics(
            public_observation,
            action=applied_action,
            action_squared_norm=action_squared_norm,
            reward=public_reward,
            provider_metrics=provider_metrics,
            healthy=healthy,
            height_margin=height_margin,
            angle_margin=angle_margin,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            start_x_position=self._start_x_position,
            minimum_x_position=self._minimum_x_position,
            maximum_x_position=self._maximum_x_position,
            minimum_torso_z_position=self._minimum_torso_z_position,
            maximum_torso_z_position=self._maximum_torso_z_position,
            minimum_torso_angle=self._minimum_torso_angle,
            maximum_torso_angle=self._maximum_torso_angle,
            minimum_height_health_margin=self._minimum_height_health_margin,
            minimum_angle_health_margin=self._minimum_angle_health_margin,
            minimum_x_velocity=self._minimum_x_velocity,
            maximum_x_velocity=self._maximum_x_velocity,
            backward_steps=self._backward_steps,
            unhealthy_steps=self._unhealthy_steps,
            maximum_observed_absolute_velocity=(self._maximum_observed_absolute_velocity),
            velocity_at_clip_limit=velocity_at_clip_limit,
            velocity_clip_limit_steps=self._velocity_clip_limit_steps,
            cumulative_action_squared_norm=(self._cumulative_action_squared_norm),
            cumulative_reward_forward=self._cumulative_reward_forward,
            cumulative_reward_control=self._cumulative_reward_control,
            cumulative_reward_survive=self._cumulative_reward_survive,
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
    if type(value) is not list or len(value) != 6:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or not -1.0 <= item <= 1.0:
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    exclude_positions: bool,
) -> dict[str, PolicyValue]:
    expected_shape = (17,) if exclude_positions else (18,)
    if (
        type(value) is not numpy.ndarray
        or value.shape != expected_shape
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Walker2d returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
        observation["torso_x_position"] = _number(
            value[0],
            name="torso x position",
        )
        offset = 1
    for name, item in zip(
        _BODY_FIELDS,
        value[offset:],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    return observation


def _provider_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("Walker2d returned invalid metrics")
    names = (
        "x_position",
        "z_distance_from_origin",
        "x_velocity",
        "reward_forward",
        "reward_ctrl",
        "reward_survive",
    )
    if not set(names).issubset(value):
        raise RuntimeError("Walker2d omitted public metrics")
    return {
        ("reward_control" if name == "reward_ctrl" else name): _number(
            value[name], name=name.replace("_", " ")
        )
        for name in names
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    *,
    action: NDArray[numpy.float32],
    action_squared_norm: float,
    reward: float,
    provider_metrics: dict[str, float],
    healthy: bool,
    height_margin: float,
    angle_margin: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: Walker2dConfig,
    start_x_position: float,
    minimum_x_position: float,
    maximum_x_position: float,
    minimum_torso_z_position: float,
    maximum_torso_z_position: float,
    minimum_torso_angle: float,
    maximum_torso_angle: float,
    minimum_height_health_margin: float,
    minimum_angle_health_margin: float,
    minimum_x_velocity: float,
    maximum_x_velocity: float,
    backward_steps: int,
    unhealthy_steps: int,
    maximum_observed_absolute_velocity: float,
    velocity_at_clip_limit: bool,
    velocity_clip_limit_steps: int,
    cumulative_action_squared_norm: float,
    cumulative_reward_forward: float,
    cumulative_reward_control: float,
    cumulative_reward_survive: float,
    cumulative_return: float,
) -> dict[str, PolicyValue]:
    x_velocity = provider_metrics["x_velocity"]
    expected_forward_reward = config.forward_reward_weight * x_velocity
    expected_control_reward = -config.ctrl_cost_weight * action_squared_norm
    expected_survival_reward = config.healthy_reward if healthy else 0.0
    expected_rewards = {
        "reward_forward": expected_forward_reward,
        "reward_control": expected_control_reward,
        "reward_survive": expected_survival_reward,
    }
    for name, expected in expected_rewards.items():
        if not math.isclose(
            provider_metrics[name],
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Walker2d {name} semantics drifted")
    reconstructed_reward = sum(provider_metrics[name] for name in expected_rewards)
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Walker2d reward decomposition drifted")
    expected_terminated = (not healthy) and config.terminate_when_unhealthy
    if terminated != expected_terminated:
        raise RuntimeError("Walker2d health termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Walker2d time-limit semantics drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_reward_forward + cumulative_reward_control + cumulative_reward_survive,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Walker2d cumulative reward decomposition drifted")
    terminal_reason = _terminal_reason(
        healthy_height=height_margin > 0.0,
        healthy_angle=angle_margin > 0.0,
        terminated=terminated,
        truncated=truncated,
    )
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    x_position = provider_metrics["x_position"]
    forward_displacement = x_position - start_x_position
    requested = tuple(float(value) for value in action)
    metrics: dict[str, PolicyValue] = {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "action_squared_norm": action_squared_norm,
        "cumulative_action_squared_norm": cumulative_action_squared_norm,
        "mean_action_squared_norm": cumulative_action_squared_norm / step_count,
        "start_x_position": start_x_position,
        "x_position": x_position,
        "minimum_x_position": minimum_x_position,
        "maximum_x_position": maximum_x_position,
        "forward_displacement": forward_displacement,
        "step_average_x_velocity": x_velocity,
        "observation_torso_x_velocity": _float_field(
            observation,
            "torso_x_velocity",
        ),
        "mean_x_velocity": forward_displacement / (step_count * seconds_per_step),
        "minimum_x_velocity": minimum_x_velocity,
        "maximum_x_velocity": maximum_x_velocity,
        "backward_step_fraction": backward_steps / step_count,
        "torso_z_position": _float_field(observation, "torso_z_position"),
        "minimum_torso_z_position": minimum_torso_z_position,
        "maximum_torso_z_position": maximum_torso_z_position,
        "healthy_z_lower_bound": config.healthy_z_range[0],
        "healthy_z_upper_bound": config.healthy_z_range[1],
        "height_health_margin": height_margin,
        "minimum_height_health_margin": minimum_height_health_margin,
        "torso_angle_radians": _float_field(observation, "torso_angle"),
        "minimum_torso_angle_radians": minimum_torso_angle,
        "maximum_torso_angle_radians": maximum_torso_angle,
        "healthy_angle_lower_bound": config.healthy_angle_range[0],
        "healthy_angle_upper_bound": config.healthy_angle_range[1],
        "angle_health_margin": angle_margin,
        "minimum_angle_health_margin": minimum_angle_health_margin,
        "healthy": healthy,
        "unhealthy_step_fraction": unhealthy_steps / step_count,
        "maximum_observed_absolute_velocity": maximum_observed_absolute_velocity,
        "velocity_observation_at_clip_limit": velocity_at_clip_limit,
        "velocity_clip_limit_step_fraction": velocity_clip_limit_steps / step_count,
        "z_offset_from_model_initial_pose": provider_metrics["z_distance_from_origin"],
        "reward_forward": provider_metrics["reward_forward"],
        "reward_control": provider_metrics["reward_control"],
        "reward_survive": provider_metrics["reward_survive"],
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_forward": cumulative_reward_forward,
        "cumulative_reward_control": cumulative_reward_control,
        "cumulative_reward_survive": cumulative_reward_survive,
        "cumulative_return": cumulative_return,
        "terminal_reason": terminal_reason,
    }
    action_names = (
        "right_thigh",
        "right_leg",
        "right_foot",
        "left_thigh",
        "left_leg",
        "left_foot",
    )
    for name, value in zip(action_names, requested, strict=True):
        metrics[f"requested_{name}_control"] = value
        metrics[f"gear_scaled_{name}_torque"] = value * _ACTUATOR_GEAR
    return metrics


def _range_margin(value: float, bounds: tuple[float, float]) -> float:
    return min(value - bounds[0], bounds[1] - value)


def _terminal_reason(
    *,
    healthy_height: bool,
    healthy_angle: bool,
    terminated: bool,
    truncated: bool,
) -> str:
    reason = "none"
    if terminated:
        if not healthy_height and not healthy_angle:
            reason = "unhealthy_height_and_angle"
        elif not healthy_height:
            reason = "unhealthy_height"
        else:
            reason = "unhealthy_angle"
    if truncated:
        return "time_limit" if reason == "none" else f"{reason}_and_time_limit"
    return reason


def _float_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Walker2d returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Walker2d returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Walker2d returned a non-finite {name}")
    return number


__all__ = ["Walker2dEnvironment"]
