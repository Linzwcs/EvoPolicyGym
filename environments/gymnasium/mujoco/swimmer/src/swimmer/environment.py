"""One fresh Gymnasium Swimmer-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import SwimmerConfig

_BODY_FIELDS = (
    "front_angle",
    "rotor1_angle",
    "rotor2_angle",
    "tip_x_velocity",
    "tip_y_velocity",
    "front_angular_velocity",
    "rotor1_angular_velocity",
    "rotor2_angular_velocity",
)
_MODEL_TIMESTEP_SECONDS = 0.01
_ACTUATOR_GEAR = 150.0
_MAX_EPISODE_STEPS = 1_000


class SwimmerEnvironment:
    """The seeded strict adapter around configured Swimmer-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: SwimmerConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not SwimmerConfig:
            raise TypeError("config must be SwimmerConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Swimmer configuration belongs in SwimmerConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._exclude_positions = config.exclude_current_positions_from_observation
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Swimmer-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
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
        self._start_y_position: float | None = None
        self._minimum_x_position = math.inf
        self._maximum_x_position = -math.inf
        self._minimum_y_position = math.inf
        self._maximum_y_position = -math.inf
        self._path_length = 0.0
        self._minimum_x_velocity = math.inf
        self._maximum_x_velocity = -math.inf
        self._maximum_absolute_y_velocity = 0.0
        self._cumulative_absolute_y_velocity = 0.0
        self._backward_steps = 0
        self._cumulative_action_squared_norm = 0.0
        self._cumulative_reward_forward = 0.0
        self._cumulative_reward_control = 0.0
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
            raise RuntimeError("Swimmer returned invalid termination flags")
        public_observation = _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        seconds_per_step = self._config.frame_skip * _MODEL_TIMESTEP_SECONDS
        x_position = provider_metrics["x_position"]
        y_position = provider_metrics["y_position"]
        x_velocity = provider_metrics["x_velocity"]
        y_velocity = provider_metrics["y_velocity"]
        if self._start_x_position is None or self._start_y_position is None:
            self._start_x_position = x_position - x_velocity * seconds_per_step
            self._start_y_position = y_position - y_velocity * seconds_per_step
            self._minimum_x_position = self._start_x_position
            self._maximum_x_position = self._start_x_position
            self._minimum_y_position = self._start_y_position
            self._maximum_y_position = self._start_y_position
        self._steps += 1
        self._minimum_x_position = min(self._minimum_x_position, x_position)
        self._maximum_x_position = max(self._maximum_x_position, x_position)
        self._minimum_y_position = min(self._minimum_y_position, y_position)
        self._maximum_y_position = max(self._maximum_y_position, y_position)
        self._path_length += seconds_per_step * math.hypot(x_velocity, y_velocity)
        self._minimum_x_velocity = min(self._minimum_x_velocity, x_velocity)
        self._maximum_x_velocity = max(self._maximum_x_velocity, x_velocity)
        self._maximum_absolute_y_velocity = max(
            self._maximum_absolute_y_velocity,
            abs(y_velocity),
        )
        self._cumulative_absolute_y_velocity += abs(y_velocity)
        self._backward_steps += int(x_velocity < 0.0)
        action_squared_norm = float(numpy.square(applied_action).sum())
        self._cumulative_action_squared_norm += action_squared_norm
        self._cumulative_reward_forward += provider_metrics["reward_forward"]
        self._cumulative_reward_control += provider_metrics["reward_control"]
        self._cumulative_return += public_reward
        metrics = _transition_metrics(
            public_observation,
            action=applied_action,
            action_squared_norm=action_squared_norm,
            reward=public_reward,
            provider_metrics=provider_metrics,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            start_x_position=self._start_x_position,
            start_y_position=self._start_y_position,
            minimum_x_position=self._minimum_x_position,
            maximum_x_position=self._maximum_x_position,
            minimum_y_position=self._minimum_y_position,
            maximum_y_position=self._maximum_y_position,
            path_length=self._path_length,
            minimum_x_velocity=self._minimum_x_velocity,
            maximum_x_velocity=self._maximum_x_velocity,
            maximum_absolute_y_velocity=self._maximum_absolute_y_velocity,
            cumulative_absolute_y_velocity=(self._cumulative_absolute_y_velocity),
            backward_steps=self._backward_steps,
            cumulative_action_squared_norm=(self._cumulative_action_squared_norm),
            cumulative_reward_forward=self._cumulative_reward_forward,
            cumulative_reward_control=self._cumulative_reward_control,
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
    if type(value) is not list or len(value) != 2:
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
    expected_shape = (8,) if exclude_positions else (10,)
    if (
        type(value) is not numpy.ndarray
        or value.shape != expected_shape
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Swimmer returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
        observation["tip_x_position"] = _number(
            value[0],
            name="tip x position",
        )
        observation["tip_y_position"] = _number(
            value[1],
            name="tip y position",
        )
        offset = 2
    for name, item in zip(
        _BODY_FIELDS,
        value[offset:],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    return observation


def _provider_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("Swimmer returned invalid metrics")
    names = (
        "x_position",
        "y_position",
        "distance_from_origin",
        "x_velocity",
        "y_velocity",
        "reward_forward",
        "reward_ctrl",
    )
    if not set(names).issubset(value):
        raise RuntimeError("Swimmer omitted public metrics")
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
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: SwimmerConfig,
    start_x_position: float,
    start_y_position: float,
    minimum_x_position: float,
    maximum_x_position: float,
    minimum_y_position: float,
    maximum_y_position: float,
    path_length: float,
    minimum_x_velocity: float,
    maximum_x_velocity: float,
    maximum_absolute_y_velocity: float,
    cumulative_absolute_y_velocity: float,
    backward_steps: int,
    cumulative_action_squared_norm: float,
    cumulative_reward_forward: float,
    cumulative_reward_control: float,
    cumulative_return: float,
) -> dict[str, PolicyValue]:
    x_velocity = provider_metrics["x_velocity"]
    y_velocity = provider_metrics["y_velocity"]
    expected_forward_reward = config.forward_reward_weight * x_velocity
    expected_control_reward = -config.ctrl_cost_weight * action_squared_norm
    if not math.isclose(
        provider_metrics["reward_forward"],
        expected_forward_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Swimmer forward-reward semantics drifted")
    if not math.isclose(
        provider_metrics["reward_control"],
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Swimmer control-reward semantics drifted")
    reconstructed_reward = provider_metrics["reward_forward"] + provider_metrics["reward_control"]
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Swimmer reward decomposition drifted")
    if terminated:
        raise RuntimeError("Swimmer natural-termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Swimmer time-limit semantics drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_reward_forward + cumulative_reward_control,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Swimmer cumulative reward decomposition drifted")
    x_position = provider_metrics["x_position"]
    y_position = provider_metrics["y_position"]
    forward_displacement = x_position - start_x_position
    lateral_displacement = y_position - start_y_position
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    requested_rotor1 = float(action[0])
    requested_rotor2 = float(action[1])
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "requested_rotor1_control": requested_rotor1,
        "requested_rotor2_control": requested_rotor2,
        "gear_scaled_rotor1_torque": requested_rotor1 * _ACTUATOR_GEAR,
        "gear_scaled_rotor2_torque": requested_rotor2 * _ACTUATOR_GEAR,
        "action_squared_norm": action_squared_norm,
        "cumulative_action_squared_norm": cumulative_action_squared_norm,
        "mean_action_squared_norm": cumulative_action_squared_norm / step_count,
        "start_x_position": start_x_position,
        "start_y_position": start_y_position,
        "x_position": x_position,
        "y_position": y_position,
        "minimum_x_position": minimum_x_position,
        "maximum_x_position": maximum_x_position,
        "minimum_y_position": minimum_y_position,
        "maximum_y_position": maximum_y_position,
        "forward_displacement": forward_displacement,
        "lateral_displacement": lateral_displacement,
        "maximum_absolute_lateral_displacement": max(
            abs(minimum_y_position - start_y_position),
            abs(maximum_y_position - start_y_position),
        ),
        "net_displacement": math.hypot(
            forward_displacement,
            lateral_displacement,
        ),
        "path_length": path_length,
        "distance_from_origin": provider_metrics["distance_from_origin"],
        "step_average_x_velocity": x_velocity,
        "step_average_y_velocity": y_velocity,
        "observation_tip_x_velocity": _float_field(
            observation,
            "tip_x_velocity",
        ),
        "observation_tip_y_velocity": _float_field(
            observation,
            "tip_y_velocity",
        ),
        "mean_x_velocity": forward_displacement / (step_count * seconds_per_step),
        "mean_absolute_y_velocity": cumulative_absolute_y_velocity / step_count,
        "minimum_x_velocity": minimum_x_velocity,
        "maximum_x_velocity": maximum_x_velocity,
        "maximum_absolute_y_velocity": maximum_absolute_y_velocity,
        "backward_step_fraction": backward_steps / step_count,
        "front_angle_radians": _float_field(observation, "front_angle"),
        "rotor1_relative_angle_radians": _float_field(
            observation,
            "rotor1_angle",
        ),
        "rotor2_relative_angle_radians": _float_field(
            observation,
            "rotor2_angle",
        ),
        "reward_forward": provider_metrics["reward_forward"],
        "reward_control": provider_metrics["reward_control"],
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_forward": cumulative_reward_forward,
        "cumulative_reward_control": cumulative_reward_control,
        "cumulative_return": cumulative_return,
        "terminal_reason": "time_limit" if truncated else "none",
    }


def _float_field(
    observation: dict[str, PolicyValue],
    name: str,
) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Swimmer returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Swimmer returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Swimmer returned a non-finite {name}")
    return number


__all__ = ["SwimmerEnvironment"]
