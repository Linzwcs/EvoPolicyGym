"""One fresh Gymnasium Hopper-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import Protocol, SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import HopperConfig

_BODY_FIELDS = (
    "torso_z_position",
    "torso_pitch_angle",
    "thigh_angle",
    "leg_angle",
    "foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_pitch_angular_velocity",
    "thigh_angular_velocity",
    "leg_angular_velocity",
    "foot_angular_velocity",
)
_ACTION_COMPONENTS = ("thigh", "leg", "foot")
_ACTUATOR_GEARS = (200.0, 200.0, 200.0)
_MAX_EPISODE_STEPS = 1_000
_MODEL_TIMESTEP_SECONDS = 0.002


class _MujocoData(Protocol):
    qpos: NDArray[numpy.float64]
    qvel: NDArray[numpy.float64]


class _HopperProvider(Protocol):
    data: _MujocoData


class HopperEnvironment:
    """The seeded strict adapter around configured Hopper-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: HopperConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HopperConfig:
            raise TypeError("config must be HopperConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Hopper configuration belongs in HopperConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._exclude_positions = config.exclude_current_positions_from_observation
        self._frame_skip = config.frame_skip
        self._forward_reward_weight = config.forward_reward_weight
        self._control_cost_weight = config.ctrl_cost_weight
        self._healthy_reward = config.healthy_reward
        self._terminate_when_unhealthy = config.terminate_when_unhealthy
        self._healthy_state_range = config.healthy_state_range
        self._healthy_z_range = config.healthy_z_range
        self._healthy_angle_range = config.healthy_angle_range
        healthy_z_upper = (
            float("inf") if config.healthy_z_range[1] is None else config.healthy_z_range[1]
        )
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Hopper-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                healthy_reward=config.healthy_reward,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_state_range=config.healthy_state_range,
                healthy_z_range=(
                    config.healthy_z_range[0],
                    healthy_z_upper,
                ),
                healthy_angle_range=config.healthy_angle_range,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
            ),
        )
        self._provider = cast(_HopperProvider, self._environment.unwrapped)
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._initial_x_position: float | None = None
        self._minimum_x_position = math.inf
        self._maximum_x_position = -math.inf
        self._minimum_x_velocity = math.inf
        self._maximum_x_velocity = -math.inf
        self._minimum_torso_z_position = math.inf
        self._maximum_absolute_torso_pitch = 0.0
        self._minimum_healthy_state_margin = math.inf
        self._minimum_healthy_z_margin = math.inf
        self._minimum_healthy_angle_margin = math.inf
        self._healthy_steps = 0
        self._forward_steps = 0
        self._cumulative_forward_reward = 0.0
        self._cumulative_control_reward = 0.0
        self._cumulative_survival_reward = 0.0
        self._cumulative_return = 0.0
        self._cumulative_absolute_action = 0.0

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
            raise RuntimeError("Hopper returned invalid termination flags")
        public_observation = _observation(
            observation,
            exclude_positions=self._exclude_positions,
        )
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        health = _health(
            self._provider,
            healthy_state_range=self._healthy_state_range,
            healthy_z_range=self._healthy_z_range,
            healthy_angle_range=self._healthy_angle_range,
        )
        self._steps += 1
        seconds_per_step = self._frame_skip * _MODEL_TIMESTEP_SECONDS
        if self._initial_x_position is None:
            self._initial_x_position = (
                provider_metrics["x_position"] - provider_metrics["x_velocity"] * seconds_per_step
            )
            self._minimum_x_position = self._initial_x_position
            self._maximum_x_position = self._initial_x_position
        x_position = provider_metrics["x_position"]
        x_velocity = provider_metrics["x_velocity"]
        self._minimum_x_position = min(self._minimum_x_position, x_position)
        self._maximum_x_position = max(self._maximum_x_position, x_position)
        self._minimum_x_velocity = min(self._minimum_x_velocity, x_velocity)
        self._maximum_x_velocity = max(self._maximum_x_velocity, x_velocity)
        torso_z = _float_field(public_observation, "torso_z_position")
        torso_pitch = _float_field(public_observation, "torso_pitch_angle")
        self._minimum_torso_z_position = min(
            self._minimum_torso_z_position,
            torso_z,
        )
        self._maximum_absolute_torso_pitch = max(
            self._maximum_absolute_torso_pitch,
            abs(torso_pitch),
        )
        self._minimum_healthy_state_margin = min(
            self._minimum_healthy_state_margin,
            health.state_margin,
        )
        self._minimum_healthy_z_margin = min(
            self._minimum_healthy_z_margin,
            health.z_margin,
        )
        self._minimum_healthy_angle_margin = min(
            self._minimum_healthy_angle_margin,
            health.angle_margin,
        )
        self._healthy_steps += int(health.healthy)
        self._forward_steps += int(x_velocity > 0.0)
        self._cumulative_forward_reward += provider_metrics["reward_forward"]
        self._cumulative_control_reward += provider_metrics["reward_control"]
        self._cumulative_survival_reward += provider_metrics["reward_survive"]
        self._cumulative_return += public_reward
        self._cumulative_absolute_action += float(numpy.sum(numpy.abs(applied_action)))
        metrics = _transition_metrics(
            public_observation,
            applied_action,
            provider_metrics=provider_metrics,
            health=health,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            frame_skip=self._frame_skip,
            forward_reward_weight=self._forward_reward_weight,
            control_cost_weight=self._control_cost_weight,
            healthy_reward=self._healthy_reward,
            terminate_when_unhealthy=self._terminate_when_unhealthy,
            initial_x_position=self._initial_x_position,
            minimum_x_position=self._minimum_x_position,
            maximum_x_position=self._maximum_x_position,
            minimum_x_velocity=self._minimum_x_velocity,
            maximum_x_velocity=self._maximum_x_velocity,
            minimum_torso_z_position=self._minimum_torso_z_position,
            maximum_absolute_torso_pitch=self._maximum_absolute_torso_pitch,
            minimum_healthy_state_margin=self._minimum_healthy_state_margin,
            minimum_healthy_z_margin=self._minimum_healthy_z_margin,
            minimum_healthy_angle_margin=self._minimum_healthy_angle_margin,
            healthy_steps=self._healthy_steps,
            forward_steps=self._forward_steps,
            cumulative_forward_reward=self._cumulative_forward_reward,
            cumulative_control_reward=self._cumulative_control_reward,
            cumulative_survival_reward=self._cumulative_survival_reward,
            cumulative_return=self._cumulative_return,
            cumulative_absolute_action=self._cumulative_absolute_action,
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


class _Health:
    def __init__(
        self,
        *,
        healthy_state: bool,
        healthy_z: bool,
        healthy_angle: bool,
        state_margin: float,
        z_margin: float,
        angle_margin: float,
    ) -> None:
        self.healthy_state = healthy_state
        self.healthy_z = healthy_z
        self.healthy_angle = healthy_angle
        self.healthy = healthy_state and healthy_z and healthy_angle
        self.state_margin = state_margin
        self.z_margin = z_margin
        self.angle_margin = angle_margin


def _health(
    provider: _HopperProvider,
    *,
    healthy_state_range: tuple[float, float],
    healthy_z_range: tuple[float, float | None],
    healthy_angle_range: tuple[float, float],
) -> _Health:
    qpos = provider.data.qpos
    qvel = provider.data.qvel
    if (
        type(qpos) is not numpy.ndarray
        or qpos.shape != (6,)
        or qpos.dtype != numpy.dtype("float64")
        or type(qvel) is not numpy.ndarray
        or qvel.shape != (6,)
        or qvel.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Hopper returned an invalid internal state")
    state = numpy.concatenate((qpos[2:], qvel))
    state_lower, state_upper = healthy_state_range
    z_lower, z_upper_optional = healthy_z_range
    z_upper = math.inf if z_upper_optional is None else z_upper_optional
    angle_lower, angle_upper = healthy_angle_range
    z = float(qpos[1])
    angle = float(qpos[2])
    healthy_state = bool(numpy.all((state_lower < state) & (state < state_upper)))
    healthy_z = z_lower < z < z_upper
    healthy_angle = angle_lower < angle < angle_upper
    state_margin = float(numpy.min(numpy.minimum(state - state_lower, state_upper - state)))
    z_margin = z - z_lower if z_upper_optional is None else min(z - z_lower, z_upper_optional - z)
    angle_margin = min(angle - angle_lower, angle_upper - angle)
    if not all(math.isfinite(value) for value in (state_margin, z_margin, angle_margin)):
        raise RuntimeError("Hopper returned non-finite health diagnostics")
    return _Health(
        healthy_state=healthy_state,
        healthy_z=healthy_z,
        healthy_angle=healthy_angle,
        state_margin=state_margin,
        z_margin=z_margin,
        angle_margin=angle_margin,
    )


def _action(value: PolicyValue) -> NDArray[numpy.float32]:
    if type(value) is not list or len(value) != 3:
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
    expected_shape = (11,) if exclude_positions else (12,)
    if (
        type(value) is not numpy.ndarray
        or value.shape != expected_shape
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Hopper returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
        observation["torso_x_position"] = _number(
            value[0],
            name="torso x position",
        )
        offset = 1
    for name, item in zip(_BODY_FIELDS, value[offset:], strict=True):
        observation[name] = _number(item, name=name)
    return observation


def _provider_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("Hopper returned invalid metrics")
    names = (
        "x_position",
        "z_distance_from_origin",
        "x_velocity",
        "reward_forward",
        "reward_ctrl",
        "reward_survive",
    )
    if not set(names).issubset(value):
        raise RuntimeError("Hopper omitted public metrics")
    return {
        ("reward_control" if name == "reward_ctrl" else name): _number(
            value[name],
            name=name.replace("_", " "),
        )
        for name in names
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    action: NDArray[numpy.float32],
    *,
    provider_metrics: dict[str, float],
    health: _Health,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    frame_skip: int,
    forward_reward_weight: float,
    control_cost_weight: float,
    healthy_reward: float,
    terminate_when_unhealthy: bool,
    initial_x_position: float,
    minimum_x_position: float,
    maximum_x_position: float,
    minimum_x_velocity: float,
    maximum_x_velocity: float,
    minimum_torso_z_position: float,
    maximum_absolute_torso_pitch: float,
    minimum_healthy_state_margin: float,
    minimum_healthy_z_margin: float,
    minimum_healthy_angle_margin: float,
    healthy_steps: int,
    forward_steps: int,
    cumulative_forward_reward: float,
    cumulative_control_reward: float,
    cumulative_survival_reward: float,
    cumulative_return: float,
    cumulative_absolute_action: float,
) -> dict[str, PolicyValue]:
    reward_forward = provider_metrics["reward_forward"]
    reward_control = provider_metrics["reward_control"]
    reward_survive = provider_metrics["reward_survive"]
    if not math.isclose(
        reward,
        reward_forward + reward_control + reward_survive,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Hopper reward decomposition drifted")
    expected_control_reward = -control_cost_weight * float(numpy.sum(numpy.square(action)))
    if not math.isclose(
        reward_control,
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Hopper control-cost semantics drifted")
    if not math.isclose(
        reward_forward,
        forward_reward_weight * provider_metrics["x_velocity"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Hopper forward-reward semantics drifted")
    expected_survival_reward = healthy_reward if health.healthy else 0.0
    if not math.isclose(
        reward_survive,
        expected_survival_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Hopper survival-reward semantics drifted")
    if terminated != (terminate_when_unhealthy and not health.healthy):
        raise RuntimeError("Hopper health-termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Hopper time-limit semantics drifted")
    if "torso_x_position" in observation and not math.isclose(
        _float_field(observation, "torso_x_position"),
        provider_metrics["x_position"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Hopper x-position observation drifted")
    if not math.isclose(
        cumulative_return,
        cumulative_forward_reward + cumulative_control_reward + cumulative_survival_reward,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Hopper cumulative reward decomposition drifted")
    action_by_joint: dict[str, PolicyValue] = {}
    gear_scaled_controls: dict[str, PolicyValue] = {}
    for component, gear, item in zip(
        _ACTION_COMPONENTS,
        _ACTUATOR_GEARS,
        action,
        strict=True,
    ):
        value = float(item)
        action_by_joint[component] = value
        gear_scaled_controls[component] = value * gear
    seconds_per_step = frame_skip * _MODEL_TIMESTEP_SECONDS
    x_position = provider_metrics["x_position"]
    net_displacement = x_position - initial_x_position
    torso_pitch = _float_field(observation, "torso_pitch_angle")
    failed_conditions = tuple(
        name
        for name, passed in (
            ("state_range", health.healthy_state),
            ("torso_height", health.healthy_z),
            ("torso_pitch", health.healthy_angle),
        )
        if not passed
    )
    terminal_reason = "none"
    if terminated and truncated:
        terminal_reason = "unhealthy_and_time_limit"
    elif terminated:
        terminal_reason = "unhealthy"
    elif truncated:
        terminal_reason = "time_limit"
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": seconds_per_step,
        "simulated_seconds": step_count * seconds_per_step,
        "requested_action_by_joint": action_by_joint,
        "actuator_gear_scaled_controls": gear_scaled_controls,
        "sum_squared_action": float(numpy.sum(numpy.square(action))),
        "sum_absolute_action": float(numpy.sum(numpy.abs(action))),
        "cumulative_absolute_action": cumulative_absolute_action,
        "initial_x_position": initial_x_position,
        "x_position": x_position,
        "net_x_displacement": net_displacement,
        "minimum_x_position": minimum_x_position,
        "maximum_x_position": maximum_x_position,
        "x_velocity": provider_metrics["x_velocity"],
        "minimum_x_velocity": minimum_x_velocity,
        "maximum_x_velocity": maximum_x_velocity,
        "mean_x_velocity_from_displacement": (net_displacement / (step_count * seconds_per_step)),
        "forward_step_fraction": forward_steps / step_count,
        "torso_z_position": _float_field(observation, "torso_z_position"),
        "minimum_torso_z_position": minimum_torso_z_position,
        "torso_pitch_radians": torso_pitch,
        "torso_pitch_degrees": math.degrees(torso_pitch),
        "maximum_absolute_torso_pitch_radians": maximum_absolute_torso_pitch,
        "healthy": health.healthy,
        "healthy_state": health.healthy_state,
        "healthy_z": health.healthy_z,
        "healthy_angle": health.healthy_angle,
        "failed_health_conditions": list(failed_conditions),
        "healthy_state_margin": health.state_margin,
        "healthy_z_margin": health.z_margin,
        "healthy_angle_margin": health.angle_margin,
        "minimum_healthy_state_margin": minimum_healthy_state_margin,
        "minimum_healthy_z_margin": minimum_healthy_z_margin,
        "minimum_healthy_angle_margin": minimum_healthy_angle_margin,
        "healthy_step_fraction": healthy_steps / step_count,
        "z_distance_from_origin": provider_metrics["z_distance_from_origin"],
        "reward_forward": reward_forward,
        "reward_control": reward_control,
        "reward_survive": reward_survive,
        "reward_from_public_terms": (reward_forward + reward_control + reward_survive),
        "cumulative_reward_forward": cumulative_forward_reward,
        "cumulative_reward_control": cumulative_control_reward,
        "cumulative_reward_survive": cumulative_survival_reward,
        "cumulative_return": cumulative_return,
        "terminal_reason": terminal_reason,
    }


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Hopper returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Hopper returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Hopper returned a non-finite {name}")
    return number


__all__ = ["HopperEnvironment"]
