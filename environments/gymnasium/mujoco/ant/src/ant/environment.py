"""One fresh Gymnasium Ant-v5 Environment per Episode."""

from __future__ import annotations

import math
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import AntConfig

_BODY_FIELDS = (
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    "front_left_hip_angle",
    "front_left_ankle_angle",
    "front_right_hip_angle",
    "front_right_ankle_angle",
    "back_left_hip_angle",
    "back_left_ankle_angle",
    "back_right_hip_angle",
    "back_right_ankle_angle",
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    "front_left_hip_angular_velocity",
    "front_left_ankle_angular_velocity",
    "front_right_hip_angular_velocity",
    "front_right_ankle_angular_velocity",
    "back_left_hip_angular_velocity",
    "back_left_ankle_angular_velocity",
    "back_right_hip_angular_velocity",
    "back_right_ankle_angular_velocity",
)
_CONTACT_BODIES = (
    "torso",
    "front_left_leg",
    "front_left_aux",
    "front_left_ankle",
    "front_right_leg",
    "front_right_aux",
    "front_right_ankle",
    "back_left_leg",
    "back_left_aux",
    "back_left_ankle",
    "back_right_leg",
    "back_right_aux",
    "back_right_ankle",
)
_CONTACT_COMPONENTS = (
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
)
_ACTION_COMPONENTS = (
    "back_right_hip",
    "back_right_ankle",
    "front_left_hip",
    "front_left_ankle",
    "front_right_hip",
    "front_right_ankle",
    "back_left_hip",
    "back_left_ankle",
)
_MAX_EPISODE_STEPS = 1_000
_MODEL_TIMESTEP_SECONDS = 0.01
_ACTUATOR_GEAR = 150.0


class AntEnvironment:
    """The seeded strict adapter around configured Ant-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: AntConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not AntConfig:
            raise TypeError("config must be AntConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Ant configuration belongs in AntConfig, "
                "not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._exclude_positions = (
            config.exclude_current_positions_from_observation
        )
        self._include_contact = config.include_cfrc_ext_in_observation
        self._frame_skip = config.frame_skip
        self._forward_reward_weight = config.forward_reward_weight
        self._control_cost_weight = config.ctrl_cost_weight
        self._contact_cost_weight = config.contact_cost_weight
        self._healthy_reward = config.healthy_reward
        self._terminate_when_unhealthy = config.terminate_when_unhealthy
        self._healthy_z_range = config.healthy_z_range
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Ant-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                contact_cost_weight=config.contact_cost_weight,
                healthy_reward=config.healthy_reward,
                main_body=config.main_body,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_z_range=config.healthy_z_range,
                contact_force_range=config.contact_force_range,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
                include_cfrc_ext_in_observation=(
                    config.include_cfrc_ext_in_observation
                ),
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._cumulative_forward_reward = 0.0
        self._cumulative_control_reward = 0.0
        self._cumulative_contact_reward = 0.0
        self._cumulative_survival_reward = 0.0
        self._cumulative_return = 0.0
        self._cumulative_absolute_action = 0.0
        self._healthy_steps = 0
        self._minimum_x_position = math.inf
        self._maximum_x_position = -math.inf
        self._maximum_torso_tilt_radians = 0.0
        self._minimum_healthy_z_margin = math.inf

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
            include_contact=self._include_contact,
        )

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        applied_action = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(
            applied_action
        )
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Ant returned invalid termination flags")
        public_observation = _observation(
            observation,
            exclude_positions=self._exclude_positions,
            include_contact=self._include_contact,
        )
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        self._steps += 1
        reward_forward = provider_metrics["reward_forward"]
        reward_control = provider_metrics["reward_control"]
        reward_contact = provider_metrics["reward_contact"]
        reward_survive = provider_metrics["reward_survive"]
        x_position = provider_metrics["x_position"]
        self._cumulative_forward_reward += reward_forward
        self._cumulative_control_reward += reward_control
        self._cumulative_contact_reward += reward_contact
        self._cumulative_survival_reward += reward_survive
        self._cumulative_return += public_reward
        self._cumulative_absolute_action += float(
            numpy.sum(numpy.abs(applied_action))
        )
        healthy, healthy_z_margin = _health(
            public_observation,
            healthy_z_range=self._healthy_z_range,
        )
        self._healthy_steps += int(healthy)
        self._minimum_x_position = min(self._minimum_x_position, x_position)
        self._maximum_x_position = max(self._maximum_x_position, x_position)
        torso_tilt = _torso_tilt(public_observation)
        self._maximum_torso_tilt_radians = max(
            self._maximum_torso_tilt_radians,
            torso_tilt,
        )
        self._minimum_healthy_z_margin = min(
            self._minimum_healthy_z_margin,
            healthy_z_margin,
        )
        metrics = _transition_metrics(
            public_observation,
            applied_action,
            provider_metrics=provider_metrics,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            frame_skip=self._frame_skip,
            forward_reward_weight=self._forward_reward_weight,
            control_cost_weight=self._control_cost_weight,
            contact_cost_weight=self._contact_cost_weight,
            healthy_reward=self._healthy_reward,
            terminate_when_unhealthy=self._terminate_when_unhealthy,
            healthy_z_range=self._healthy_z_range,
            include_contact=self._include_contact,
            cumulative_forward_reward=self._cumulative_forward_reward,
            cumulative_control_reward=self._cumulative_control_reward,
            cumulative_contact_reward=self._cumulative_contact_reward,
            cumulative_survival_reward=self._cumulative_survival_reward,
            cumulative_return=self._cumulative_return,
            cumulative_absolute_action=self._cumulative_absolute_action,
            healthy_steps=self._healthy_steps,
            minimum_x_position=self._minimum_x_position,
            maximum_x_position=self._maximum_x_position,
            maximum_torso_tilt_radians=self._maximum_torso_tilt_radians,
            minimum_healthy_z_margin=self._minimum_healthy_z_margin,
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
    if type(value) is not list or len(value) != 8:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or not -1.0 <= item <= 1.0
        ):
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    exclude_positions: bool,
    include_contact: bool,
) -> dict[str, PolicyValue]:
    expected_length = (
        27
        + (0 if exclude_positions else 2)
        + (78 if include_contact else 0)
    )
    if (
        type(value) is not numpy.ndarray
        or value.shape != (expected_length,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Ant returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not exclude_positions:
        observation["torso_x_position"] = _number(
            value[0],
            name="torso x position",
        )
        observation["torso_y_position"] = _number(
            value[1],
            name="torso y position",
        )
        offset = 2
    for name, item in zip(
        _BODY_FIELDS,
        value[offset : offset + len(_BODY_FIELDS)],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    offset += len(_BODY_FIELDS)
    if include_contact:
        observation["contact_forces"] = _contact_forces(
            value[offset:],
        )
    return observation


def _contact_forces(value: NDArray[numpy.float64]) -> PolicyValue:
    if value.shape != (78,):
        raise RuntimeError("Ant returned invalid contact forces")
    contacts: dict[str, PolicyValue] = {}
    for body_index, body in enumerate(_CONTACT_BODIES):
        start = body_index * len(_CONTACT_COMPONENTS)
        contacts[body] = {
            component: _number(
                value[start + component_index],
                name=f"{body} contact {component}",
            )
            for component_index, component in enumerate(_CONTACT_COMPONENTS)
        }
    return contacts


def _provider_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict:
        raise RuntimeError("Ant returned invalid metrics")
    names = (
        "x_position",
        "y_position",
        "distance_from_origin",
        "x_velocity",
        "y_velocity",
        "reward_forward",
        "reward_ctrl",
        "reward_contact",
        "reward_survive",
    )
    if not set(names).issubset(value):
        raise RuntimeError("Ant omitted public metrics")
    return {
        (
            "reward_control" if name == "reward_ctrl" else name
        ): _number(value[name], name=name.replace("_", " "))
        for name in names
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    action: NDArray[numpy.float32],
    *,
    provider_metrics: dict[str, float],
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    frame_skip: int,
    forward_reward_weight: float,
    control_cost_weight: float,
    contact_cost_weight: float,
    healthy_reward: float,
    terminate_when_unhealthy: bool,
    healthy_z_range: tuple[float, float],
    include_contact: bool,
    cumulative_forward_reward: float,
    cumulative_control_reward: float,
    cumulative_contact_reward: float,
    cumulative_survival_reward: float,
    cumulative_return: float,
    cumulative_absolute_action: float,
    healthy_steps: int,
    minimum_x_position: float,
    maximum_x_position: float,
    maximum_torso_tilt_radians: float,
    minimum_healthy_z_margin: float,
) -> dict[str, PolicyValue]:
    reward_forward = provider_metrics["reward_forward"]
    reward_control = provider_metrics["reward_control"]
    reward_contact = provider_metrics["reward_contact"]
    reward_survive = provider_metrics["reward_survive"]
    reconstructed_reward = (
        reward_forward + reward_control + reward_contact + reward_survive
    )
    if not math.isclose(reward, reconstructed_reward, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("Ant reward decomposition drifted")
    expected_control_reward = -control_cost_weight * float(
        numpy.sum(numpy.square(action))
    )
    if not math.isclose(
        reward_control,
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Ant control-cost semantics drifted")
    if not math.isclose(
        reward_forward,
        forward_reward_weight * provider_metrics["x_velocity"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Ant forward-reward semantics drifted")
    healthy, healthy_z_margin = _health(
        observation,
        healthy_z_range=healthy_z_range,
    )
    expected_survival_reward = healthy_reward if healthy else 0.0
    if not math.isclose(
        reward_survive,
        expected_survival_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Ant healthy-reward semantics drifted")
    if terminated != (terminate_when_unhealthy and not healthy):
        raise RuntimeError("Ant health termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Ant time-limit semantics drifted")
    if not math.isclose(
        provider_metrics["distance_from_origin"],
        math.hypot(provider_metrics["x_position"], provider_metrics["y_position"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Ant position metrics drifted")
    contact_summary = _contact_summary(observation)
    if include_contact:
        expected_contact_reward = -contact_cost_weight * contact_summary[0]
        if not math.isclose(
            reward_contact,
            expected_contact_reward,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError("Ant contact-cost semantics drifted")
    cumulative_reconstructed_reward = (
        cumulative_forward_reward
        + cumulative_control_reward
        + cumulative_contact_reward
        + cumulative_survival_reward
    )
    if not math.isclose(
        cumulative_return,
        cumulative_reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Ant cumulative reward decomposition drifted")
    action_by_joint: dict[str, PolicyValue] = {}
    gear_scaled_controls: dict[str, PolicyValue] = {}
    for component, item in zip(_ACTION_COMPONENTS, action, strict=True):
        value = float(item)
        action_by_joint[component] = value
        gear_scaled_controls[component] = value * _ACTUATOR_GEAR
    torso_tilt = _torso_tilt(observation)
    quaternion_norm_error = _quaternion_norm_error(observation)
    terminal_reason = (
        "unhealthy"
        if terminated
        else "time_limit"
        if truncated
        else "none"
    )
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "seconds_per_step": frame_skip * _MODEL_TIMESTEP_SECONDS,
        "simulated_seconds": step_count * frame_skip * _MODEL_TIMESTEP_SECONDS,
        "requested_action_by_joint": action_by_joint,
        "actuator_gear_scaled_controls": gear_scaled_controls,
        "sum_squared_action": float(numpy.sum(numpy.square(action))),
        "sum_absolute_action": float(numpy.sum(numpy.abs(action))),
        "cumulative_absolute_action": cumulative_absolute_action,
        "x_position": provider_metrics["x_position"],
        "y_position": provider_metrics["y_position"],
        "distance_from_origin": provider_metrics["distance_from_origin"],
        "x_velocity": provider_metrics["x_velocity"],
        "y_velocity": provider_metrics["y_velocity"],
        "speed_in_horizontal_plane": math.hypot(
            provider_metrics["x_velocity"],
            provider_metrics["y_velocity"],
        ),
        "minimum_x_position": minimum_x_position,
        "maximum_x_position": maximum_x_position,
        "torso_z_position": _float_field(observation, "torso_z_position"),
        "healthy": healthy,
        "healthy_z_lower_bound": healthy_z_range[0],
        "healthy_z_upper_bound": healthy_z_range[1],
        "healthy_z_margin": healthy_z_margin,
        "minimum_healthy_z_margin": minimum_healthy_z_margin,
        "healthy_step_fraction": healthy_steps / step_count,
        "torso_tilt_radians": torso_tilt,
        "torso_tilt_degrees": math.degrees(torso_tilt),
        "maximum_torso_tilt_radians": maximum_torso_tilt_radians,
        "quaternion_norm_error": quaternion_norm_error,
        "contact_forces_in_observation": include_contact,
        "sum_squared_clipped_contact_force_components": contact_summary[0],
        "maximum_contact_body_norm": contact_summary[1],
        "maximum_contact_body": contact_summary[2],
        "reward_forward": reward_forward,
        "reward_control": reward_control,
        "reward_contact": reward_contact,
        "reward_survive": reward_survive,
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_forward": cumulative_forward_reward,
        "cumulative_reward_control": cumulative_control_reward,
        "cumulative_reward_contact": cumulative_contact_reward,
        "cumulative_reward_survive": cumulative_survival_reward,
        "cumulative_return": cumulative_return,
        "terminal_reason": terminal_reason,
    }


def _health(
    observation: dict[str, PolicyValue],
    *,
    healthy_z_range: tuple[float, float],
) -> tuple[bool, float]:
    z_position = _float_field(observation, "torso_z_position")
    lower, upper = healthy_z_range
    margin = min(z_position - lower, upper - z_position)
    return lower <= z_position <= upper, margin


def _torso_tilt(observation: dict[str, PolicyValue]) -> float:
    x = _float_field(observation, "torso_orientation_x")
    y = _float_field(observation, "torso_orientation_y")
    up_z = 1.0 - 2.0 * (x**2 + y**2)
    return math.acos(min(max(up_z, -1.0), 1.0))


def _quaternion_norm_error(observation: dict[str, PolicyValue]) -> float:
    components = (
        _float_field(observation, "torso_orientation_w"),
        _float_field(observation, "torso_orientation_x"),
        _float_field(observation, "torso_orientation_y"),
        _float_field(observation, "torso_orientation_z"),
    )
    return abs(math.fsum(value**2 for value in components) - 1.0)


def _contact_summary(
    observation: dict[str, PolicyValue],
) -> tuple[float, float, str]:
    raw_contacts = observation.get("contact_forces")
    if raw_contacts is None:
        return 0.0, 0.0, "unavailable"
    if type(raw_contacts) is not dict or set(raw_contacts) != set(_CONTACT_BODIES):
        raise RuntimeError("Ant returned invalid contact forces")
    total_squared = 0.0
    maximum_norm = 0.0
    maximum_body = _CONTACT_BODIES[0]
    for body in _CONTACT_BODIES:
        raw_components = raw_contacts[body]
        if (
            type(raw_components) is not dict
            or set(raw_components) != set(_CONTACT_COMPONENTS)
        ):
            raise RuntimeError("Ant returned invalid contact forces")
        squared = 0.0
        for component in _CONTACT_COMPONENTS:
            value = raw_components[component]
            if type(value) is not float:
                raise RuntimeError("Ant returned invalid contact forces")
            squared += value**2
        total_squared += squared
        norm = math.sqrt(squared)
        if norm > maximum_norm:
            maximum_norm = norm
            maximum_body = body
    return total_squared, maximum_norm, maximum_body


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Ant returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Ant returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Ant returned a non-finite {name}")
    return number


__all__ = ["AntEnvironment"]
