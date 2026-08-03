"""One fresh Gymnasium Humanoid-v5 Environment per Episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import HumanoidConfig

_JOINTS = (
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder_1",
    "right_shoulder_2",
    "right_elbow",
    "left_shoulder_1",
    "left_shoulder_2",
    "left_elbow",
)
_ACTION_COMPONENTS = (
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder_1",
    "right_shoulder_2",
    "right_elbow",
    "left_shoulder_1",
    "left_shoulder_2",
    "left_elbow",
)
_ACTUATOR_GEARS = (
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    300.0,
    200.0,
    100.0,
    100.0,
    300.0,
    200.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
)
_STATE_FIELDS = (
    "torso_z_position",
    "torso_orientation_w",
    "torso_orientation_x",
    "torso_orientation_y",
    "torso_orientation_z",
    *(f"{joint}_angle" for joint in _JOINTS),
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_x_angular_velocity",
    "torso_y_angular_velocity",
    "torso_z_angular_velocity",
    *(f"{joint}_angular_velocity" for joint in _JOINTS),
)
_BODIES = (
    "torso",
    "lower_waist",
    "pelvis",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_upper_arm",
    "right_lower_arm",
    "left_upper_arm",
    "left_lower_arm",
)
_INERTIA_COMPONENTS = (
    "inertia_upper_0",
    "inertia_upper_1",
    "inertia_upper_2",
    "inertia_upper_3",
    "inertia_upper_4",
    "inertia_upper_5",
    "mass_times_com_offset_x",
    "mass_times_com_offset_y",
    "mass_times_com_offset_z",
    "mass",
)
_BODY_VELOCITY_COMPONENTS = (
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
)
_EXTERNAL_FORCE_COMPONENTS = (
    "torque_x",
    "torque_y",
    "torque_z",
    "force_x",
    "force_y",
    "force_z",
)
_TENDONS = ("left_hip_to_knee", "right_hip_to_knee")
_MODEL_TIMESTEP_SECONDS = 0.003
_MAX_EPISODE_STEPS = 1_000


@dataclass(frozen=True)
class _ContactSummary:
    total_squared: float | None
    maximum_body_norm: float | None
    maximum_body: str


@dataclass(frozen=True)
class _ActuatorForceSummary:
    maximum_absolute_force: float | None
    maximum_force_joint: str


class HumanoidEnvironment:
    """The seeded strict adapter around configured Humanoid-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: HumanoidConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HumanoidConfig:
            raise TypeError("config must be HumanoidConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Humanoid configuration belongs in HumanoidConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        contact_lower = (
            float("-inf") if config.contact_cost_range[0] is None else config.contact_cost_range[0]
        )
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "Humanoid-v5",
                frame_skip=config.frame_skip,
                forward_reward_weight=config.forward_reward_weight,
                ctrl_cost_weight=config.ctrl_cost_weight,
                contact_cost_weight=config.contact_cost_weight,
                contact_cost_range=(
                    contact_lower,
                    config.contact_cost_range[1],
                ),
                healthy_reward=config.healthy_reward,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_z_range=config.healthy_z_range,
                reset_noise_scale=config.reset_noise_scale,
                exclude_current_positions_from_observation=(
                    config.exclude_current_positions_from_observation
                ),
                include_cinert_in_observation=(config.include_cinert_in_observation),
                include_cvel_in_observation=(config.include_cvel_in_observation),
                include_qfrc_actuator_in_observation=(config.include_qfrc_actuator_in_observation),
                include_cfrc_ext_in_observation=(config.include_cfrc_ext_in_observation),
            ),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._initial_x_position: float | None = None
        self._initial_y_position: float | None = None
        self._minimum_x_position = math.inf
        self._maximum_x_position = -math.inf
        self._minimum_x_velocity = math.inf
        self._maximum_x_velocity = -math.inf
        self._maximum_horizontal_speed = 0.0
        self._minimum_torso_z_position = math.inf
        self._maximum_torso_tilt_radians = 0.0
        self._minimum_healthy_z_margin = math.inf
        self._healthy_steps = 0
        self._forward_steps = 0
        self._cumulative_absolute_action = 0.0
        self._cumulative_forward_reward = 0.0
        self._cumulative_control_reward = 0.0
        self._cumulative_contact_reward = 0.0
        self._cumulative_survival_reward = 0.0
        self._cumulative_return = 0.0
        self._maximum_external_force_body_norm: float | None = None
        self._maximum_external_force_body = "unavailable"
        self._maximum_absolute_actuator_force: float | None = None
        self._maximum_actuator_force_joint = "unavailable"
        self._maximum_absolute_tendon_velocity = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, info = self._environment.reset(seed=self._seed)
        reset_metrics = _reset_metrics(info)
        self._initial_x_position = reset_metrics["x_position"]
        self._initial_y_position = reset_metrics["y_position"]
        self._minimum_x_position = self._initial_x_position
        self._maximum_x_position = self._initial_x_position
        self._started = True
        return _observation(observation, config=self._config)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if self._initial_x_position is None or self._initial_y_position is None:
            raise RuntimeError("Humanoid reset diagnostics are unavailable")

        applied_action = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(applied_action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("Humanoid returned invalid termination flags")
        public_observation = _observation(observation, config=self._config)
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        provider_scalars = _scalar_provider_metrics(provider_metrics)
        contact = _contact_summary(public_observation)
        actuator_force = _actuator_force_summary(public_observation)
        self._steps += 1
        x_position = provider_scalars["x_position"]
        x_velocity = provider_scalars["x_velocity"]
        y_velocity = provider_scalars["y_velocity"]
        horizontal_speed = math.hypot(x_velocity, y_velocity)
        torso_z = _float_field(public_observation, "torso_z_position")
        healthy, healthy_z_margin = _health(
            torso_z,
            healthy_z_range=self._config.healthy_z_range,
        )
        torso_tilt = _torso_tilt(public_observation)
        self._minimum_x_position = min(self._minimum_x_position, x_position)
        self._maximum_x_position = max(self._maximum_x_position, x_position)
        self._minimum_x_velocity = min(self._minimum_x_velocity, x_velocity)
        self._maximum_x_velocity = max(self._maximum_x_velocity, x_velocity)
        self._maximum_horizontal_speed = max(
            self._maximum_horizontal_speed,
            horizontal_speed,
        )
        self._minimum_torso_z_position = min(
            self._minimum_torso_z_position,
            torso_z,
        )
        self._maximum_torso_tilt_radians = max(
            self._maximum_torso_tilt_radians,
            torso_tilt,
        )
        self._minimum_healthy_z_margin = min(
            self._minimum_healthy_z_margin,
            healthy_z_margin,
        )
        self._healthy_steps += int(healthy)
        self._forward_steps += int(x_velocity > 0.0)
        self._cumulative_absolute_action += float(numpy.sum(numpy.abs(applied_action)))
        self._cumulative_forward_reward += provider_scalars["reward_forward"]
        self._cumulative_control_reward += provider_scalars["reward_control"]
        self._cumulative_contact_reward += provider_scalars["reward_contact"]
        self._cumulative_survival_reward += provider_scalars["reward_survive"]
        self._cumulative_return += public_reward
        if contact.maximum_body_norm is not None and (
            self._maximum_external_force_body_norm is None
            or contact.maximum_body_norm > self._maximum_external_force_body_norm
        ):
            self._maximum_external_force_body_norm = contact.maximum_body_norm
            self._maximum_external_force_body = contact.maximum_body
        if actuator_force.maximum_absolute_force is not None and (
            self._maximum_absolute_actuator_force is None
            or actuator_force.maximum_absolute_force > self._maximum_absolute_actuator_force
        ):
            self._maximum_absolute_actuator_force = actuator_force.maximum_absolute_force
            self._maximum_actuator_force_joint = actuator_force.maximum_force_joint
        tendon_velocities = _tendon_metric(
            provider_metrics,
            "tendon_velocities",
        )
        self._maximum_absolute_tendon_velocity = max(
            self._maximum_absolute_tendon_velocity,
            *(abs(value) for value in tendon_velocities.values()),
        )
        metrics = _transition_metrics(
            public_observation,
            applied_action,
            provider_metrics=provider_metrics,
            contact=contact,
            actuator_force=actuator_force,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            initial_x_position=self._initial_x_position,
            initial_y_position=self._initial_y_position,
            minimum_x_position=self._minimum_x_position,
            maximum_x_position=self._maximum_x_position,
            minimum_x_velocity=self._minimum_x_velocity,
            maximum_x_velocity=self._maximum_x_velocity,
            maximum_horizontal_speed=self._maximum_horizontal_speed,
            minimum_torso_z_position=self._minimum_torso_z_position,
            maximum_torso_tilt_radians=self._maximum_torso_tilt_radians,
            minimum_healthy_z_margin=self._minimum_healthy_z_margin,
            healthy_steps=self._healthy_steps,
            forward_steps=self._forward_steps,
            cumulative_absolute_action=self._cumulative_absolute_action,
            cumulative_forward_reward=self._cumulative_forward_reward,
            cumulative_control_reward=self._cumulative_control_reward,
            cumulative_contact_reward=self._cumulative_contact_reward,
            cumulative_survival_reward=self._cumulative_survival_reward,
            cumulative_return=self._cumulative_return,
            maximum_external_force_body_norm=(self._maximum_external_force_body_norm),
            maximum_external_force_body=self._maximum_external_force_body,
            maximum_absolute_actuator_force=(self._maximum_absolute_actuator_force),
            maximum_actuator_force_joint=(self._maximum_actuator_force_joint),
            maximum_absolute_tendon_velocity=(self._maximum_absolute_tendon_velocity),
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
    if type(value) is not list or len(value) != 17:
        raise InvalidAction()
    action: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or not -0.4 <= item <= 0.4:
            raise InvalidAction()
        action.append(item)
    return numpy.asarray(action, dtype=numpy.float32)


def _observation(
    value: object,
    *,
    config: HumanoidConfig,
) -> dict[str, PolicyValue]:
    expected_length = (
        len(_STATE_FIELDS)
        + (0 if config.exclude_current_positions_from_observation else 2)
        + 130 * config.include_cinert_in_observation
        + 78 * config.include_cvel_in_observation
        + 17 * config.include_qfrc_actuator_in_observation
        + 78 * config.include_cfrc_ext_in_observation
    )
    if (
        type(value) is not numpy.ndarray
        or value.shape != (expected_length,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError("Humanoid returned an invalid observation")
    offset = 0
    observation: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
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
        _STATE_FIELDS,
        value[offset : offset + len(_STATE_FIELDS)],
        strict=True,
    ):
        observation[name] = _number(item, name=name)
    offset += len(_STATE_FIELDS)
    if config.include_cinert_in_observation:
        observation["body_inertias"] = _body_matrix(
            value[offset : offset + 130],
            components=_INERTIA_COMPONENTS,
            name="body inertia",
        )
        offset += 130
    if config.include_cvel_in_observation:
        observation["body_velocities"] = _body_matrix(
            value[offset : offset + 78],
            components=_BODY_VELOCITY_COMPONENTS,
            name="body velocity",
        )
        offset += 78
    if config.include_qfrc_actuator_in_observation:
        observation["actuator_forces"] = {
            joint: _number(
                value[offset + index],
                name=f"{joint} actuator force",
            )
            for index, joint in enumerate(_JOINTS)
        }
        offset += 17
    if config.include_cfrc_ext_in_observation:
        observation["external_forces"] = _body_matrix(
            value[offset : offset + 78],
            components=_EXTERNAL_FORCE_COMPONENTS,
            name="external force",
        )
    return observation


def _body_matrix(
    value: NDArray[numpy.float64],
    *,
    components: tuple[str, ...],
    name: str,
) -> PolicyValue:
    if value.shape != (len(_BODIES) * len(components),):
        raise RuntimeError(f"Humanoid returned invalid {name} values")
    bodies: dict[str, PolicyValue] = {}
    for body_index, body in enumerate(_BODIES):
        start = body_index * len(components)
        bodies[body] = {
            component: _number(
                value[start + component_index],
                name=f"{body} {name} {component}",
            )
            for component_index, component in enumerate(components)
        }
    return bodies


def _reset_metrics(value: object) -> dict[str, float]:
    if type(value) is not dict or not {"x_position", "y_position"}.issubset(value):
        raise RuntimeError("Humanoid omitted reset position metrics")
    return {
        "x_position": _number(value["x_position"], name="reset x position"),
        "y_position": _number(value["y_position"], name="reset y position"),
    }


def _provider_metrics(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError("Humanoid returned invalid metrics")
    scalar_names = (
        "x_position",
        "y_position",
        "distance_from_origin",
        "x_velocity",
        "y_velocity",
        "reward_survive",
        "reward_forward",
        "reward_ctrl",
        "reward_contact",
    )
    if not set((*scalar_names, "tendon_length", "tendon_velocity")).issubset(value):
        raise RuntimeError("Humanoid omitted public metrics")
    metrics: dict[str, object] = {
        ("reward_control" if name == "reward_ctrl" else name): _number(
            value[name],
            name=name.replace("_", " "),
        )
        for name in scalar_names
    }
    metrics["tendon_lengths"] = _tendons(
        value["tendon_length"],
        name="tendon length",
    )
    metrics["tendon_velocities"] = _tendons(
        value["tendon_velocity"],
        name="tendon velocity",
    )
    return metrics


def _tendons(value: object, *, name: str) -> dict[str, float]:
    if (
        type(value) is not numpy.ndarray
        or value.shape != (2,)
        or value.dtype != numpy.dtype("float64")
    ):
        raise RuntimeError(f"Humanoid returned invalid {name} values")
    return {
        "left_hip_to_knee": _number(value[0], name=f"left {name}"),
        "right_hip_to_knee": _number(value[1], name=f"right {name}"),
    }


def _transition_metrics(
    observation: dict[str, PolicyValue],
    action: NDArray[numpy.float32],
    *,
    provider_metrics: dict[str, object],
    contact: _ContactSummary,
    actuator_force: _ActuatorForceSummary,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: HumanoidConfig,
    initial_x_position: float,
    initial_y_position: float,
    minimum_x_position: float,
    maximum_x_position: float,
    minimum_x_velocity: float,
    maximum_x_velocity: float,
    maximum_horizontal_speed: float,
    minimum_torso_z_position: float,
    maximum_torso_tilt_radians: float,
    minimum_healthy_z_margin: float,
    healthy_steps: int,
    forward_steps: int,
    cumulative_absolute_action: float,
    cumulative_forward_reward: float,
    cumulative_control_reward: float,
    cumulative_contact_reward: float,
    cumulative_survival_reward: float,
    cumulative_return: float,
    maximum_external_force_body_norm: float | None,
    maximum_external_force_body: str,
    maximum_absolute_actuator_force: float | None,
    maximum_actuator_force_joint: str,
    maximum_absolute_tendon_velocity: float,
) -> dict[str, PolicyValue]:
    scalars = _scalar_provider_metrics(provider_metrics)
    reward_forward = scalars["reward_forward"]
    reward_control = scalars["reward_control"]
    reward_contact = scalars["reward_contact"]
    reward_survive = scalars["reward_survive"]
    reconstructed_reward = reward_forward + reward_control + reward_contact + reward_survive
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Humanoid reward decomposition drifted")
    expected_control_reward = -config.ctrl_cost_weight * float(numpy.sum(numpy.square(action)))
    if not math.isclose(
        reward_control,
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Humanoid control-cost semantics drifted")
    if not math.isclose(
        reward_forward,
        config.forward_reward_weight * scalars["x_velocity"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Humanoid forward-reward semantics drifted")
    torso_z = _float_field(observation, "torso_z_position")
    healthy, healthy_z_margin = _health(
        torso_z,
        healthy_z_range=config.healthy_z_range,
    )
    expected_survival_reward = config.healthy_reward if healthy else 0.0
    if not math.isclose(
        reward_survive,
        expected_survival_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Humanoid healthy-reward semantics drifted")
    if terminated != (config.terminate_when_unhealthy and not healthy):
        raise RuntimeError("Humanoid health termination semantics drifted")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("Humanoid time-limit semantics drifted")
    if not math.isclose(
        scalars["distance_from_origin"],
        math.hypot(scalars["x_position"], scalars["y_position"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Humanoid root-position metrics drifted")
    raw_contact_cost: float | None = None
    if contact.total_squared is not None:
        raw_contact_cost = config.contact_cost_weight * contact.total_squared
        lower, upper = config.contact_cost_range
        clamped_contact_cost = min(
            max(raw_contact_cost, -math.inf if lower is None else lower),
            upper,
        )
        if not math.isclose(
            reward_contact,
            -clamped_contact_cost,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Humanoid contact-cost semantics drifted")
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
        raise RuntimeError("Humanoid cumulative reward decomposition drifted")
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
    tendon_lengths = _tendon_metric(provider_metrics, "tendon_lengths")
    tendon_velocities = _tendon_metric(provider_metrics, "tendon_velocities")
    public_tendon_lengths: dict[str, PolicyValue] = dict(tendon_lengths)
    public_tendon_velocities: dict[str, PolicyValue] = dict(tendon_velocities)
    torso_tilt = _torso_tilt(observation)
    quaternion_norm_error = _quaternion_norm_error(observation)
    terminal_reason = "none"
    if terminated and truncated:
        terminal_reason = "unhealthy_and_time_limit"
    elif terminated:
        terminal_reason = "unhealthy"
    elif truncated:
        terminal_reason = "time_limit"
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    x_position = scalars["x_position"]
    y_position = scalars["y_position"]
    net_x_displacement = x_position - initial_x_position
    net_y_displacement = y_position - initial_y_position
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
        "initial_y_position": initial_y_position,
        "x_position": x_position,
        "y_position": y_position,
        "net_x_displacement": net_x_displacement,
        "net_y_displacement": net_y_displacement,
        "distance_from_origin": scalars["distance_from_origin"],
        "minimum_x_position": minimum_x_position,
        "maximum_x_position": maximum_x_position,
        "center_of_mass_x_velocity": scalars["x_velocity"],
        "center_of_mass_y_velocity": scalars["y_velocity"],
        "horizontal_center_of_mass_speed": math.hypot(
            scalars["x_velocity"],
            scalars["y_velocity"],
        ),
        "minimum_center_of_mass_x_velocity": minimum_x_velocity,
        "maximum_center_of_mass_x_velocity": maximum_x_velocity,
        "maximum_horizontal_center_of_mass_speed": maximum_horizontal_speed,
        "forward_step_fraction": forward_steps / step_count,
        "mean_root_x_velocity_from_displacement": (
            net_x_displacement / (step_count * seconds_per_step)
        ),
        "torso_z_position": torso_z,
        "minimum_torso_z_position": minimum_torso_z_position,
        "healthy": healthy,
        "healthy_z_lower_bound": config.healthy_z_range[0],
        "healthy_z_upper_bound": config.healthy_z_range[1],
        "healthy_z_margin": healthy_z_margin,
        "minimum_healthy_z_margin": minimum_healthy_z_margin,
        "healthy_step_fraction": healthy_steps / step_count,
        "torso_tilt_radians": torso_tilt,
        "torso_tilt_degrees": math.degrees(torso_tilt),
        "maximum_torso_tilt_radians": maximum_torso_tilt_radians,
        "quaternion_norm_error": quaternion_norm_error,
        "external_forces_in_observation": (config.include_cfrc_ext_in_observation),
        "sum_squared_external_force_components": contact.total_squared,
        "raw_contact_cost_before_clamp": raw_contact_cost,
        "maximum_external_force_body_norm_this_step": (contact.maximum_body_norm),
        "maximum_external_force_body_this_step": contact.maximum_body,
        "maximum_external_force_body_norm": maximum_external_force_body_norm,
        "maximum_external_force_body": maximum_external_force_body,
        "actuator_forces_in_observation": (config.include_qfrc_actuator_in_observation),
        "maximum_absolute_actuator_force_this_step": (actuator_force.maximum_absolute_force),
        "maximum_actuator_force_joint_this_step": (actuator_force.maximum_force_joint),
        "maximum_absolute_actuator_force": maximum_absolute_actuator_force,
        "maximum_actuator_force_joint": maximum_actuator_force_joint,
        "tendon_lengths": public_tendon_lengths,
        "tendon_velocities": public_tendon_velocities,
        "maximum_absolute_tendon_velocity": maximum_absolute_tendon_velocity,
        "reward_survive": reward_survive,
        "reward_forward": reward_forward,
        "reward_control": reward_control,
        "reward_contact": reward_contact,
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_survive": cumulative_survival_reward,
        "cumulative_reward_forward": cumulative_forward_reward,
        "cumulative_reward_control": cumulative_control_reward,
        "cumulative_reward_contact": cumulative_contact_reward,
        "cumulative_return": cumulative_return,
        "terminal_reason": terminal_reason,
    }


def _scalar_provider_metrics(value: dict[str, object]) -> dict[str, float]:
    names = (
        "x_position",
        "y_position",
        "distance_from_origin",
        "x_velocity",
        "y_velocity",
        "reward_survive",
        "reward_forward",
        "reward_control",
        "reward_contact",
    )
    scalars: dict[str, float] = {}
    for name in names:
        item = value.get(name)
        if type(item) is not float:
            raise RuntimeError(f"Humanoid returned invalid {name}")
        scalars[name] = item
    return scalars


def _tendon_metric(value: dict[str, object], name: str) -> dict[str, float]:
    item = value.get(name)
    if type(item) is not dict or set(item) != set(_TENDONS):
        raise RuntimeError(f"Humanoid returned invalid {name}")
    result: dict[str, float] = {}
    for tendon in _TENDONS:
        number = item[tendon]
        if type(number) is not float:
            raise RuntimeError(f"Humanoid returned invalid {name}")
        result[tendon] = number
    return result


def _health(
    torso_z: float,
    *,
    healthy_z_range: tuple[float, float],
) -> tuple[bool, float]:
    lower, upper = healthy_z_range
    return lower < torso_z < upper, min(torso_z - lower, upper - torso_z)


def _torso_tilt(observation: dict[str, PolicyValue]) -> float:
    w = _float_field(observation, "torso_orientation_w")
    x = _float_field(observation, "torso_orientation_x")
    y = _float_field(observation, "torso_orientation_y")
    z = _float_field(observation, "torso_orientation_z")
    norm_squared = w**2 + x**2 + y**2 + z**2
    if norm_squared <= 0.0:
        raise RuntimeError("Humanoid returned an invalid torso quaternion")
    up_z = 1.0 - 2.0 * (x**2 + y**2) / norm_squared
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
) -> _ContactSummary:
    raw_forces = observation.get("external_forces")
    if raw_forces is None:
        return _ContactSummary(None, None, "unavailable")
    if type(raw_forces) is not dict or set(raw_forces) != set(_BODIES):
        raise RuntimeError("Humanoid returned invalid external forces")
    total_squared = 0.0
    maximum_norm = 0.0
    maximum_body = _BODIES[0]
    for body in _BODIES:
        raw_components = raw_forces[body]
        if type(raw_components) is not dict or set(raw_components) != set(
            _EXTERNAL_FORCE_COMPONENTS
        ):
            raise RuntimeError("Humanoid returned invalid external forces")
        squared = 0.0
        for component in _EXTERNAL_FORCE_COMPONENTS:
            value = raw_components[component]
            if type(value) is not float:
                raise RuntimeError("Humanoid returned invalid external forces")
            squared += value**2
        total_squared += squared
        norm = math.sqrt(squared)
        if norm > maximum_norm:
            maximum_norm = norm
            maximum_body = body
    return _ContactSummary(total_squared, maximum_norm, maximum_body)


def _actuator_force_summary(
    observation: dict[str, PolicyValue],
) -> _ActuatorForceSummary:
    raw_forces = observation.get("actuator_forces")
    if raw_forces is None:
        return _ActuatorForceSummary(None, "unavailable")
    if type(raw_forces) is not dict or set(raw_forces) != set(_JOINTS):
        raise RuntimeError("Humanoid returned invalid actuator forces")
    maximum_force = 0.0
    maximum_joint = _JOINTS[0]
    for joint in _JOINTS:
        value = raw_forces[joint]
        if type(value) is not float:
            raise RuntimeError("Humanoid returned invalid actuator forces")
        absolute = abs(value)
        if absolute > maximum_force:
            maximum_force = absolute
            maximum_joint = joint
    return _ActuatorForceSummary(maximum_force, maximum_joint)


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"Humanoid returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"Humanoid returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Humanoid returned a non-finite {name}")
    return number


__all__ = ["HumanoidEnvironment"]
