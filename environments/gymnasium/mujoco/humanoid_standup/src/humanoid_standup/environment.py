"""One fresh Gymnasium HumanoidStandup-v5 Environment per Episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import SupportsFloat, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue
from numpy.typing import NDArray

from .config import HumanoidStandupConfig

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
_NOMINAL_PRONE_TORSO_Z = 0.105
_MAX_EPISODE_STEPS = 1_000


@dataclass(frozen=True)
class _ForceSummary:
    total_squared: float | None
    maximum_norm: float | None
    maximum_name: str


class HumanoidStandupEnvironment:
    """The seeded strict adapter around configured HumanoidStandup-v5."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: HumanoidStandupConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not HumanoidStandupConfig:
            raise TypeError("config must be HumanoidStandupConfig")
        if episode.scenario is not None:
            raise ValueError(
                "HumanoidStandup configuration belongs in "
                "HumanoidStandupConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        impact_lower = (
            float("-inf") if config.impact_cost_range[0] is None else config.impact_cost_range[0]
        )
        self._environment = cast(
            gymnasium.Env[object, object],
            gymnasium.make(
                "HumanoidStandup-v5",
                frame_skip=config.frame_skip,
                ctrl_cost_weight=config.ctrl_cost_weight,
                impact_cost_weight=config.impact_cost_weight,
                impact_cost_range=(
                    impact_lower,
                    config.impact_cost_range[1],
                ),
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
        self._initial_torso_z_position: float | None = None
        self._minimum_torso_z_position = math.inf
        self._maximum_torso_z_position = -math.inf
        self._minimum_torso_z_velocity = math.inf
        self._maximum_torso_z_velocity = -math.inf
        self._minimum_torso_tilt_radians = math.inf
        self._maximum_torso_tilt_radians = 0.0
        self._upward_steps = 0
        self._cumulative_absolute_action = 0.0
        self._cumulative_upward_reward = 0.0
        self._cumulative_control_reward = 0.0
        self._cumulative_impact_reward = 0.0
        self._cumulative_constant_reward = 0.0
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
        public_observation = _observation(observation, config=self._config)
        reset_metrics = _reset_metrics(info)
        self._initial_x_position = reset_metrics["x_position"]
        self._initial_y_position = reset_metrics["y_position"]
        self._initial_torso_z_position = _float_field(
            public_observation,
            "torso_z_position",
        )
        self._minimum_torso_z_position = self._initial_torso_z_position
        self._maximum_torso_z_position = self._initial_torso_z_position
        initial_tilt = _torso_tilt(public_observation)
        self._minimum_torso_tilt_radians = initial_tilt
        self._maximum_torso_tilt_radians = initial_tilt
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
            self._initial_x_position is None
            or self._initial_y_position is None
            or self._initial_torso_z_position is None
        ):
            raise RuntimeError("HumanoidStandup reset diagnostics are unavailable")

        applied_action = _action(action)
        observation, reward, terminated, truncated, info = self._environment.step(applied_action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("HumanoidStandup returned invalid termination flags")
        public_observation = _observation(observation, config=self._config)
        public_reward = _number(reward, name="reward")
        provider_metrics = _provider_metrics(info)
        scalars = _scalar_provider_metrics(provider_metrics)
        external_force = _external_force_summary(public_observation)
        actuator_force = _actuator_force_summary(public_observation)
        self._steps += 1
        torso_z = _float_field(public_observation, "torso_z_position")
        torso_z_velocity = _float_field(
            public_observation,
            "torso_z_velocity",
        )
        torso_tilt = _torso_tilt(public_observation)
        self._minimum_torso_z_position = min(
            self._minimum_torso_z_position,
            torso_z,
        )
        self._maximum_torso_z_position = max(
            self._maximum_torso_z_position,
            torso_z,
        )
        self._minimum_torso_z_velocity = min(
            self._minimum_torso_z_velocity,
            torso_z_velocity,
        )
        self._maximum_torso_z_velocity = max(
            self._maximum_torso_z_velocity,
            torso_z_velocity,
        )
        self._minimum_torso_tilt_radians = min(
            self._minimum_torso_tilt_radians,
            torso_tilt,
        )
        self._maximum_torso_tilt_radians = max(
            self._maximum_torso_tilt_radians,
            torso_tilt,
        )
        self._upward_steps += int(torso_z_velocity > 0.0)
        self._cumulative_absolute_action += float(numpy.sum(numpy.abs(applied_action)))
        self._cumulative_upward_reward += scalars["reward_upward"]
        self._cumulative_control_reward += scalars["reward_control"]
        self._cumulative_impact_reward += scalars["reward_impact"]
        self._cumulative_constant_reward += 1.0
        self._cumulative_return += public_reward
        if external_force.maximum_norm is not None and (
            self._maximum_external_force_body_norm is None
            or external_force.maximum_norm > self._maximum_external_force_body_norm
        ):
            self._maximum_external_force_body_norm = external_force.maximum_norm
            self._maximum_external_force_body = external_force.maximum_name
        if actuator_force.maximum_norm is not None and (
            self._maximum_absolute_actuator_force is None
            or actuator_force.maximum_norm > self._maximum_absolute_actuator_force
        ):
            self._maximum_absolute_actuator_force = actuator_force.maximum_norm
            self._maximum_actuator_force_joint = actuator_force.maximum_name
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
            external_force=external_force,
            actuator_force=actuator_force,
            reward=public_reward,
            terminated=terminated,
            truncated=truncated,
            step_count=self._steps,
            config=self._config,
            initial_x_position=self._initial_x_position,
            initial_y_position=self._initial_y_position,
            initial_torso_z_position=self._initial_torso_z_position,
            minimum_torso_z_position=self._minimum_torso_z_position,
            maximum_torso_z_position=self._maximum_torso_z_position,
            minimum_torso_z_velocity=self._minimum_torso_z_velocity,
            maximum_torso_z_velocity=self._maximum_torso_z_velocity,
            minimum_torso_tilt_radians=self._minimum_torso_tilt_radians,
            maximum_torso_tilt_radians=self._maximum_torso_tilt_radians,
            upward_steps=self._upward_steps,
            cumulative_absolute_action=self._cumulative_absolute_action,
            cumulative_upward_reward=self._cumulative_upward_reward,
            cumulative_control_reward=self._cumulative_control_reward,
            cumulative_impact_reward=self._cumulative_impact_reward,
            cumulative_constant_reward=self._cumulative_constant_reward,
            cumulative_return=self._cumulative_return,
            maximum_external_force_body_norm=(self._maximum_external_force_body_norm),
            maximum_external_force_body=self._maximum_external_force_body,
            maximum_absolute_actuator_force=(self._maximum_absolute_actuator_force),
            maximum_actuator_force_joint=self._maximum_actuator_force_joint,
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
    config: HumanoidStandupConfig,
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
        raise RuntimeError("HumanoidStandup returned an invalid observation")
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
        raise RuntimeError(f"HumanoidStandup returned invalid {name} values")
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
        raise RuntimeError("HumanoidStandup omitted reset position metrics")
    return {
        "x_position": _number(value["x_position"], name="reset x position"),
        "y_position": _number(value["y_position"], name="reset y position"),
    }


def _provider_metrics(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError("HumanoidStandup returned invalid metrics")
    scalar_names = (
        "x_position",
        "y_position",
        "z_distance_from_origin",
        "reward_linup",
        "reward_quadctrl",
        "reward_impact",
    )
    if not set((*scalar_names, "tendon_length", "tendon_velocity")).issubset(value):
        raise RuntimeError("HumanoidStandup omitted public metrics")
    renamed = {
        "reward_linup": "reward_upward",
        "reward_quadctrl": "reward_control",
    }
    metrics: dict[str, object] = {
        renamed.get(name, name): _number(
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
        raise RuntimeError(f"HumanoidStandup returned invalid {name} values")
    return {
        "left_hip_to_knee": _number(value[0], name=f"left {name}"),
        "right_hip_to_knee": _number(value[1], name=f"right {name}"),
    }


def _scalar_provider_metrics(value: dict[str, object]) -> dict[str, float]:
    names = (
        "x_position",
        "y_position",
        "z_distance_from_origin",
        "reward_upward",
        "reward_control",
        "reward_impact",
    )
    scalars: dict[str, float] = {}
    for name in names:
        item = value.get(name)
        if type(item) is not float:
            raise RuntimeError(f"HumanoidStandup returned invalid {name}")
        scalars[name] = item
    return scalars


def _tendon_metric(value: dict[str, object], name: str) -> dict[str, float]:
    item = value.get(name)
    if type(item) is not dict or set(item) != set(_TENDONS):
        raise RuntimeError(f"HumanoidStandup returned invalid {name}")
    result: dict[str, float] = {}
    for tendon in _TENDONS:
        number = item[tendon]
        if type(number) is not float:
            raise RuntimeError(f"HumanoidStandup returned invalid {name}")
        result[tendon] = number
    return result


def _transition_metrics(
    observation: dict[str, PolicyValue],
    action: NDArray[numpy.float32],
    *,
    provider_metrics: dict[str, object],
    external_force: _ForceSummary,
    actuator_force: _ForceSummary,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_count: int,
    config: HumanoidStandupConfig,
    initial_x_position: float,
    initial_y_position: float,
    initial_torso_z_position: float,
    minimum_torso_z_position: float,
    maximum_torso_z_position: float,
    minimum_torso_z_velocity: float,
    maximum_torso_z_velocity: float,
    minimum_torso_tilt_radians: float,
    maximum_torso_tilt_radians: float,
    upward_steps: int,
    cumulative_absolute_action: float,
    cumulative_upward_reward: float,
    cumulative_control_reward: float,
    cumulative_impact_reward: float,
    cumulative_constant_reward: float,
    cumulative_return: float,
    maximum_external_force_body_norm: float | None,
    maximum_external_force_body: str,
    maximum_absolute_actuator_force: float | None,
    maximum_actuator_force_joint: str,
    maximum_absolute_tendon_velocity: float,
) -> dict[str, PolicyValue]:
    scalars = _scalar_provider_metrics(provider_metrics)
    reward_upward = scalars["reward_upward"]
    reward_control = scalars["reward_control"]
    reward_impact = scalars["reward_impact"]
    reconstructed_reward = reward_upward + reward_control + reward_impact + 1.0
    if not math.isclose(
        reward,
        reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("HumanoidStandup reward decomposition drifted")
    expected_control_reward = -config.ctrl_cost_weight * float(numpy.sum(numpy.square(action)))
    if not math.isclose(
        reward_control,
        expected_control_reward,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("HumanoidStandup control-cost semantics drifted")
    torso_z = _float_field(observation, "torso_z_position")
    if not math.isclose(
        reward_upward,
        torso_z / _MODEL_TIMESTEP_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("HumanoidStandup upward-reward semantics drifted")
    if not math.isclose(
        scalars["z_distance_from_origin"],
        torso_z - _NOMINAL_PRONE_TORSO_Z,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("HumanoidStandup z-distance semantics drifted")
    raw_impact_cost: float | None = None
    if external_force.total_squared is not None:
        raw_impact_cost = config.impact_cost_weight * external_force.total_squared
        lower, upper = config.impact_cost_range
        clamped_impact_cost = min(
            max(raw_impact_cost, -math.inf if lower is None else lower),
            upper,
        )
        if not math.isclose(
            reward_impact,
            -clamped_impact_cost,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("HumanoidStandup impact-cost semantics drifted")
    if terminated:
        raise RuntimeError("HumanoidStandup unexpectedly terminated")
    if truncated != (step_count == _MAX_EPISODE_STEPS):
        raise RuntimeError("HumanoidStandup time-limit semantics drifted")
    cumulative_reconstructed_reward = (
        cumulative_upward_reward
        + cumulative_control_reward
        + cumulative_impact_reward
        + cumulative_constant_reward
    )
    if not math.isclose(
        cumulative_return,
        cumulative_reconstructed_reward,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("HumanoidStandup cumulative reward decomposition drifted")
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
    torso_z_velocity = _float_field(observation, "torso_z_velocity")
    torso_tilt = _torso_tilt(observation)
    seconds_per_step = config.frame_skip * _MODEL_TIMESTEP_SECONDS
    x_position = scalars["x_position"]
    y_position = scalars["y_position"]
    return {
        "step_count": step_count,
        "remaining_steps": max(_MAX_EPISODE_STEPS - step_count, 0),
        "model_timestep_seconds": _MODEL_TIMESTEP_SECONDS,
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
        "net_x_displacement": x_position - initial_x_position,
        "net_y_displacement": y_position - initial_y_position,
        "initial_torso_z_position": initial_torso_z_position,
        "torso_z_position": torso_z,
        "height_gain_from_reset": torso_z - initial_torso_z_position,
        "z_distance_from_nominal_origin": scalars["z_distance_from_origin"],
        "minimum_torso_z_position": minimum_torso_z_position,
        "maximum_torso_z_position": maximum_torso_z_position,
        "maximum_height_gain_from_reset": (maximum_torso_z_position - initial_torso_z_position),
        "torso_z_velocity": torso_z_velocity,
        "minimum_torso_z_velocity": minimum_torso_z_velocity,
        "maximum_torso_z_velocity": maximum_torso_z_velocity,
        "upward_step_fraction": upward_steps / step_count,
        "torso_tilt_radians": torso_tilt,
        "torso_tilt_degrees": math.degrees(torso_tilt),
        "minimum_torso_tilt_radians": minimum_torso_tilt_radians,
        "maximum_torso_tilt_radians": maximum_torso_tilt_radians,
        "quaternion_norm_error": _quaternion_norm_error(observation),
        "external_forces_in_observation": (config.include_cfrc_ext_in_observation),
        "sum_squared_external_force_components": external_force.total_squared,
        "raw_impact_cost_before_clamp": raw_impact_cost,
        "maximum_external_force_body_norm_this_step": (external_force.maximum_norm),
        "maximum_external_force_body_this_step": external_force.maximum_name,
        "maximum_external_force_body_norm": maximum_external_force_body_norm,
        "maximum_external_force_body": maximum_external_force_body,
        "actuator_forces_in_observation": (config.include_qfrc_actuator_in_observation),
        "maximum_absolute_actuator_force_this_step": (actuator_force.maximum_norm),
        "maximum_actuator_force_joint_this_step": (actuator_force.maximum_name),
        "maximum_absolute_actuator_force": maximum_absolute_actuator_force,
        "maximum_actuator_force_joint": maximum_actuator_force_joint,
        "tendon_lengths": public_tendon_lengths,
        "tendon_velocities": public_tendon_velocities,
        "maximum_absolute_tendon_velocity": maximum_absolute_tendon_velocity,
        "reward_upward": reward_upward,
        "reward_control": reward_control,
        "reward_impact": reward_impact,
        "reward_constant": 1.0,
        "reward_from_public_terms": reconstructed_reward,
        "cumulative_reward_upward": cumulative_upward_reward,
        "cumulative_reward_control": cumulative_control_reward,
        "cumulative_reward_impact": cumulative_impact_reward,
        "cumulative_reward_constant": cumulative_constant_reward,
        "cumulative_return": cumulative_return,
        "terminal_reason": "time_limit" if truncated else "none",
    }


def _torso_tilt(observation: dict[str, PolicyValue]) -> float:
    w = _float_field(observation, "torso_orientation_w")
    x = _float_field(observation, "torso_orientation_x")
    y = _float_field(observation, "torso_orientation_y")
    z = _float_field(observation, "torso_orientation_z")
    norm_squared = w**2 + x**2 + y**2 + z**2
    if norm_squared <= 0.0:
        raise RuntimeError("HumanoidStandup returned an invalid torso quaternion")
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


def _external_force_summary(
    observation: dict[str, PolicyValue],
) -> _ForceSummary:
    raw_forces = observation.get("external_forces")
    if raw_forces is None:
        return _ForceSummary(None, None, "unavailable")
    if type(raw_forces) is not dict or set(raw_forces) != set(_BODIES):
        raise RuntimeError("HumanoidStandup returned invalid external forces")
    total_squared = 0.0
    maximum_norm = 0.0
    maximum_body = _BODIES[0]
    for body in _BODIES:
        raw_components = raw_forces[body]
        if type(raw_components) is not dict or set(raw_components) != set(
            _EXTERNAL_FORCE_COMPONENTS
        ):
            raise RuntimeError("HumanoidStandup returned invalid external forces")
        squared = 0.0
        for component in _EXTERNAL_FORCE_COMPONENTS:
            value = raw_components[component]
            if type(value) is not float:
                raise RuntimeError("HumanoidStandup returned invalid external forces")
            squared += value**2
        total_squared += squared
        norm = math.sqrt(squared)
        if norm > maximum_norm:
            maximum_norm = norm
            maximum_body = body
    return _ForceSummary(total_squared, maximum_norm, maximum_body)


def _actuator_force_summary(
    observation: dict[str, PolicyValue],
) -> _ForceSummary:
    raw_forces = observation.get("actuator_forces")
    if raw_forces is None:
        return _ForceSummary(None, None, "unavailable")
    if type(raw_forces) is not dict or set(raw_forces) != set(_JOINTS):
        raise RuntimeError("HumanoidStandup returned invalid actuator forces")
    maximum_force = 0.0
    maximum_joint = _JOINTS[0]
    for joint in _JOINTS:
        value = raw_forces[joint]
        if type(value) is not float:
            raise RuntimeError("HumanoidStandup returned invalid actuator forces")
        absolute = abs(value)
        if absolute > maximum_force:
            maximum_force = absolute
            maximum_joint = joint
    return _ForceSummary(None, maximum_force, maximum_joint)


def _float_field(observation: dict[str, PolicyValue], name: str) -> float:
    value = observation.get(name)
    if type(value) is not float:
        raise RuntimeError(f"HumanoidStandup returned invalid {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"HumanoidStandup returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"HumanoidStandup returned a non-finite {name}")
    return number


__all__ = ["HumanoidStandupEnvironment"]
