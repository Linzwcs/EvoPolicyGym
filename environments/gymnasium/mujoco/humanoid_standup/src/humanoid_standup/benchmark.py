"""A parameterized HumanoidStandup-v5 Benchmark with nested traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import HumanoidStandupConfig
from .environment import HumanoidStandupEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-humanoid-standup/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_TRANSITIONS_PER_EPISODE = 100
_MAX_EPISODE_STEPS = 1_000
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
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "model_timestep_seconds",
        "seconds_per_step",
        "simulated_seconds",
        "requested_action_by_joint",
        "actuator_gear_scaled_controls",
        "sum_squared_action",
        "sum_absolute_action",
        "cumulative_absolute_action",
        "initial_x_position",
        "initial_y_position",
        "x_position",
        "y_position",
        "net_x_displacement",
        "net_y_displacement",
        "initial_torso_z_position",
        "torso_z_position",
        "height_gain_from_reset",
        "z_distance_from_nominal_origin",
        "minimum_torso_z_position",
        "maximum_torso_z_position",
        "maximum_height_gain_from_reset",
        "torso_z_velocity",
        "minimum_torso_z_velocity",
        "maximum_torso_z_velocity",
        "upward_step_fraction",
        "torso_tilt_radians",
        "torso_tilt_degrees",
        "minimum_torso_tilt_radians",
        "maximum_torso_tilt_radians",
        "quaternion_norm_error",
        "external_forces_in_observation",
        "sum_squared_external_force_components",
        "raw_impact_cost_before_clamp",
        "maximum_external_force_body_norm_this_step",
        "maximum_external_force_body_this_step",
        "maximum_external_force_body_norm",
        "maximum_external_force_body",
        "actuator_forces_in_observation",
        "maximum_absolute_actuator_force_this_step",
        "maximum_actuator_force_joint_this_step",
        "maximum_absolute_actuator_force",
        "maximum_actuator_force_joint",
        "tendon_lengths",
        "tendon_velocities",
        "maximum_absolute_tendon_velocity",
        "reward_upward",
        "reward_control",
        "reward_impact",
        "reward_constant",
        "reward_from_public_terms",
        "cumulative_reward_upward",
        "cumulative_reward_control",
        "cumulative_reward_impact",
        "cumulative_reward_constant",
        "cumulative_return",
        "terminal_reason",
    }
)
_BOOL_METRICS = frozenset({"external_forces_in_observation", "actuator_forces_in_observation"})
_STRING_METRICS = frozenset(
    {
        "maximum_external_force_body_this_step",
        "maximum_external_force_body",
        "maximum_actuator_force_joint_this_step",
        "maximum_actuator_force_joint",
        "terminal_reason",
    }
)
_OPTIONAL_FLOAT_METRICS = frozenset(
    {
        "sum_squared_external_force_components",
        "raw_impact_cost_before_clamp",
        "maximum_external_force_body_norm_this_step",
        "maximum_external_force_body_norm",
        "maximum_absolute_actuator_force_this_step",
        "maximum_absolute_actuator_force",
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    initial_torso_z_position: float
    final_torso_z_position: float
    final_height_gain_from_reset: float
    minimum_torso_z_position: float
    maximum_torso_z_position: float
    maximum_height_gain_from_reset: float
    minimum_torso_z_velocity: float
    maximum_torso_z_velocity: float
    upward_step_fraction: float
    final_torso_tilt_radians: float
    minimum_torso_tilt_radians: float
    maximum_torso_tilt_radians: float
    net_x_displacement: float
    net_y_displacement: float
    mean_absolute_action: float
    maximum_external_force_body_norm: float | None
    maximum_absolute_actuator_force: float | None
    maximum_absolute_tendon_velocity: float
    cumulative_reward_upward: float
    cumulative_reward_control: float
    cumulative_reward_impact: float
    cumulative_reward_constant: float
    outcome: str


class HumanoidStandupBenchmark:
    """Mean HumanoidStandup return over deterministic Episode plans."""

    def __init__(
        self,
        config: HumanoidStandupConfig | None = None,
    ) -> None:
        if config is None:
            config = HumanoidStandupConfig()
        if type(config) is not HumanoidStandupConfig:
            raise TypeError("config must be HumanoidStandupConfig")
        self._config = config
        self._failure_return = _failure_return(config)
        self._spec = _benchmark_spec(
            config,
            failure_return=self._failure_return,
        )

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        if type(split) is not str or split not in _SPLITS:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(
            EpisodeSpec(environment_seed=_episode_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return HumanoidStandupEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        returns = tuple(
            record.total_reward if record.policy_failure is None else self._failure_return
            for record in records
        )
        score = statistics.fmean(returns)
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        outcomes = tuple(_episode_outcome(record) for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        mean_final_height = _mean_or_none(
            tuple(item.final_torso_z_position for item in diagnostics)
        )
        mean_max_height = _mean_or_none(
            tuple(item.maximum_torso_z_position for item in diagnostics)
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"mean final torso height "
                    f"{_height_summary(mean_final_height)}; mean maximum "
                    f"height {_height_summary(mean_max_height)}."
                ),
                "mean_return": score,
                "mean_steps": statistics.fmean(record.steps for record in records),
                "mean_initial_torso_height": _mean_or_none(
                    tuple(item.initial_torso_z_position for item in diagnostics)
                ),
                "mean_final_torso_height": mean_final_height,
                "mean_final_height_gain_from_reset": _mean_or_none(
                    tuple(item.final_height_gain_from_reset for item in diagnostics)
                ),
                "mean_episode_minimum_torso_height": _mean_or_none(
                    tuple(item.minimum_torso_z_position for item in diagnostics)
                ),
                "mean_episode_maximum_torso_height": mean_max_height,
                "mean_episode_maximum_height_gain_from_reset": _mean_or_none(
                    tuple(item.maximum_height_gain_from_reset for item in diagnostics)
                ),
                "mean_episode_minimum_torso_z_velocity": _mean_or_none(
                    tuple(item.minimum_torso_z_velocity for item in diagnostics)
                ),
                "mean_episode_maximum_torso_z_velocity": _mean_or_none(
                    tuple(item.maximum_torso_z_velocity for item in diagnostics)
                ),
                "mean_upward_step_fraction": _mean_or_none(
                    tuple(item.upward_step_fraction for item in diagnostics)
                ),
                "mean_final_torso_tilt_radians": _mean_or_none(
                    tuple(item.final_torso_tilt_radians for item in diagnostics)
                ),
                "mean_episode_minimum_torso_tilt_radians": _mean_or_none(
                    tuple(item.minimum_torso_tilt_radians for item in diagnostics)
                ),
                "mean_episode_maximum_torso_tilt_radians": _mean_or_none(
                    tuple(item.maximum_torso_tilt_radians for item in diagnostics)
                ),
                "mean_absolute_net_horizontal_displacement": _mean_or_none(
                    tuple(
                        math.hypot(
                            item.net_x_displacement,
                            item.net_y_displacement,
                        )
                        for item in diagnostics
                    )
                ),
                "mean_absolute_action": _mean_or_none(
                    tuple(item.mean_absolute_action for item in diagnostics)
                ),
                "mean_episode_maximum_external_force_body_norm": (
                    _mean_optional(
                        tuple(item.maximum_external_force_body_norm for item in diagnostics)
                    )
                ),
                "mean_episode_maximum_absolute_actuator_force": (
                    _mean_optional(
                        tuple(item.maximum_absolute_actuator_force for item in diagnostics)
                    )
                ),
                "mean_episode_maximum_absolute_tendon_velocity": _mean_or_none(
                    tuple(item.maximum_absolute_tendon_velocity for item in diagnostics)
                ),
                "mean_episode_upward_reward": _mean_or_none(
                    tuple(item.cumulative_reward_upward for item in diagnostics)
                ),
                "mean_episode_control_reward": _mean_or_none(
                    tuple(item.cumulative_reward_control for item in diagnostics)
                ),
                "mean_episode_impact_reward": _mean_or_none(
                    tuple(item.cumulative_reward_impact for item in diagnostics)
                ),
                "mean_episode_constant_reward": _mean_or_none(
                    tuple(item.cumulative_reward_constant for item in diagnostics)
                ),
                "time_limit_episodes": outcomes.count("time_limit"),
                "incomplete_episodes": outcomes.count("incomplete"),
                "episodes": len(records),
                "policy_failures": sum(record.policy_failure is not None for record in records),
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "traced_transitions": sum(
                    len(_trace_transition_indices(record)) for record in traced
                ),
                "trace_transitions_omitted": sum(
                    len(record.transitions) - len(_trace_transition_indices(record))
                    for record in traced
                ),
                "trace_transition_limit_per_episode": (_MAX_TRACED_TRANSITIONS_PER_EPISODE),
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    config=self._config,
                    failure_return=self._failure_return,
                ),
            ),
        )


def _benchmark_spec(
    config: HumanoidStandupConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["torso_x_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Root qpos x position.",
        }
        fields["torso_y_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Root qpos y position.",
        }
    for name in _STATE_FIELDS:
        fields[name] = _state_space(name)
    if config.include_cinert_in_observation:
        fields["body_inertias"] = _body_space(
            _INERTIA_COMPONENTS,
            unit=_inertia_unit,
        )
    if config.include_cvel_in_observation:
        fields["body_velocities"] = _body_space(
            _BODY_VELOCITY_COMPONENTS,
            unit=_body_velocity_unit,
        )
    if config.include_qfrc_actuator_in_observation:
        fields["actuator_forces"] = {
            "type": "object",
            "meaning": (
                "Generalized actuator forces in state-joint order; abdomen_z "
                "precedes abdomen_y, unlike Action order."
            ),
            "fields": {joint: {"type": "float", "unit": "generalized_force"} for joint in _JOINTS},
        }
    if config.include_cfrc_ext_in_observation:
        fields["external_forces"] = _body_space(
            _EXTERNAL_FORCE_COMPONENTS,
            unit=_external_force_unit,
        )
    return BenchmarkSpec(
        id="gymnasium/HumanoidStandup-v5/mean-return-v1",
        description=(
            "Coordinate 17 bounded controls to raise a prone 3D Humanoid and "
            "keep its torso high. Every step rewards absolute torso root z "
            "divided by the 0.003-second model timestep, plus one, minus "
            "squared-control and clamped impact costs. The environment has no "
            "natural termination and truncates at 1000 steps."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, PolicyValue]",
            "source_dtype": "float64",
            "fields": fields,
        },
        action_space={
            "type": "array",
            "shape": [17],
            "items": {
                "type": "float",
                "minimum": -0.4,
                "maximum": 0.4,
            },
            "policy_carrier": "list[float]",
            "components": list(_ACTION_COMPONENTS),
            "actuator_gears": list(_ACTUATOR_GEARS),
            "meaning": ("Controls are in actuator order; abdomen_y precedes abdomen_z."),
        },
        metadata={
            "environment": "HumanoidStandup-v5",
            "provider": "Gymnasium",
            "official_model": "humanoidstandup.xml",
            "failure_return": failure_return,
            "inactive_upstream_parameter": "uph_cost_weight",
        },
        environment_parameters=_environment_parameters(config),
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _environment_parameters(
    config: HumanoidStandupConfig,
) -> dict[str, PolicyValue]:
    return {
        "frame_skip": config.frame_skip,
        "model_timestep_seconds": 0.003,
        "seconds_per_step": 0.003 * config.frame_skip,
        "upward_reward_time_divisor_seconds": 0.003,
        "nominal_prone_torso_z": 0.105,
        "action_components": list(_ACTION_COMPONENTS),
        "actuator_gears": list(_ACTUATOR_GEARS),
        "ctrl_cost_weight": config.ctrl_cost_weight,
        "impact_cost_weight": config.impact_cost_weight,
        "impact_cost_range": list(config.impact_cost_range),
        "reward_formula": (
            "torso_z/0.003+1-ctrl_cost_weight*sum(action^2)-clip("
            "impact_cost_weight*sum(cfrc_ext^2),impact_cost_range)"
        ),
        "natural_termination": "none",
        "time_limit": _MAX_EPISODE_STEPS,
        "reset_noise_scale": config.reset_noise_scale,
        "exclude_current_positions_from_observation": (
            config.exclude_current_positions_from_observation
        ),
        "include_cinert_in_observation": (config.include_cinert_in_observation),
        "include_cvel_in_observation": config.include_cvel_in_observation,
        "include_qfrc_actuator_in_observation": (config.include_qfrc_actuator_in_observation),
        "include_cfrc_ext_in_observation": (config.include_cfrc_ext_in_observation),
    }


def _state_space(name: str) -> dict[str, PolicyValue]:
    meaning = f"{name.replace('_', ' ').capitalize()} from MuJoCo state."
    if name == "torso_z_position":
        meaning = (
            "Root qpos z height; absolute value divided by 0.003 seconds produces reward_upward."
        )
    elif name.startswith("torso_orientation_"):
        meaning = "Component of the root qpos w,x,y,z quaternion."
    return {"type": "float", "unit": _state_unit(name), "meaning": meaning}


def _body_space(
    components: tuple[str, ...],
    *,
    unit: Callable[[str], str],
) -> PolicyValue:
    bodies: dict[str, PolicyValue] = {}
    for body in _BODIES:
        component_fields: dict[str, PolicyValue] = {}
        for component in components:
            component_fields[component] = {
                "type": "float",
                "unit": unit(component),
            }
        bodies[body] = {"type": "object", "fields": component_fields}
    return {"type": "object", "fields": bodies}


def _state_unit(name: str) -> str:
    if name == "torso_z_position":
        return "meters"
    if name.startswith("torso_orientation_"):
        return "quaternion_component"
    if name in {
        "torso_x_velocity",
        "torso_y_velocity",
        "torso_z_velocity",
    }:
        return "meters_per_second"
    if name.endswith("_angular_velocity"):
        return "radians_per_second"
    return "radians"


def _inertia_unit(component: str) -> str:
    if component == "mass":
        return "kilograms"
    if component.startswith("mass_times_"):
        return "kilogram_meters"
    return "kilogram_meter_squared"


def _body_velocity_unit(component: str) -> str:
    return "radians_per_second" if component.startswith("angular_") else "meters_per_second"


def _external_force_unit(component: str) -> str:
    return "newton_meters" if component.startswith("torque_") else "newtons"


def _failure_return(config: HumanoidStandupConfig) -> float:
    lower = config.impact_cost_range[0]
    impact_bound = max(
        abs(lower) if lower is not None else 0.0,
        abs(config.impact_cost_range[1]),
    )
    return -1_000.0 * max(
        1.0,
        17.0 * 0.4**2 * config.ctrl_cost_weight,
        impact_bound,
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if not record.transitions:
        return "incomplete"
    metrics = _trace_metrics(record.transitions[-1].step.metrics)
    reason = metrics["terminal_reason"]
    if type(reason) is not str:
        raise ValueError("HumanoidStandup terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("HumanoidStandup diagnostics require a transition")
    metrics = tuple(_trace_metrics(transition.step.metrics) for transition in record.transitions)
    final = metrics[-1]
    return _EpisodeDiagnostics(
        initial_torso_z_position=_float_metric(
            final,
            "initial_torso_z_position",
        ),
        final_torso_z_position=_float_metric(final, "torso_z_position"),
        final_height_gain_from_reset=_float_metric(
            final,
            "height_gain_from_reset",
        ),
        minimum_torso_z_position=_float_metric(
            final,
            "minimum_torso_z_position",
        ),
        maximum_torso_z_position=_float_metric(
            final,
            "maximum_torso_z_position",
        ),
        maximum_height_gain_from_reset=_float_metric(
            final,
            "maximum_height_gain_from_reset",
        ),
        minimum_torso_z_velocity=_float_metric(
            final,
            "minimum_torso_z_velocity",
        ),
        maximum_torso_z_velocity=_float_metric(
            final,
            "maximum_torso_z_velocity",
        ),
        upward_step_fraction=_float_metric(final, "upward_step_fraction"),
        final_torso_tilt_radians=_float_metric(final, "torso_tilt_radians"),
        minimum_torso_tilt_radians=_float_metric(
            final,
            "minimum_torso_tilt_radians",
        ),
        maximum_torso_tilt_radians=_float_metric(
            final,
            "maximum_torso_tilt_radians",
        ),
        net_x_displacement=_float_metric(final, "net_x_displacement"),
        net_y_displacement=_float_metric(final, "net_y_displacement"),
        mean_absolute_action=statistics.fmean(
            abs(value)
            for transition in record.transitions
            for value in _trace_action(transition.action)
        ),
        maximum_external_force_body_norm=_optional_float_metric(
            final,
            "maximum_external_force_body_norm",
        ),
        maximum_absolute_actuator_force=_optional_float_metric(
            final,
            "maximum_absolute_actuator_force",
        ),
        maximum_absolute_tendon_velocity=_float_metric(
            final,
            "maximum_absolute_tendon_velocity",
        ),
        cumulative_reward_upward=_float_metric(
            final,
            "cumulative_reward_upward",
        ),
        cumulative_reward_control=_float_metric(
            final,
            "cumulative_reward_control",
        ),
        cumulative_reward_impact=_float_metric(
            final,
            "cumulative_reward_impact",
        ),
        cumulative_reward_constant=_float_metric(
            final,
            "cumulative_reward_constant",
        ),
        outcome=_episode_outcome(record),
    )


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"HumanoidStandup metric {name} is invalid")
    return value


def _optional_float_metric(
    metrics: dict[str, object],
    name: str,
) -> float | None:
    value = metrics.get(name)
    if value is not None and type(value) is not float:
        raise ValueError(f"HumanoidStandup metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_optional(values: tuple[float | None, ...]) -> float | None:
    available = tuple(value for value in values if value is not None)
    return statistics.fmean(available) if available else None


def _height_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f} m"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    config: HumanoidStandupConfig,
    failure_return: float,
) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        traced_indices = _trace_transition_indices(record)
        diagnostics = (
            _episode_diagnostics(record)
            if record.policy_failure is None and record.transitions
            else None
        )
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "traced_transitions": len(traced_indices),
                    "omitted_transitions": (len(record.transitions) - len(traced_indices)),
                    "trace_sampling": (
                        "complete"
                        if len(traced_indices) == len(record.transitions)
                        else "uniform_including_endpoints"
                    ),
                    "return": (
                        record.total_reward if record.policy_failure is None else failure_return
                    ),
                    "outcome": _episode_outcome(record),
                    "initial_torso_height": (
                        diagnostics.initial_torso_z_position if diagnostics is not None else None
                    ),
                    "final_torso_height": (
                        diagnostics.final_torso_z_position if diagnostics is not None else None
                    ),
                    "maximum_torso_height": (
                        diagnostics.maximum_torso_z_position if diagnostics is not None else None
                    ),
                    "maximum_height_gain_from_reset": (
                        diagnostics.maximum_height_gain_from_reset
                        if diagnostics is not None
                        else None
                    ),
                    "minimum_torso_tilt_radians": (
                        diagnostics.minimum_torso_tilt_radians if diagnostics is not None else None
                    ),
                    "mean_absolute_action": (
                        diagnostics.mean_absolute_action if diagnostics is not None else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        for step_index in traced_indices:
            transition = record.transitions[step_index]
            observation = _trace_observation(
                (
                    record.initial_observation
                    if step_index == 0
                    else record.transitions[step_index - 1].step.observation
                ),
                config=config,
            )
            action = _trace_action(transition.action)
            next_observation = _trace_observation(
                transition.step.observation,
                config=config,
            )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "action_components": dict(zip(_ACTION_COMPONENTS, action, strict=True)),
                        "reward": transition.step.reward,
                        "metrics": _trace_metrics(transition.step.metrics),
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_transition_indices(record: EpisodeRecord) -> tuple[int, ...]:
    count = len(record.transitions)
    if count <= _MAX_TRACED_TRANSITIONS_PER_EPISODE:
        return tuple(range(count))
    last = count - 1
    divisor = _MAX_TRACED_TRANSITIONS_PER_EPISODE - 1
    return tuple(index * last // divisor for index in range(_MAX_TRACED_TRANSITIONS_PER_EPISODE))


def _trace_action(action: PolicyValue) -> list[float]:
    if type(action) is not list or len(action) != 17:
        raise ValueError("HumanoidStandup trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if type(value) is not float or not math.isfinite(value) or not -0.4 <= value <= 0.4:
            raise ValueError("HumanoidStandup trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    config: HumanoidStandupConfig,
) -> dict[str, object]:
    scalar_fields = (
        _STATE_FIELDS
        if config.exclude_current_positions_from_observation
        else ("torso_x_position", "torso_y_position", *_STATE_FIELDS)
    )
    expected = set(scalar_fields)
    optional = (
        ("body_inertias", config.include_cinert_in_observation),
        ("body_velocities", config.include_cvel_in_observation),
        ("actuator_forces", config.include_qfrc_actuator_in_observation),
        ("external_forces", config.include_cfrc_ext_in_observation),
    )
    expected.update(name for name, included in optional if included)
    if type(observation) is not dict or set(observation) != expected:
        raise ValueError("HumanoidStandup trace observation is invalid")
    traced: dict[str, object] = {}
    for key in scalar_fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("HumanoidStandup trace observation is invalid")
        traced[key] = value
    if config.include_cinert_in_observation:
        traced["body_inertias"] = _trace_nested(
            observation["body_inertias"],
            outer=_BODIES,
            inner=_INERTIA_COMPONENTS,
            name="body inertias",
        )
    if config.include_cvel_in_observation:
        traced["body_velocities"] = _trace_nested(
            observation["body_velocities"],
            outer=_BODIES,
            inner=_BODY_VELOCITY_COMPONENTS,
            name="body velocities",
        )
    if config.include_qfrc_actuator_in_observation:
        traced["actuator_forces"] = _trace_float_object(
            observation["actuator_forces"],
            fields=_JOINTS,
            name="actuator forces",
        )
    if config.include_cfrc_ext_in_observation:
        traced["external_forces"] = _trace_nested(
            observation["external_forces"],
            outer=_BODIES,
            inner=_EXTERNAL_FORCE_COMPONENTS,
            name="external forces",
        )
    return traced


def _trace_nested(
    value: PolicyValue,
    *,
    outer: tuple[str, ...],
    inner: tuple[str, ...],
    name: str,
) -> dict[str, dict[str, float]]:
    if type(value) is not dict or set(value) != set(outer):
        raise ValueError(f"HumanoidStandup trace {name} are invalid")
    return {key: _trace_float_object(value[key], fields=inner, name=name) for key in outer}


def _trace_float_object(
    value: PolicyValue,
    *,
    fields: tuple[str, ...],
    name: str,
) -> dict[str, float]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"HumanoidStandup trace {name} are invalid")
    traced: dict[str, float] = {}
    for key in fields:
        item = value[key]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"HumanoidStandup trace {name} are invalid")
        traced[key] = item
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("HumanoidStandup trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("HumanoidStandup trace metrics are invalid")
        elif key in _BOOL_METRICS:
            if type(value) is not bool:
                raise ValueError("HumanoidStandup trace metrics are invalid")
        elif key in _STRING_METRICS:
            if type(value) is not str:
                raise ValueError("HumanoidStandup trace metrics are invalid")
        elif key in _OPTIONAL_FLOAT_METRICS:
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise ValueError("HumanoidStandup trace metrics are invalid")
        elif key in {
            "requested_action_by_joint",
            "actuator_gear_scaled_controls",
        }:
            if type(value) is not dict or set(value) != set(_ACTION_COMPONENTS):
                raise ValueError("HumanoidStandup trace metrics are invalid")
            for item in value.values():
                if type(item) is not float or not math.isfinite(item):
                    raise ValueError("HumanoidStandup trace metrics are invalid")
        elif key in {"tendon_lengths", "tendon_velocities"}:
            traced[key] = _trace_float_object(
                value,
                fields=_TENDONS,
                name=key.replace("_", " "),
            )
            continue
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("HumanoidStandup trace metrics are invalid")
        traced[key] = value
    return traced


def _json_line(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


__all__ = ["HumanoidStandupBenchmark"]
