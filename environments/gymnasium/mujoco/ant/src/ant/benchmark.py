"""A parameterized Ant-v5 Benchmark with public nested traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
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

from .config import AntConfig
from .environment import AntEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-ant/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 1
_MAX_EPISODE_STEPS = 1_000
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
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "seconds_per_step",
        "simulated_seconds",
        "requested_action_by_joint",
        "actuator_gear_scaled_controls",
        "sum_squared_action",
        "sum_absolute_action",
        "cumulative_absolute_action",
        "x_position",
        "y_position",
        "distance_from_origin",
        "x_velocity",
        "y_velocity",
        "speed_in_horizontal_plane",
        "minimum_x_position",
        "maximum_x_position",
        "torso_z_position",
        "healthy",
        "healthy_z_lower_bound",
        "healthy_z_upper_bound",
        "healthy_z_margin",
        "minimum_healthy_z_margin",
        "healthy_step_fraction",
        "torso_tilt_radians",
        "torso_tilt_degrees",
        "maximum_torso_tilt_radians",
        "quaternion_norm_error",
        "contact_forces_in_observation",
        "sum_squared_clipped_contact_force_components",
        "maximum_contact_body_norm",
        "maximum_contact_body",
        "reward_forward",
        "reward_control",
        "reward_contact",
        "reward_survive",
        "reward_from_public_terms",
        "cumulative_reward_forward",
        "cumulative_reward_control",
        "cumulative_reward_contact",
        "cumulative_reward_survive",
        "cumulative_return",
        "terminal_reason",
    }
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
_MAIN_BODY_NAMES = (
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


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    final_x_position: float
    final_distance_from_origin: float
    minimum_x_position: float
    maximum_x_position: float
    mean_x_velocity: float
    cumulative_reward_forward: float
    cumulative_reward_control: float
    cumulative_reward_contact: float
    cumulative_reward_survive: float
    healthy_step_fraction: float
    maximum_torso_tilt_radians: float
    minimum_healthy_z_margin: float
    mean_absolute_action: float
    maximum_contact_body_norm: float
    outcome: str
class AntBenchmark:
    """Mean Ant return over deterministic Episode plans."""

    def __init__(self, config: AntConfig | None = None) -> None:
        if config is None:
            config = AntConfig()
        if type(config) is not AntConfig:
            raise TypeError("config must be AntConfig")
        self._config = config
        self._scalar_observation_fields = _scalar_observation_fields(config)
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
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return AntEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(
            (
                record.total_reward
                if record.policy_failure is None
                else self._failure_return
            )
            for record in records
        )
        score = statistics.fmean(returns)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        outcomes = tuple(_episode_outcome(record) for record in records)
        mean_final_x = _mean_or_none(
            tuple(item.final_x_position for item in diagnostics)
        )
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"mean final x position "
                    f"{_position_summary(mean_final_x)}."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_final_x_position": mean_final_x,
                "mean_final_distance_from_origin": _mean_or_none(
                    tuple(item.final_distance_from_origin for item in diagnostics)
                ),
                "mean_episode_minimum_x_position": _mean_or_none(
                    tuple(item.minimum_x_position for item in diagnostics)
                ),
                "mean_episode_maximum_x_position": _mean_or_none(
                    tuple(item.maximum_x_position for item in diagnostics)
                ),
                "mean_x_velocity": _mean_or_none(
                    tuple(item.mean_x_velocity for item in diagnostics)
                ),
                "mean_episode_forward_reward": _mean_or_none(
                    tuple(item.cumulative_reward_forward for item in diagnostics)
                ),
                "mean_episode_control_reward": _mean_or_none(
                    tuple(item.cumulative_reward_control for item in diagnostics)
                ),
                "mean_episode_contact_reward": _mean_or_none(
                    tuple(item.cumulative_reward_contact for item in diagnostics)
                ),
                "mean_episode_survival_reward": _mean_or_none(
                    tuple(item.cumulative_reward_survive for item in diagnostics)
                ),
                "mean_healthy_step_fraction": _mean_or_none(
                    tuple(item.healthy_step_fraction for item in diagnostics)
                ),
                "mean_episode_maximum_torso_tilt_radians": _mean_or_none(
                    tuple(item.maximum_torso_tilt_radians for item in diagnostics)
                ),
                "mean_episode_minimum_healthy_z_margin": _mean_or_none(
                    tuple(item.minimum_healthy_z_margin for item in diagnostics)
                ),
                "mean_absolute_action": _mean_or_none(
                    tuple(item.mean_absolute_action for item in diagnostics)
                ),
                "mean_episode_maximum_contact_body_norm": _mean_or_none(
                    tuple(item.maximum_contact_body_norm for item in diagnostics)
                ),
                "unhealthy_termination_episodes": outcomes.count("unhealthy"),
                "time_limit_episodes": outcomes.count("time_limit"),
                "incomplete_episodes": outcomes.count("incomplete"),
                "episodes": len(records),
                "policy_failures": failures,
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    scalar_observation_fields=(
                        self._scalar_observation_fields
                    ),
                    include_contact=(
                        self._config.include_cfrc_ext_in_observation
                    ),
                    failure_return=self._failure_return,
                ),
            ),
        )


def _benchmark_spec(
    config: AntConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["torso_x_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Global torso x position from root qpos; positive x is forward.",
        }
        fields["torso_y_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Global torso y position from root qpos.",
        }
    for name in _BODY_FIELDS:
        fields[name] = _body_field_space(name)
    if config.include_cfrc_ext_in_observation:
        fields["contact_forces"] = _contact_space(config.contact_force_range)
    return BenchmarkSpec(
        id="gymnasium/Ant-v5/mean-return-v1",
        description=(
            "Coordinate eight bounded actuator controls to keep a quadruped Ant "
            "healthy and move the configured main body in positive x. Each "
            "control is multiplied by actuator gear 150. Reward is "
            "forward_reward_weight*x_velocity + healthy_reward_if_healthy "
            "- ctrl_cost_weight*sum(action^2) "
            "- contact_cost_weight*sum(clipped_contact_force^2). Health requires "
            "finite state and inclusive torso z within healthy_z_range; an "
            "unhealthy Ant terminates only when configured. The Episode otherwise "
            "truncates at 1000 steps. Maximize mean return."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, PolicyValue]",
            "source_dtype": "float64",
            "fields": fields,
        },
        action_space={
            "type": "array",
            "shape": [8],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "policy_carrier": "list[float]",
            "components": list(_ACTION_COMPONENTS),
            "meaning": (
                "Actuator order hip_4,ankle_4,hip_1,ankle_1,hip_2,ankle_2,"
                "hip_3,ankle_3, semantically back-right, front-left, "
                "front-right, back-left. Each control is gear-scaled by 150."
            ),
        },
        metadata={
            "environment": "Ant-v5",
            "provider": "Gymnasium",
            "reward_threshold": 6000.0,
            "official_model": "ant.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": 0.01,
            "seconds_per_step": 0.01 * config.frame_skip,
            "actuator_gear": 150.0,
            "action_components": list(_ACTION_COMPONENTS),
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "contact_cost_weight": config.contact_cost_weight,
            "healthy_reward": config.healthy_reward,
            "main_body": config.main_body,
            "main_body_name": _MAIN_BODY_NAMES[config.main_body - 1],
            "terminate_when_unhealthy": config.terminate_when_unhealthy,
            "healthy_z_range": list(config.healthy_z_range),
            "contact_force_range": list(config.contact_force_range),
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
            "include_cfrc_ext_in_observation": (
                config.include_cfrc_ext_in_observation
            ),
            "reward_formula": (
                "forward_weight*x_velocity+healthy_reward_if_healthy"
                "-control_weight*sum(action^2)"
                "-contact_weight*sum(clipped_contact_force^2)"
            ),
            "health_requires_finite_state": True,
            "healthy_z_bounds_are_inclusive": True,
            "contact_forces_are_clipped_before_observation_and_cost": True,
            "contact_force_component_count_excluding_world": 78,
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _body_field_space(name: str) -> dict[str, PolicyValue]:
    readable_name = name.replace("_", " ")
    if name == "torso_z_position":
        return {
            "type": "float",
            "unit": "meters",
            "meaning": "Global torso height used by the inclusive health bounds.",
        }
    if name.startswith("torso_orientation_"):
        component = name.removeprefix("torso_orientation_")
        return {
            "type": "float",
            "unit": "quaternion_component",
            "meaning": f"Torso unit-quaternion {component} component in w,x,y,z order.",
        }
    if name in {
        "torso_x_velocity",
        "torso_y_velocity",
        "torso_z_velocity",
    }:
        return {
            "type": "float",
            "unit": "meters_per_second",
            "meaning": f"{readable_name.capitalize()} from root qvel.",
        }
    if name.startswith("torso_") and name.endswith("_angular_velocity"):
        return {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": f"{readable_name.capitalize()} from root qvel.",
        }
    if name.endswith("_angular_velocity"):
        return {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": f"{readable_name.capitalize()} in qvel joint order.",
        }
    return {
        "type": "float",
        "unit": "radians",
        "meaning": f"{readable_name.capitalize()} in qpos joint order.",
    }


def _contact_space(contact_force_range: tuple[float, float]) -> PolicyValue:
    bodies: dict[str, PolicyValue] = {}
    for body in _CONTACT_BODIES:
        components: dict[str, PolicyValue] = {}
        for component in _CONTACT_COMPONENTS:
            components[component] = {
                "type": "float",
                "minimum": contact_force_range[0],
                "maximum": contact_force_range[1],
                "unit": (
                    "newtons"
                    if component.startswith("force_")
                    else "newton_meters"
                ),
            }
        bodies[body] = {"type": "object", "fields": components}
    return {
        "type": "object",
        "fields": bodies,
        "meaning": (
            "Clipped external torque and force components for 13 non-world "
            "bodies; the same clipped values determine contact cost."
        ),
    }


def _scalar_observation_fields(config: AntConfig) -> tuple[str, ...]:
    return (
        _BODY_FIELDS
        if config.exclude_current_positions_from_observation
        else ("torso_x_position", "torso_y_position", *_BODY_FIELDS)
    )


def _failure_return(config: AntConfig) -> float:
    maximum_contact = max(
        abs(config.contact_force_range[0]),
        abs(config.contact_force_range[1]),
    )
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
        config.healthy_reward,
        8.0 * config.ctrl_cost_weight,
        78.0 * config.contact_cost_weight * maximum_contact**2,
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _final_x_position(record: EpisodeRecord) -> float:
    return _episode_diagnostics(record).final_x_position


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if not record.transitions:
        return "incomplete"
    metrics = _trace_metrics(record.transitions[-1].step.metrics)
    reason = metrics["terminal_reason"]
    if type(reason) is not str:
        raise ValueError("Ant trace terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("Ant diagnostics require an executed transition")
    metrics = tuple(
        _trace_metrics(transition.step.metrics)
        for transition in record.transitions
    )
    final = metrics[-1]
    mean_absolute_action = statistics.fmean(
        abs(value)
        for transition in record.transitions
        for value in _trace_action(transition.action)
    )
    return _EpisodeDiagnostics(
        final_x_position=_float_metric(final, "x_position"),
        final_distance_from_origin=_float_metric(final, "distance_from_origin"),
        minimum_x_position=min(
            _float_metric(item, "x_position") for item in metrics
        ),
        maximum_x_position=max(
            _float_metric(item, "x_position") for item in metrics
        ),
        mean_x_velocity=statistics.fmean(
            _float_metric(item, "x_velocity") for item in metrics
        ),
        cumulative_reward_forward=_float_metric(
            final,
            "cumulative_reward_forward",
        ),
        cumulative_reward_control=_float_metric(
            final,
            "cumulative_reward_control",
        ),
        cumulative_reward_contact=_float_metric(
            final,
            "cumulative_reward_contact",
        ),
        cumulative_reward_survive=_float_metric(
            final,
            "cumulative_reward_survive",
        ),
        healthy_step_fraction=_float_metric(final, "healthy_step_fraction"),
        maximum_torso_tilt_radians=max(
            _float_metric(item, "torso_tilt_radians") for item in metrics
        ),
        minimum_healthy_z_margin=min(
            _float_metric(item, "healthy_z_margin") for item in metrics
        ),
        mean_absolute_action=mean_absolute_action,
        maximum_contact_body_norm=max(
            _float_metric(item, "maximum_contact_body_norm") for item in metrics
        ),
        outcome=_episode_outcome(record),
    )


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"Ant trace metric {name} is invalid")
    return value


def _position_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    scalar_observation_fields: tuple[str, ...],
    include_contact: bool,
    failure_return: float,
) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
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
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": (
                        record.total_reward
                        if record.policy_failure is None
                        else failure_return
                    ),
                    "final_x_position": (
                        diagnostics.final_x_position
                        if diagnostics is not None
                        else None
                    ),
                    "outcome": _episode_outcome(record),
                    "final_distance_from_origin": (
                        diagnostics.final_distance_from_origin
                        if diagnostics is not None
                        else None
                    ),
                    "minimum_x_position": (
                        diagnostics.minimum_x_position
                        if diagnostics is not None
                        else None
                    ),
                    "maximum_x_position": (
                        diagnostics.maximum_x_position
                        if diagnostics is not None
                        else None
                    ),
                    "mean_x_velocity": (
                        diagnostics.mean_x_velocity
                        if diagnostics is not None
                        else None
                    ),
                    "healthy_step_fraction": (
                        diagnostics.healthy_step_fraction
                        if diagnostics is not None
                        else None
                    ),
                    "maximum_torso_tilt_radians": (
                        diagnostics.maximum_torso_tilt_radians
                        if diagnostics is not None
                        else None
                    ),
                    "minimum_healthy_z_margin": (
                        diagnostics.minimum_healthy_z_margin
                        if diagnostics is not None
                        else None
                    ),
                    "mean_absolute_action": (
                        diagnostics.mean_absolute_action
                        if diagnostics is not None
                        else None
                    ),
                    "maximum_contact_body_norm": (
                        diagnostics.maximum_contact_body_norm
                        if diagnostics is not None
                        else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(
            record.initial_observation,
            scalar_fields=scalar_observation_fields,
            include_contact=include_contact,
        )
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
            next_observation = _trace_observation(
                transition.step.observation,
                scalar_fields=scalar_observation_fields,
                include_contact=include_contact,
            )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "action_components": dict(
                            zip(_ACTION_COMPONENTS, action, strict=True)
                        ),
                        "reward": transition.step.reward,
                        "metrics": _trace_metrics(
                            transition.step.metrics
                        ),
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
            observation = next_observation
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_action(action: PolicyValue) -> list[float]:
    if type(action) is not list or len(action) != 8:
        raise ValueError("Ant trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError("Ant trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    scalar_fields: tuple[str, ...],
    include_contact: bool,
) -> dict[str, object]:
    expected = set(scalar_fields)
    if include_contact:
        expected.add("contact_forces")
    if type(observation) is not dict or set(observation) != expected:
        raise ValueError("Ant trace observation is invalid")
    traced: dict[str, object] = {}
    for key in scalar_fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Ant trace observation is invalid")
        traced[key] = value
    if include_contact:
        traced["contact_forces"] = _trace_contact_forces(
            observation["contact_forces"]
        )
    return traced


def _trace_contact_forces(value: PolicyValue) -> dict[str, dict[str, float]]:
    if type(value) is not dict or set(value) != set(_CONTACT_BODIES):
        raise ValueError("Ant trace contact forces are invalid")
    traced: dict[str, dict[str, float]] = {}
    for body in _CONTACT_BODIES:
        components = value[body]
        if (
            type(components) is not dict
            or set(components) != set(_CONTACT_COMPONENTS)
        ):
            raise ValueError("Ant trace contact forces are invalid")
        traced_components: dict[str, float] = {}
        for component in _CONTACT_COMPONENTS:
            item = components[component]
            if type(item) is not float or not math.isfinite(item):
                raise ValueError("Ant trace contact forces are invalid")
            traced_components[component] = item
        traced[body] = traced_components
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Ant trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("Ant trace metrics are invalid")
        elif key in {"healthy", "contact_forces_in_observation"}:
            if type(value) is not bool:
                raise ValueError("Ant trace metrics are invalid")
        elif key in {"maximum_contact_body", "terminal_reason"}:
            if type(value) is not str:
                raise ValueError("Ant trace metrics are invalid")
        elif key in {"requested_action_by_joint", "actuator_gear_scaled_controls"}:
            if type(value) is not dict or set(value) != set(_ACTION_COMPONENTS):
                raise ValueError("Ant trace metrics are invalid")
            for item in value.values():
                if type(item) is not float or not math.isfinite(item):
                    raise ValueError("Ant trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("Ant trace metrics are invalid")
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


__all__ = ["AntBenchmark"]
