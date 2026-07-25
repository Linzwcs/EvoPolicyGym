"""A parameterized Humanoid-v5 Benchmark with public nested traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Sequence

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import HumanoidConfig
from .environment import HumanoidEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-humanoid/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
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
_SCALAR_METRIC_FIELDS = (
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


class HumanoidBenchmark:
    """Mean Humanoid return over deterministic Episode plans."""

    def __init__(self, config: HumanoidConfig | None = None) -> None:
        if config is None:
            config = HumanoidConfig()
        if type(config) is not HumanoidConfig:
            raise TypeError("config must be HumanoidConfig")
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
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return HumanoidEnvironment(episode, config=self._config)

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
        final_positions = tuple(
            _final_x_position(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        mean_final_x = (
            statistics.fmean(final_positions)
            if final_positions
            else None
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
                "episodes": len(records),
                "policy_failures": failures,
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
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
    config: HumanoidConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["torso_x_position"] = {
            "type": "float",
            "unit": "meters",
        }
        fields["torso_y_position"] = {
            "type": "float",
            "unit": "meters",
        }
    for name in _STATE_FIELDS:
        fields[name] = {"type": "float", "unit": _state_unit(name)}
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
            "fields": {
                joint: {"type": "float", "unit": "generalized_force"}
                for joint in _JOINTS
            },
        }
    if config.include_cfrc_ext_in_observation:
        fields["external_forces"] = _body_space(
            _EXTERNAL_FORCE_COMPONENTS,
            unit=_external_force_unit,
        )
    return BenchmarkSpec(
        id="gymnasium/Humanoid-v5/mean-return-v1",
        description=(
            "Coordinate 17 joint torques to keep a 3D Humanoid upright and "
            "move its center of mass in the positive x direction. Maximize "
            "mean Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space={
            "type": "array",
            "shape": [17],
            "items": {
                "type": "float",
                "minimum": -0.4,
                "maximum": 0.4,
            },
            "components": [
                "abdomen_y_torque",
                "abdomen_z_torque",
                "abdomen_x_torque",
                "right_hip_x_torque",
                "right_hip_z_torque",
                "right_hip_y_torque",
                "right_knee_torque",
                "left_hip_x_torque",
                "left_hip_z_torque",
                "left_hip_y_torque",
                "left_knee_torque",
                "right_shoulder_1_torque",
                "right_shoulder_2_torque",
                "right_elbow_torque",
                "left_shoulder_1_torque",
                "left_shoulder_2_torque",
                "left_elbow_torque",
            ],
        },
        metadata={
            "environment": "Humanoid-v5",
            "provider": "Gymnasium",
            "official_model": "humanoid.xml",
            "failure_return": failure_return,
        },
        environment_parameters=_environment_parameters(config),
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _environment_parameters(config: HumanoidConfig) -> dict[str, PolicyValue]:
    return {
        "frame_skip": config.frame_skip,
        "forward_reward_weight": config.forward_reward_weight,
        "ctrl_cost_weight": config.ctrl_cost_weight,
        "contact_cost_weight": config.contact_cost_weight,
        "contact_cost_range": list(config.contact_cost_range),
        "healthy_reward": config.healthy_reward,
        "terminate_when_unhealthy": config.terminate_when_unhealthy,
        "healthy_z_range": list(config.healthy_z_range),
        "reset_noise_scale": config.reset_noise_scale,
        "exclude_current_positions_from_observation": (
            config.exclude_current_positions_from_observation
        ),
        "include_cinert_in_observation": (
            config.include_cinert_in_observation
        ),
        "include_cvel_in_observation": config.include_cvel_in_observation,
        "include_qfrc_actuator_in_observation": (
            config.include_qfrc_actuator_in_observation
        ),
        "include_cfrc_ext_in_observation": (
            config.include_cfrc_ext_in_observation
        ),
    }


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
    return (
        "radians_per_second"
        if component.startswith("angular_")
        else "meters_per_second"
    )


def _external_force_unit(component: str) -> str:
    return (
        "newton_meters"
        if component.startswith("torque_")
        else "newtons"
    )


def _failure_return(config: HumanoidConfig) -> float:
    lower = config.contact_cost_range[0]
    contact_bound = max(
        abs(lower) if lower is not None else 0.0,
        abs(config.contact_cost_range[1]),
    )
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
        config.healthy_reward,
        17.0 * 0.4**2 * config.ctrl_cost_weight,
        contact_bound,
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
    metrics = _trace_metrics(record.transitions[-1].step.metrics)
    value = metrics["x_position"]
    assert type(value) is float
    return value


def _position_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    config: HumanoidConfig,
    failure_return: float,
) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
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
                        _final_x_position(record)
                        if record.policy_failure is None
                        and record.transitions
                        else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(
            record.initial_observation,
            config=config,
        )
        for step_index, transition in enumerate(record.transitions):
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
                        "action": _trace_action(transition.action),
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
    if type(action) is not list or len(action) != 17:
        raise ValueError("Humanoid trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -0.4 <= value <= 0.4
        ):
            raise ValueError("Humanoid trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    config: HumanoidConfig,
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
        raise ValueError("Humanoid trace observation is invalid")
    traced: dict[str, object] = {}
    for key in scalar_fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Humanoid trace observation is invalid")
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
        raise ValueError(f"Humanoid trace {name} are invalid")
    return {
        key: _trace_float_object(
            value[key],
            fields=inner,
            name=name,
        )
        for key in outer
    }


def _trace_float_object(
    value: PolicyValue,
    *,
    fields: tuple[str, ...],
    name: str,
) -> dict[str, float]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"Humanoid trace {name} are invalid")
    traced: dict[str, float] = {}
    for key in fields:
        item = value[key]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"Humanoid trace {name} are invalid")
        traced[key] = item
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    expected = {
        *_SCALAR_METRIC_FIELDS,
        "tendon_lengths",
        "tendon_velocities",
    }
    if type(metrics) is not dict or set(metrics) != expected:
        raise ValueError("Humanoid trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _SCALAR_METRIC_FIELDS:
        value = metrics[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Humanoid trace metrics are invalid")
        traced[key] = value
    traced["tendon_lengths"] = _trace_float_object(
        metrics["tendon_lengths"],
        fields=_TENDONS,
        name="tendon lengths",
    )
    traced["tendon_velocities"] = _trace_float_object(
        metrics["tendon_velocities"],
        fields=_TENDONS,
        name="tendon velocities",
    )
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


__all__ = ["HumanoidBenchmark"]
