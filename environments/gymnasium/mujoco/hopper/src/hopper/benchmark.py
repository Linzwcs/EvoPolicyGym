"""A parameterized Hopper-v5 Benchmark with public traces."""

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

from .config import HopperConfig
from .environment import HopperEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-hopper/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_000
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
        "initial_x_position",
        "x_position",
        "net_x_displacement",
        "minimum_x_position",
        "maximum_x_position",
        "x_velocity",
        "minimum_x_velocity",
        "maximum_x_velocity",
        "mean_x_velocity_from_displacement",
        "forward_step_fraction",
        "torso_z_position",
        "minimum_torso_z_position",
        "torso_pitch_radians",
        "torso_pitch_degrees",
        "maximum_absolute_torso_pitch_radians",
        "healthy",
        "healthy_state",
        "healthy_z",
        "healthy_angle",
        "failed_health_conditions",
        "healthy_state_margin",
        "healthy_z_margin",
        "healthy_angle_margin",
        "minimum_healthy_state_margin",
        "minimum_healthy_z_margin",
        "minimum_healthy_angle_margin",
        "healthy_step_fraction",
        "z_distance_from_origin",
        "reward_forward",
        "reward_control",
        "reward_survive",
        "reward_from_public_terms",
        "cumulative_reward_forward",
        "cumulative_reward_control",
        "cumulative_reward_survive",
        "cumulative_return",
        "terminal_reason",
    }
)
_BOOL_METRICS = frozenset({"healthy", "healthy_state", "healthy_z", "healthy_angle"})


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    initial_x_position: float
    final_x_position: float
    net_x_displacement: float
    minimum_x_position: float
    maximum_x_position: float
    mean_x_velocity: float
    minimum_x_velocity: float
    maximum_x_velocity: float
    forward_step_fraction: float
    minimum_torso_z_position: float
    maximum_absolute_torso_pitch_radians: float
    healthy_step_fraction: float
    minimum_healthy_state_margin: float
    minimum_healthy_z_margin: float
    minimum_healthy_angle_margin: float
    mean_absolute_action: float
    cumulative_reward_forward: float
    cumulative_reward_control: float
    cumulative_reward_survive: float
    outcome: str


class HopperBenchmark:
    """Mean Hopper return over deterministic Episode plans."""

    def __init__(self, config: HopperConfig | None = None) -> None:
        if config is None:
            config = HopperConfig()
        if type(config) is not HopperConfig:
            raise TypeError("config must be HopperConfig")
        self._config = config
        self._observation_fields = _observation_fields(config)
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
        return HopperEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(
            (record.total_reward if record.policy_failure is None else self._failure_return)
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
        mean_final_x = _mean_or_none(tuple(item.final_x_position for item in diagnostics))
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"mean final x position "
                    f"{_position_summary(mean_final_x)}; "
                    f"{outcomes.count('unhealthy')} unhealthy terminations."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_initial_x_position": _mean_or_none(
                    tuple(item.initial_x_position for item in diagnostics)
                ),
                "mean_final_x_position": mean_final_x,
                "mean_net_x_displacement": _mean_or_none(
                    tuple(item.net_x_displacement for item in diagnostics)
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
                "mean_episode_minimum_x_velocity": _mean_or_none(
                    tuple(item.minimum_x_velocity for item in diagnostics)
                ),
                "mean_episode_maximum_x_velocity": _mean_or_none(
                    tuple(item.maximum_x_velocity for item in diagnostics)
                ),
                "mean_forward_step_fraction": _mean_or_none(
                    tuple(item.forward_step_fraction for item in diagnostics)
                ),
                "mean_episode_minimum_torso_z_position": _mean_or_none(
                    tuple(item.minimum_torso_z_position for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_torso_pitch_radians": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_torso_pitch_radians for item in diagnostics)
                    )
                ),
                "mean_healthy_step_fraction": _mean_or_none(
                    tuple(item.healthy_step_fraction for item in diagnostics)
                ),
                "mean_episode_minimum_healthy_state_margin": _mean_or_none(
                    tuple(item.minimum_healthy_state_margin for item in diagnostics)
                ),
                "mean_episode_minimum_healthy_z_margin": _mean_or_none(
                    tuple(item.minimum_healthy_z_margin for item in diagnostics)
                ),
                "mean_episode_minimum_healthy_angle_margin": _mean_or_none(
                    tuple(item.minimum_healthy_angle_margin for item in diagnostics)
                ),
                "mean_absolute_action": _mean_or_none(
                    tuple(item.mean_absolute_action for item in diagnostics)
                ),
                "mean_episode_forward_reward": _mean_or_none(
                    tuple(item.cumulative_reward_forward for item in diagnostics)
                ),
                "mean_episode_control_reward": _mean_or_none(
                    tuple(item.cumulative_reward_control for item in diagnostics)
                ),
                "mean_episode_survival_reward": _mean_or_none(
                    tuple(item.cumulative_reward_survive for item in diagnostics)
                ),
                "unhealthy_termination_episodes": outcomes.count("unhealthy"),
                "unhealthy_at_time_limit_episodes": outcomes.count("unhealthy_and_time_limit"),
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
                    observation_fields=self._observation_fields,
                    failure_return=self._failure_return,
                ),
            ),
        )


def _benchmark_spec(
    config: HopperConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["torso_x_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Global torso root x position; positive x is forward.",
        }
    for name in _BODY_FIELDS:
        fields[name] = _body_field_space(name)
    return BenchmarkSpec(
        id="gymnasium/Hopper-v5/mean-return-v1",
        description=(
            "Coordinate thigh, leg, and foot controls to keep a planar "
            "one-legged Hopper upright and propel it in positive x. Reward is "
            "forward_reward_weight*x_velocity + healthy_reward_if_healthy - "
            "ctrl_cost_weight*sum(action^2). Each control has actuator gear "
            "200. Health uses strict open bounds on the un-clipped MuJoCo "
            "state, torso height, and torso pitch. Maximize mean return."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float]",
            "source_dtype": "float64",
            "velocity_clipping": [-10.0, 10.0],
            "fields": fields,
        },
        action_space={
            "type": "array",
            "shape": [3],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "policy_carrier": "list[float]",
            "components": list(_ACTION_COMPONENTS),
            "actuator_gears": list(_ACTUATOR_GEARS),
            "meaning": "Controls are applied in thigh, leg, then foot order.",
        },
        metadata={
            "environment": "Hopper-v5",
            "provider": "Gymnasium",
            "reward_threshold": 3800.0,
            "official_model": "hopper.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": 0.002,
            "seconds_per_step": 0.002 * config.frame_skip,
            "action_components": list(_ACTION_COMPONENTS),
            "actuator_gears": list(_ACTUATOR_GEARS),
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "healthy_reward": config.healthy_reward,
            "terminate_when_unhealthy": config.terminate_when_unhealthy,
            "healthy_state_range": list(config.healthy_state_range),
            "healthy_z_range": list(config.healthy_z_range),
            "healthy_angle_range": list(config.healthy_angle_range),
            "health_bounds": "strict_open_intervals",
            "health_state_source": "unclipped_qpos[2:]+qvel",
            "observation_velocity_clipping": [-10.0, 10.0],
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
            "reward_formula": (
                "forward_reward_weight*x_velocity+healthy_reward_if_healthy-"
                "ctrl_cost_weight*sum(action^2)"
            ),
            "natural_termination": ("unhealthy when terminate_when_unhealthy is true"),
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
            "meaning": "Global torso root z position.",
        }
    if name in {"torso_x_velocity", "torso_z_velocity"}:
        return {
            "type": "float",
            "unit": "meters_per_second",
            "meaning": (f"{readable_name.capitalize()} from qvel, clipped to [-10, 10]."),
        }
    if name.endswith("_angular_velocity"):
        return {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": (f"{readable_name.capitalize()} from qvel, clipped to [-10, 10]."),
        }
    return {
        "type": "float",
        "unit": "radians",
        "meaning": f"{readable_name.capitalize()} from qpos.",
    }


def _observation_fields(config: HopperConfig) -> tuple[str, ...]:
    return (
        _BODY_FIELDS
        if config.exclude_current_positions_from_observation
        else ("torso_x_position", *_BODY_FIELDS)
    )


def _failure_return(config: HopperConfig) -> float:
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
        config.healthy_reward,
        3.0 * config.ctrl_cost_weight,
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
        raise ValueError("Hopper trace terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("Hopper diagnostics require a transition")
    metrics = tuple(_trace_metrics(transition.step.metrics) for transition in record.transitions)
    final = metrics[-1]
    return _EpisodeDiagnostics(
        initial_x_position=_float_metric(final, "initial_x_position"),
        final_x_position=_float_metric(final, "x_position"),
        net_x_displacement=_float_metric(final, "net_x_displacement"),
        minimum_x_position=_float_metric(final, "minimum_x_position"),
        maximum_x_position=_float_metric(final, "maximum_x_position"),
        mean_x_velocity=statistics.fmean(_float_metric(item, "x_velocity") for item in metrics),
        minimum_x_velocity=_float_metric(final, "minimum_x_velocity"),
        maximum_x_velocity=_float_metric(final, "maximum_x_velocity"),
        forward_step_fraction=_float_metric(final, "forward_step_fraction"),
        minimum_torso_z_position=_float_metric(
            final,
            "minimum_torso_z_position",
        ),
        maximum_absolute_torso_pitch_radians=_float_metric(
            final,
            "maximum_absolute_torso_pitch_radians",
        ),
        healthy_step_fraction=_float_metric(final, "healthy_step_fraction"),
        minimum_healthy_state_margin=_float_metric(
            final,
            "minimum_healthy_state_margin",
        ),
        minimum_healthy_z_margin=_float_metric(
            final,
            "minimum_healthy_z_margin",
        ),
        minimum_healthy_angle_margin=_float_metric(
            final,
            "minimum_healthy_angle_margin",
        ),
        mean_absolute_action=statistics.fmean(
            abs(value)
            for transition in record.transitions
            for value in _trace_action(transition.action)
        ),
        cumulative_reward_forward=_float_metric(
            final,
            "cumulative_reward_forward",
        ),
        cumulative_reward_control=_float_metric(
            final,
            "cumulative_reward_control",
        ),
        cumulative_reward_survive=_float_metric(
            final,
            "cumulative_reward_survive",
        ),
        outcome=_episode_outcome(record),
    )


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"Hopper trace metric {name} is invalid")
    return value


def _position_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    observation_fields: tuple[str, ...],
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
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "return": (
                        record.total_reward if record.policy_failure is None else failure_return
                    ),
                    "outcome": _episode_outcome(record),
                    "initial_x_position": (
                        diagnostics.initial_x_position if diagnostics is not None else None
                    ),
                    "final_x_position": (
                        diagnostics.final_x_position if diagnostics is not None else None
                    ),
                    "net_x_displacement": (
                        diagnostics.net_x_displacement if diagnostics is not None else None
                    ),
                    "mean_x_velocity": (
                        diagnostics.mean_x_velocity if diagnostics is not None else None
                    ),
                    "healthy_step_fraction": (
                        diagnostics.healthy_step_fraction if diagnostics is not None else None
                    ),
                    "maximum_absolute_torso_pitch_radians": (
                        diagnostics.maximum_absolute_torso_pitch_radians
                        if diagnostics is not None
                        else None
                    ),
                    "mean_absolute_action": (
                        diagnostics.mean_absolute_action if diagnostics is not None else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(
            record.initial_observation,
            fields=observation_fields,
        )
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
            next_observation = _trace_observation(
                transition.step.observation,
                fields=observation_fields,
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
            observation = next_observation
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_action(action: PolicyValue) -> list[float]:
    if type(action) is not list or len(action) != 3:
        raise ValueError("Hopper trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("Hopper trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    fields: tuple[str, ...],
) -> dict[str, float]:
    if type(observation) is not dict or set(observation) != set(fields):
        raise ValueError("Hopper trace observation is invalid")
    traced: dict[str, float] = {}
    for key in fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Hopper trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Hopper trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("Hopper trace metrics are invalid")
        elif key in _BOOL_METRICS:
            if type(value) is not bool:
                raise ValueError("Hopper trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("Hopper trace metrics are invalid")
        elif key == "failed_health_conditions":
            if type(value) is not list or any(type(item) is not str for item in value):
                raise ValueError("Hopper trace metrics are invalid")
        elif key in {
            "requested_action_by_joint",
            "actuator_gear_scaled_controls",
        }:
            if type(value) is not dict or set(value) != set(_ACTION_COMPONENTS):
                raise ValueError("Hopper trace metrics are invalid")
            for item in value.values():
                if type(item) is not float or not math.isfinite(item):
                    raise ValueError("Hopper trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("Hopper trace metrics are invalid")
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


__all__ = ["HopperBenchmark"]
