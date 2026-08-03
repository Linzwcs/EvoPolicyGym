"""A parameterized Walker2d-v5 Benchmark with public traces."""

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

from .config import Walker2dConfig
from .environment import Walker2dEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-walker2d/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_000
_MODEL_TIMESTEP_SECONDS = 0.002
_ACTUATOR_GEAR = 100.0
_VELOCITY_CLIP_LIMIT = 10.0
_ACTION_NAMES = (
    "right_thigh",
    "right_leg",
    "right_foot",
    "left_thigh",
    "left_leg",
    "left_foot",
)
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
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "seconds_per_step",
        "simulated_seconds",
        *(f"requested_{name}_control" for name in _ACTION_NAMES),
        *(f"gear_scaled_{name}_torque" for name in _ACTION_NAMES),
        "action_squared_norm",
        "cumulative_action_squared_norm",
        "mean_action_squared_norm",
        "start_x_position",
        "x_position",
        "minimum_x_position",
        "maximum_x_position",
        "forward_displacement",
        "step_average_x_velocity",
        "observation_torso_x_velocity",
        "mean_x_velocity",
        "minimum_x_velocity",
        "maximum_x_velocity",
        "backward_step_fraction",
        "torso_z_position",
        "minimum_torso_z_position",
        "maximum_torso_z_position",
        "healthy_z_lower_bound",
        "healthy_z_upper_bound",
        "height_health_margin",
        "minimum_height_health_margin",
        "torso_angle_radians",
        "minimum_torso_angle_radians",
        "maximum_torso_angle_radians",
        "healthy_angle_lower_bound",
        "healthy_angle_upper_bound",
        "angle_health_margin",
        "minimum_angle_health_margin",
        "healthy",
        "unhealthy_step_fraction",
        "maximum_observed_absolute_velocity",
        "velocity_observation_at_clip_limit",
        "velocity_clip_limit_step_fraction",
        "z_offset_from_model_initial_pose",
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


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    final_x_position: float
    forward_displacement: float
    minimum_x_velocity: float
    maximum_x_velocity: float
    backward_step_fraction: float
    final_torso_z_position: float
    minimum_torso_z_position: float
    maximum_torso_z_position: float
    minimum_height_health_margin: float
    final_torso_angle_radians: float
    minimum_torso_angle_radians: float
    maximum_torso_angle_radians: float
    minimum_angle_health_margin: float
    unhealthy_step_fraction: float
    velocity_clip_limit_step_fraction: float
    mean_action_squared_norm: float
    cumulative_reward_forward: float
    cumulative_reward_control: float
    cumulative_reward_survive: float
    outcome: str


class Walker2dBenchmark:
    """Mean Walker2d return over deterministic Episode plans."""

    def __init__(self, config: Walker2dConfig | None = None) -> None:
        if config is None:
            config = Walker2dConfig()
        if type(config) is not Walker2dConfig:
            raise TypeError("config must be Walker2dConfig")
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
        return Walker2dEnvironment(episode, config=self._config)

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
                    f"{_position_summary(mean_final_x)}."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_final_x_position": mean_final_x,
                "mean_forward_displacement": _mean_or_none(
                    tuple(item.forward_displacement for item in diagnostics)
                ),
                "mean_episode_minimum_x_velocity": _mean_or_none(
                    tuple(item.minimum_x_velocity for item in diagnostics)
                ),
                "mean_episode_maximum_x_velocity": _mean_or_none(
                    tuple(item.maximum_x_velocity for item in diagnostics)
                ),
                "mean_backward_step_fraction": _mean_or_none(
                    tuple(item.backward_step_fraction for item in diagnostics)
                ),
                "mean_final_torso_z_position": _mean_or_none(
                    tuple(item.final_torso_z_position for item in diagnostics)
                ),
                "mean_episode_minimum_torso_z_position": _mean_or_none(
                    tuple(item.minimum_torso_z_position for item in diagnostics)
                ),
                "mean_episode_maximum_torso_z_position": _mean_or_none(
                    tuple(item.maximum_torso_z_position for item in diagnostics)
                ),
                "mean_episode_minimum_height_health_margin": _mean_or_none(
                    tuple(item.minimum_height_health_margin for item in diagnostics)
                ),
                "mean_final_torso_angle_radians": _mean_or_none(
                    tuple(item.final_torso_angle_radians for item in diagnostics)
                ),
                "mean_episode_minimum_torso_angle_radians": _mean_or_none(
                    tuple(item.minimum_torso_angle_radians for item in diagnostics)
                ),
                "mean_episode_maximum_torso_angle_radians": _mean_or_none(
                    tuple(item.maximum_torso_angle_radians for item in diagnostics)
                ),
                "mean_episode_minimum_angle_health_margin": _mean_or_none(
                    tuple(item.minimum_angle_health_margin for item in diagnostics)
                ),
                "mean_unhealthy_step_fraction": _mean_or_none(
                    tuple(item.unhealthy_step_fraction for item in diagnostics)
                ),
                "mean_velocity_clip_limit_step_fraction": _mean_or_none(
                    tuple(item.velocity_clip_limit_step_fraction for item in diagnostics)
                ),
                "mean_action_squared_norm": _mean_or_none(
                    tuple(item.mean_action_squared_norm for item in diagnostics)
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
                "full_horizon_episodes": outcomes.count("time_limit"),
                "unhealthy_height_episodes": outcomes.count("unhealthy_height"),
                "unhealthy_angle_episodes": outcomes.count("unhealthy_angle"),
                "unhealthy_height_and_angle_episodes": outcomes.count("unhealthy_height_and_angle"),
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
    config: Walker2dConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["torso_x_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Torso root x qpos; positive x is the reward direction.",
        }
    for name in _BODY_FIELDS:
        if name == "torso_z_position":
            unit = "meters"
        elif name in {
            "torso_x_velocity",
            "torso_z_velocity",
        }:
            unit = "meters_per_second"
        elif name.endswith("_angular_velocity"):
            unit = "radians_per_second"
        else:
            unit = "radians"
        meaning = (
            "Velocity clipped to [-10, 10] by Gymnasium."
            if "velocity" in name
            else (
                "Torso root height; health uses strict configured bounds."
                if name == "torso_z_position"
                else (
                    "Torso pitch; health uses strict configured bounds."
                    if name == "torso_angle"
                    else "Joint angle relative to its parent body."
                )
            )
        )
        fields[name] = {"type": "float", "unit": unit, "meaning": meaning}
    return BenchmarkSpec(
        id="gymnasium/Walker2d-v5/mean-return-v1",
        description=(
            "Coordinate six leg controls, each scaled by actuator gear 100, "
            "to keep a planar walker strictly inside configured height and "
            "torso-angle bounds while moving in positive x."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float]",
            "source_dtype": "float64",
            "velocity_clipping": [-_VELOCITY_CLIP_LIMIT, _VELOCITY_CLIP_LIMIT],
            "fields": fields,
        },
        action_space={
            "type": "array",
            "shape": [6],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "policy_carrier": "list[float]",
            "components": [f"{name}_control" for name in _ACTION_NAMES],
            "actuator_gears": [_ACTUATOR_GEAR] * 6,
            "meaning": (
                "Each requested control is multiplied by actuator gear 100; "
                "controls are not direct torques in newton-meters."
            ),
        },
        metadata={
            "environment": "Walker2d-v5",
            "provider": "Gymnasium",
            "official_model": "walker2d_v5.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": _MODEL_TIMESTEP_SECONDS,
            "seconds_per_step": _MODEL_TIMESTEP_SECONDS * config.frame_skip,
            "actuator_gears": [_ACTUATOR_GEAR] * 6,
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "healthy_reward": config.healthy_reward,
            "reward_formula": (
                "forward_reward_weight*((x_after-x_before)/seconds_per_step)+"
                "healthy_reward_if_strictly_healthy-ctrl_cost_weight*sum(action^2)"
            ),
            "terminate_when_unhealthy": config.terminate_when_unhealthy,
            "healthy_z_range": list(config.healthy_z_range),
            "healthy_angle_range": list(config.healthy_angle_range),
            "healthy_bounds_inclusive": False,
            "observation_velocity_clipping": [
                -_VELOCITY_CLIP_LIMIT,
                _VELOCITY_CLIP_LIMIT,
            ],
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _observation_fields(config: Walker2dConfig) -> tuple[str, ...]:
    return (
        _BODY_FIELDS
        if config.exclude_current_positions_from_observation
        else ("torso_x_position", *_BODY_FIELDS)
    )


def _failure_return(config: Walker2dConfig) -> float:
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
        config.healthy_reward,
        6.0 * config.ctrl_cost_weight,
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
        raise ValueError("Walker2d terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("Walker2d diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        final_x_position=_float_metric(final, "x_position"),
        forward_displacement=_float_metric(final, "forward_displacement"),
        minimum_x_velocity=_float_metric(final, "minimum_x_velocity"),
        maximum_x_velocity=_float_metric(final, "maximum_x_velocity"),
        backward_step_fraction=_float_metric(
            final,
            "backward_step_fraction",
        ),
        final_torso_z_position=_float_metric(final, "torso_z_position"),
        minimum_torso_z_position=_float_metric(
            final,
            "minimum_torso_z_position",
        ),
        maximum_torso_z_position=_float_metric(
            final,
            "maximum_torso_z_position",
        ),
        minimum_height_health_margin=_float_metric(
            final,
            "minimum_height_health_margin",
        ),
        final_torso_angle_radians=_float_metric(
            final,
            "torso_angle_radians",
        ),
        minimum_torso_angle_radians=_float_metric(
            final,
            "minimum_torso_angle_radians",
        ),
        maximum_torso_angle_radians=_float_metric(
            final,
            "maximum_torso_angle_radians",
        ),
        minimum_angle_health_margin=_float_metric(
            final,
            "minimum_angle_health_margin",
        ),
        unhealthy_step_fraction=_float_metric(
            final,
            "unhealthy_step_fraction",
        ),
        velocity_clip_limit_step_fraction=_float_metric(
            final,
            "velocity_clip_limit_step_fraction",
        ),
        mean_action_squared_norm=_float_metric(
            final,
            "mean_action_squared_norm",
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


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"Walker2d metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


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
                    "final_x_position": (
                        diagnostics.final_x_position if diagnostics is not None else None
                    ),
                    "forward_displacement": (
                        diagnostics.forward_displacement if diagnostics is not None else None
                    ),
                    "minimum_height_health_margin": (
                        diagnostics.minimum_height_health_margin
                        if diagnostics is not None
                        else None
                    ),
                    "minimum_angle_health_margin": (
                        diagnostics.minimum_angle_health_margin if diagnostics is not None else None
                    ),
                    "velocity_clip_limit_step_fraction": (
                        diagnostics.velocity_clip_limit_step_fraction
                        if diagnostics is not None
                        else None
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
                        "action_components": dict(
                            zip(
                                (f"{name}_control" for name in _ACTION_NAMES),
                                action,
                                strict=True,
                            )
                        ),
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
    if type(action) is not list or len(action) != 6:
        raise ValueError("Walker2d trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("Walker2d trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    fields: tuple[str, ...],
) -> dict[str, float]:
    if type(observation) is not dict or set(observation) != set(fields):
        raise ValueError("Walker2d trace observation is invalid")
    traced: dict[str, float] = {}
    for key in fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Walker2d trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Walker2d trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("Walker2d trace metrics are invalid")
        elif key in {"healthy", "velocity_observation_at_clip_limit"}:
            if type(value) is not bool:
                raise ValueError("Walker2d trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("Walker2d trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("Walker2d trace metrics are invalid")
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


__all__ = ["Walker2dBenchmark"]
