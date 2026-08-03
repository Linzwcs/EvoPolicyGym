"""A parameterized Swimmer-v5 Benchmark with public traces."""

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

from .config import SwimmerConfig
from .environment import SwimmerEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-swimmer/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_000
_MODEL_TIMESTEP_SECONDS = 0.01
_ACTUATOR_GEAR = 150.0
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
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "seconds_per_step",
        "simulated_seconds",
        "requested_rotor1_control",
        "requested_rotor2_control",
        "gear_scaled_rotor1_torque",
        "gear_scaled_rotor2_torque",
        "action_squared_norm",
        "cumulative_action_squared_norm",
        "mean_action_squared_norm",
        "start_x_position",
        "start_y_position",
        "x_position",
        "y_position",
        "minimum_x_position",
        "maximum_x_position",
        "minimum_y_position",
        "maximum_y_position",
        "forward_displacement",
        "lateral_displacement",
        "maximum_absolute_lateral_displacement",
        "net_displacement",
        "path_length",
        "distance_from_origin",
        "step_average_x_velocity",
        "step_average_y_velocity",
        "observation_tip_x_velocity",
        "observation_tip_y_velocity",
        "mean_x_velocity",
        "mean_absolute_y_velocity",
        "minimum_x_velocity",
        "maximum_x_velocity",
        "maximum_absolute_y_velocity",
        "backward_step_fraction",
        "front_angle_radians",
        "rotor1_relative_angle_radians",
        "rotor2_relative_angle_radians",
        "reward_forward",
        "reward_control",
        "reward_from_public_terms",
        "cumulative_reward_forward",
        "cumulative_reward_control",
        "cumulative_return",
        "terminal_reason",
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    final_x_position: float
    final_y_position: float
    forward_displacement: float
    lateral_displacement: float
    maximum_absolute_lateral_displacement: float
    net_displacement: float
    path_length: float
    mean_x_velocity: float
    minimum_x_velocity: float
    maximum_x_velocity: float
    maximum_absolute_y_velocity: float
    backward_step_fraction: float
    mean_action_squared_norm: float
    cumulative_reward_forward: float
    cumulative_reward_control: float
    outcome: str


class SwimmerBenchmark:
    """Mean Swimmer return over deterministic Episode plans."""

    def __init__(self, config: SwimmerConfig | None = None) -> None:
        if config is None:
            config = SwimmerConfig()
        if type(config) is not SwimmerConfig:
            raise TypeError("config must be SwimmerConfig")
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
        return SwimmerEnvironment(episode, config=self._config)

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
                "mean_final_y_position": _mean_or_none(
                    tuple(item.final_y_position for item in diagnostics)
                ),
                "mean_forward_displacement": _mean_or_none(
                    tuple(item.forward_displacement for item in diagnostics)
                ),
                "mean_lateral_displacement": _mean_or_none(
                    tuple(item.lateral_displacement for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_lateral_displacement": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_lateral_displacement for item in diagnostics)
                    )
                ),
                "mean_net_displacement": _mean_or_none(
                    tuple(item.net_displacement for item in diagnostics)
                ),
                "mean_path_length": _mean_or_none(tuple(item.path_length for item in diagnostics)),
                "mean_x_velocity": _mean_or_none(
                    tuple(item.mean_x_velocity for item in diagnostics)
                ),
                "mean_episode_minimum_x_velocity": _mean_or_none(
                    tuple(item.minimum_x_velocity for item in diagnostics)
                ),
                "mean_episode_maximum_x_velocity": _mean_or_none(
                    tuple(item.maximum_x_velocity for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_y_velocity": _mean_or_none(
                    tuple(item.maximum_absolute_y_velocity for item in diagnostics)
                ),
                "mean_backward_step_fraction": _mean_or_none(
                    tuple(item.backward_step_fraction for item in diagnostics)
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
                "episodes_with_positive_forward_displacement": sum(
                    item.forward_displacement > 0.0 for item in diagnostics
                ),
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
    config: SwimmerConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    if not config.exclude_current_positions_from_observation:
        fields["tip_x_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Front-tip x qpos; positive x is the reward direction.",
        }
        fields["tip_y_position"] = {
            "type": "float",
            "unit": "meters",
            "meaning": "Front-tip y qpos.",
        }
    for name in _BODY_FIELDS:
        unit = (
            "meters_per_second"
            if name in {"tip_x_velocity", "tip_y_velocity"}
            else ("radians_per_second" if name.endswith("_angular_velocity") else "radians")
        )
        meaning = {
            "front_angle": "Absolute front-link angle.",
            "rotor1_angle": "Rotor 1 hinge angle relative to the front link.",
            "rotor2_angle": "Rotor 2 hinge angle relative to rotor 1.",
            "tip_x_velocity": "Instantaneous front-tip x qvel; not the step-average reward velocity.",
            "tip_y_velocity": "Instantaneous front-tip y qvel; not the step-average info velocity.",
            "front_angular_velocity": "Absolute front-link angular qvel.",
            "rotor1_angular_velocity": "Rotor 1 hinge qvel relative to the front link.",
            "rotor2_angular_velocity": "Rotor 2 hinge qvel relative to rotor 1.",
        }[name]
        fields[name] = {"type": "float", "unit": unit, "meaning": meaning}
    return BenchmarkSpec(
        id="gymnasium/Swimmer-v5/mean-return-v1",
        description=(
            "Coordinate two rotor controls, each scaled by actuator gear 150, "
            "to propel a three-link swimmer in positive x. Reward uses the "
            "step-average x velocity minus squared-control cost."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float]",
            "source_dtype": "float64",
            "fields": fields,
        },
        action_space={
            "type": "array",
            "shape": [2],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "policy_carrier": "list[float]",
            "components": ["rotor1_control", "rotor2_control"],
            "actuator_gears": [_ACTUATOR_GEAR, _ACTUATOR_GEAR],
            "meaning": (
                "Requested controls are multiplied by actuator gear 150; "
                "they are not direct torques in newton-meters."
            ),
        },
        metadata={
            "environment": "Swimmer-v5",
            "provider": "Gymnasium",
            "reward_threshold": 360.0,
            "official_model": "swimmer.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": _MODEL_TIMESTEP_SECONDS,
            "seconds_per_step": _MODEL_TIMESTEP_SECONDS * config.frame_skip,
            "actuator_gears": [_ACTUATOR_GEAR, _ACTUATOR_GEAR],
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "reward_formula": (
                "forward_reward_weight*((x_after-x_before)/seconds_per_step)-"
                "ctrl_cost_weight*sum(action^2)"
            ),
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
            "natural_termination": "none",
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _observation_fields(config: SwimmerConfig) -> tuple[str, ...]:
    return (
        _BODY_FIELDS
        if config.exclude_current_positions_from_observation
        else ("tip_x_position", "tip_y_position", *_BODY_FIELDS)
    )


def _failure_return(config: SwimmerConfig) -> float:
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
        2.0 * config.ctrl_cost_weight,
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
        raise ValueError("Swimmer terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("Swimmer diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        final_x_position=_float_metric(final, "x_position"),
        final_y_position=_float_metric(final, "y_position"),
        forward_displacement=_float_metric(final, "forward_displacement"),
        lateral_displacement=_float_metric(final, "lateral_displacement"),
        maximum_absolute_lateral_displacement=_float_metric(
            final,
            "maximum_absolute_lateral_displacement",
        ),
        net_displacement=_float_metric(final, "net_displacement"),
        path_length=_float_metric(final, "path_length"),
        mean_x_velocity=_float_metric(final, "mean_x_velocity"),
        minimum_x_velocity=_float_metric(final, "minimum_x_velocity"),
        maximum_x_velocity=_float_metric(final, "maximum_x_velocity"),
        maximum_absolute_y_velocity=_float_metric(
            final,
            "maximum_absolute_y_velocity",
        ),
        backward_step_fraction=_float_metric(
            final,
            "backward_step_fraction",
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
        outcome=_episode_outcome(record),
    )


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"Swimmer metric {name} is invalid")
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
                    "lateral_displacement": (
                        diagnostics.lateral_displacement if diagnostics is not None else None
                    ),
                    "path_length": (diagnostics.path_length if diagnostics is not None else None),
                    "backward_step_fraction": (
                        diagnostics.backward_step_fraction if diagnostics is not None else None
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
                        "action_components": {
                            "rotor1_control": action[0],
                            "rotor2_control": action[1],
                        },
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
    if type(action) is not list or len(action) != 2:
        raise ValueError("Swimmer trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("Swimmer trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    fields: tuple[str, ...],
) -> dict[str, float]:
    if type(observation) is not dict or set(observation) != set(fields):
        raise ValueError("Swimmer trace observation is invalid")
    traced: dict[str, float] = {}
    for key in fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Swimmer trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Swimmer trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("Swimmer trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("Swimmer trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("Swimmer trace metrics are invalid")
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


__all__ = ["SwimmerBenchmark"]
