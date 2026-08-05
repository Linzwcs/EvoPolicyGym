"""A parameterized HalfCheetah-v5 Benchmark with public traces."""

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

from .config import HalfCheetahConfig
from .environment import HalfCheetahEnvironment
from .visual import (
    VISUAL_FRAME_SHAPE,
    VISUAL_MAX_FRAMES_PER_EPISODE,
    visual_capture_interval,
    visual_feedback,
)
from .visual import trace_metrics as strip_visual_metrics

_EPISODE_SEED_DOMAIN = b"evopolicygym-half-cheetah/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_000
_BODY_FIELDS = (
    "torso_z_position",
    "torso_pitch_angle",
    "back_thigh_angle",
    "back_shin_angle",
    "back_foot_angle",
    "front_thigh_angle",
    "front_shin_angle",
    "front_foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_pitch_angular_velocity",
    "back_thigh_angular_velocity",
    "back_shin_angular_velocity",
    "back_foot_angular_velocity",
    "front_thigh_angular_velocity",
    "front_shin_angular_velocity",
    "front_foot_angular_velocity",
)
_ACTION_COMPONENTS = (
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
)
_ACTUATOR_GEARS = (120.0, 90.0, 60.0, 120.0, 60.0, 30.0)
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
        "backward_or_stationary_step_fraction",
        "torso_z_position",
        "minimum_torso_z_position",
        "maximum_torso_z_position",
        "torso_pitch_radians",
        "torso_pitch_degrees",
        "maximum_absolute_torso_pitch_radians",
        "torso_x_velocity",
        "torso_z_velocity",
        "torso_pitch_angular_velocity",
        "reward_forward",
        "reward_control",
        "reward_from_public_terms",
        "cumulative_reward_forward",
        "cumulative_reward_control",
        "cumulative_return",
        "terminal_reason",
        "feedback_visual_capture_failed",
    }
)


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
    maximum_torso_z_position: float
    maximum_absolute_torso_pitch_radians: float
    mean_absolute_action: float
    cumulative_reward_forward: float
    cumulative_reward_control: float
    outcome: str


class HalfCheetahBenchmark:
    """Mean HalfCheetah return over deterministic Episode plans."""

    def __init__(self, config: HalfCheetahConfig | None = None) -> None:
        if config is None:
            config = HalfCheetahConfig()
        if type(config) is not HalfCheetahConfig:
            raise TypeError("config must be HalfCheetahConfig")
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
        return HalfCheetahEnvironment(episode, config=self._config)

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
        mean_final_x = _mean_or_none(
            tuple(item.final_x_position for item in diagnostics)
        )
        outcomes = tuple(_episode_outcome(record) for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        capture_interval = visual_capture_interval(_MAX_EPISODE_STEPS)
        visual_artifacts, visual_manifests, visual_unavailable = visual_feedback(
            records,
            capture_interval=capture_interval,
        )
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
                "mean_initial_x_position": _mean_or_none(
                    tuple(item.initial_x_position for item in diagnostics)
                ),
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
                "mean_episode_maximum_torso_z_position": _mean_or_none(
                    tuple(item.maximum_torso_z_position for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_torso_pitch_radians": _mean_or_none(
                    tuple(
                        item.maximum_absolute_torso_pitch_radians
                        for item in diagnostics
                    )
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
                "time_limit_episodes": outcomes.count("time_limit"),
                "incomplete_episodes": outcomes.count("incomplete"),
                "episodes": len(records),
                "policy_failures": failures,
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "rendered_frame_evidence_episodes": _artifact_episode_count(
                    visual_manifests,
                    "evidence_artifact",
                ),
                "video_episodes": _artifact_episode_count(
                    visual_manifests,
                    "video_artifact",
                ),
                "visual_episode_results": len(visual_manifests),
                "visual_capture_unavailable_episodes": visual_unavailable,
                "visual_frame_shape": list(VISUAL_FRAME_SHAPE),
                "visual_capture_interval_steps": capture_interval,
                "visual_frame_cap_per_episode": VISUAL_MAX_FRAMES_PER_EPISODE,
                "rendered_frame_evidence": visual_manifests,
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    observation_fields=self._observation_fields,
                    failure_return=self._failure_return,
                ),
                *visual_artifacts,
            ),
        )


def _artifact_episode_count(manifests: Sequence[PolicyValue], key: str) -> int:
    return sum(
        type(manifest) is dict and type(manifest.get(key)) is str
        for manifest in manifests
    )


def _benchmark_spec(
    config: HalfCheetahConfig,
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
        id="gymnasium/HalfCheetah-v5/mean-return-v1",
        description=(
            "Coordinate six bounded joint controls to propel a planar "
            "HalfCheetah in positive x. Reward is "
            "forward_reward_weight*x_velocity - ctrl_cost_weight*sum(action^2). "
            "The six actuator gears are 120,90,60,120,60,30 for back "
            "thigh/shin/foot then front thigh/shin/foot. HalfCheetah never "
            "terminates naturally and truncates at 1000 steps. Maximize mean return."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float]",
            "source_dtype": "float64",
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
            "components": list(_ACTION_COMPONENTS),
            "actuator_gears": list(_ACTUATOR_GEARS),
            "meaning": "Controls are applied in back thigh/shin/foot, then front thigh/shin/foot order.",
        },
        metadata={
            "environment": "HalfCheetah-v5",
            "provider": "Gymnasium",
            "reward_threshold": 4800.0,
            "official_model": "half_cheetah.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": 0.01,
            "seconds_per_step": 0.01 * config.frame_skip,
            "action_components": list(_ACTION_COMPONENTS),
            "actuator_gears": list(_ACTUATOR_GEARS),
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
            "reward_formula": (
                "forward_reward_weight*x_velocity-ctrl_cost_weight*sum(action^2)"
            ),
            "natural_termination": "none",
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _observation_fields(config: HalfCheetahConfig) -> tuple[str, ...]:
    return (
        _BODY_FIELDS
        if config.exclude_current_positions_from_observation
        else ("torso_x_position", *_BODY_FIELDS)
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
            "meaning": f"{readable_name.capitalize()} from root qvel.",
        }
    if name.endswith("_angular_velocity"):
        return {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": f"{readable_name.capitalize()} from qvel.",
        }
    return {
        "type": "float",
        "unit": "radians",
        "meaning": f"{readable_name.capitalize()} from qpos.",
    }


def _failure_return(config: HalfCheetahConfig) -> float:
    return -1_000.0 * max(
        1.0,
        config.forward_reward_weight,
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
        raise ValueError("HalfCheetah trace terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("HalfCheetah diagnostics require a transition")
    metrics = tuple(
        _trace_metrics(transition.step.metrics)
        for transition in record.transitions
    )
    final = metrics[-1]
    return _EpisodeDiagnostics(
        initial_x_position=_float_metric(final, "initial_x_position"),
        final_x_position=_float_metric(final, "x_position"),
        net_x_displacement=_float_metric(final, "net_x_displacement"),
        minimum_x_position=min(
            _float_metric(item, "x_position") for item in metrics
        ),
        maximum_x_position=max(
            _float_metric(item, "x_position") for item in metrics
        ),
        mean_x_velocity=statistics.fmean(
            _float_metric(item, "x_velocity") for item in metrics
        ),
        minimum_x_velocity=min(
            _float_metric(item, "x_velocity") for item in metrics
        ),
        maximum_x_velocity=max(
            _float_metric(item, "x_velocity") for item in metrics
        ),
        forward_step_fraction=statistics.fmean(
            1.0 if _float_metric(item, "x_velocity") > 0.0 else 0.0
            for item in metrics
        ),
        minimum_torso_z_position=min(
            _float_metric(item, "torso_z_position") for item in metrics
        ),
        maximum_torso_z_position=max(
            _float_metric(item, "torso_z_position") for item in metrics
        ),
        maximum_absolute_torso_pitch_radians=max(
            abs(_float_metric(item, "torso_pitch_radians")) for item in metrics
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
        outcome=_episode_outcome(record),
    )


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"HalfCheetah trace metric {name} is invalid")
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
                    "initial_x_position": (
                        diagnostics.initial_x_position
                        if diagnostics is not None
                        else None
                    ),
                    "net_x_displacement": (
                        diagnostics.net_x_displacement
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
                    "forward_step_fraction": (
                        diagnostics.forward_step_fraction
                        if diagnostics is not None
                        else None
                    ),
                    "maximum_absolute_torso_pitch_radians": (
                        diagnostics.maximum_absolute_torso_pitch_radians
                        if diagnostics is not None
                        else None
                    ),
                    "mean_absolute_action": (
                        diagnostics.mean_absolute_action
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
    if type(action) is not list or len(action) != 6:
        raise ValueError("HalfCheetah trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError("HalfCheetah trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
    *,
    fields: tuple[str, ...],
) -> dict[str, float]:
    if type(observation) is not dict or set(observation) != set(fields):
        raise ValueError("HalfCheetah trace observation is invalid")
    traced: dict[str, float] = {}
    for key in fields:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("HalfCheetah trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    metrics = strip_visual_metrics(metrics)
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("HalfCheetah trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("HalfCheetah trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("HalfCheetah trace metrics are invalid")
        elif key == "feedback_visual_capture_failed":
            if type(value) is not bool:
                raise ValueError("HalfCheetah trace metrics are invalid")
        elif key in {"requested_action_by_joint", "actuator_gear_scaled_controls"}:
            if type(value) is not dict or set(value) != set(_ACTION_COMPONENTS):
                raise ValueError("HalfCheetah trace metrics are invalid")
            for item in value.values():
                if type(item) is not float or not math.isfinite(item):
                    raise ValueError("HalfCheetah trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("HalfCheetah trace metrics are invalid")
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


__all__ = ["HalfCheetahBenchmark"]
