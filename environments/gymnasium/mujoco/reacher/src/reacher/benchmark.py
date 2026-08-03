"""A parameterized Reacher-v5 Benchmark with public traces."""

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

from .config import ReacherConfig
from .environment import ReacherEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-reacher/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 50
_MODEL_TIMESTEP_SECONDS = 0.01
_ACTUATOR_GEAR = 200.0
_OBSERVATION_FIELDS = (
    "joint0_cos",
    "joint1_cos",
    "joint0_sin",
    "joint1_sin",
    "target_x",
    "target_y",
    "joint0_angular_velocity",
    "joint1_angular_velocity",
    "fingertip_target_x",
    "fingertip_target_y",
)
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "seconds_per_step",
        "simulated_seconds",
        "requested_joint0_control",
        "requested_joint1_control",
        "gear_scaled_joint0_torque",
        "gear_scaled_joint1_torque",
        "action_squared_norm",
        "cumulative_action_squared_norm",
        "mean_action_squared_norm",
        "joint0_angle_radians",
        "joint1_relative_angle_radians",
        "joint0_unit_circle_error",
        "joint1_unit_circle_error",
        "joint0_angular_velocity",
        "joint1_angular_velocity",
        "maximum_absolute_joint0_angular_velocity",
        "maximum_absolute_joint1_angular_velocity",
        "target_x",
        "target_y",
        "target_radius",
        "fingertip_x",
        "fingertip_y",
        "fingertip_target_delta_x",
        "fingertip_target_delta_y",
        "initial_fingertip_target_distance",
        "fingertip_target_distance",
        "minimum_fingertip_target_distance",
        "maximum_fingertip_target_distance",
        "fingertip_target_distance_reduction",
        "closest_approach_step",
        "reward_distance",
        "reward_control",
        "reward_from_public_terms",
        "cumulative_reward_distance",
        "cumulative_reward_control",
        "cumulative_return",
        "terminal_reason",
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    initial_distance: float
    final_distance: float
    minimum_distance: float
    maximum_distance: float
    distance_reduction: float
    closest_approach_step: int
    maximum_absolute_joint0_angular_velocity: float
    maximum_absolute_joint1_angular_velocity: float
    mean_action_squared_norm: float
    cumulative_reward_distance: float
    cumulative_reward_control: float
    outcome: str


class ReacherBenchmark:
    """Mean Reacher return over deterministic Episode plans."""

    def __init__(self, config: ReacherConfig | None = None) -> None:
        if config is None:
            config = ReacherConfig()
        if type(config) is not ReacherConfig:
            raise TypeError("config must be ReacherConfig")
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
        return ReacherEnvironment(episode, config=self._config)

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
        mean_final_distance = _mean_or_none(tuple(item.final_distance for item in diagnostics))
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"mean final distance "
                    f"{_distance_summary(mean_final_distance)}."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_initial_distance": _mean_or_none(
                    tuple(item.initial_distance for item in diagnostics)
                ),
                "mean_final_distance": mean_final_distance,
                "mean_episode_minimum_distance": _mean_or_none(
                    tuple(item.minimum_distance for item in diagnostics)
                ),
                "mean_episode_maximum_distance": _mean_or_none(
                    tuple(item.maximum_distance for item in diagnostics)
                ),
                "mean_final_distance_reduction": _mean_or_none(
                    tuple(item.distance_reduction for item in diagnostics)
                ),
                "mean_closest_approach_step": _mean_or_none(
                    tuple(float(item.closest_approach_step) for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_joint0_angular_velocity": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_joint0_angular_velocity for item in diagnostics)
                    )
                ),
                "mean_episode_maximum_absolute_joint1_angular_velocity": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_joint1_angular_velocity for item in diagnostics)
                    )
                ),
                "mean_action_squared_norm": _mean_or_none(
                    tuple(item.mean_action_squared_norm for item in diagnostics)
                ),
                "mean_episode_distance_reward": _mean_or_none(
                    tuple(item.cumulative_reward_distance for item in diagnostics)
                ),
                "mean_episode_control_reward": _mean_or_none(
                    tuple(item.cumulative_reward_control for item in diagnostics)
                ),
                "episodes_ending_closer_to_target": sum(
                    item.distance_reduction > 0.0 for item in diagnostics
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
                    failure_return=self._failure_return,
                ),
            ),
        )


def _benchmark_spec(
    config: ReacherConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {
        "joint0_cos": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
            "meaning": "Cosine of shoulder angle relative to the world x-axis.",
        },
        "joint1_cos": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
            "meaning": "Cosine of elbow angle relative to link 1.",
        },
        "joint0_sin": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
            "meaning": "Sine of shoulder angle relative to the world x-axis.",
        },
        "joint1_sin": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
            "meaning": "Sine of elbow angle relative to link 1.",
        },
        "target_x": {
            "type": "float",
            "unit": "meters",
            "meaning": "Target world x-coordinate.",
        },
        "target_y": {
            "type": "float",
            "unit": "meters",
            "meaning": "Target world y-coordinate.",
        },
        "joint0_angular_velocity": {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": "Unclipped shoulder hinge qvel.",
        },
        "joint1_angular_velocity": {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": "Unclipped elbow hinge qvel relative to link 1.",
        },
        "fingertip_target_x": {
            "type": "float",
            "unit": "meters",
            "meaning": "fingertip_x - target_x.",
        },
        "fingertip_target_y": {
            "type": "float",
            "unit": "meters",
            "meaning": "fingertip_y - target_y.",
        },
    }
    return BenchmarkSpec(
        id="gymnasium/Reacher-v5/mean-return-v1",
        description=(
            "Apply two actuator controls, each scaled by gear 200, to move a "
            "two-link planar arm fingertip toward a target sampled inside a "
            "0.2-meter disk while minimizing squared control effort."
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
            "components": ["joint0_control", "joint1_control"],
            "actuator_gears": [_ACTUATOR_GEAR, _ACTUATOR_GEAR],
            "meaning": (
                "Requested controls are multiplied by actuator gear 200; "
                "they are not direct torques in newton-meters."
            ),
        },
        metadata={
            "environment": "Reacher-v5",
            "provider": "Gymnasium",
            "reward_threshold": -3.75,
            "official_model": "reacher.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": _MODEL_TIMESTEP_SECONDS,
            "seconds_per_step": _MODEL_TIMESTEP_SECONDS * config.frame_skip,
            "actuator_gears": [_ACTUATOR_GEAR, _ACTUATOR_GEAR],
            "reward_dist_weight": config.reward_dist_weight,
            "reward_control_weight": config.reward_control_weight,
            "reward_formula": (
                "-reward_dist_weight*fingertip_target_distance-reward_control_weight*sum(action^2)"
            ),
            "target_sampling": "uniform x,y in [-0.2,0.2], reject radius >=0.2",
            "arm_geom_lengths_meters": [0.1, 0.1],
            "fingertip_kinematic_offsets_meters": [0.1, 0.11],
            "natural_termination": "none",
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _failure_return(config: ReacherConfig) -> float:
    return -1_000.0 * max(
        1.0,
        config.reward_dist_weight,
        config.reward_control_weight,
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
        raise ValueError("Reacher terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("Reacher diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    closest_approach_step = final.get("closest_approach_step")
    if type(closest_approach_step) is not int:
        raise ValueError("Reacher closest approach step is invalid")
    return _EpisodeDiagnostics(
        initial_distance=_float_metric(
            final,
            "initial_fingertip_target_distance",
        ),
        final_distance=_float_metric(final, "fingertip_target_distance"),
        minimum_distance=_float_metric(
            final,
            "minimum_fingertip_target_distance",
        ),
        maximum_distance=_float_metric(
            final,
            "maximum_fingertip_target_distance",
        ),
        distance_reduction=_float_metric(
            final,
            "fingertip_target_distance_reduction",
        ),
        closest_approach_step=closest_approach_step,
        maximum_absolute_joint0_angular_velocity=_float_metric(
            final,
            "maximum_absolute_joint0_angular_velocity",
        ),
        maximum_absolute_joint1_angular_velocity=_float_metric(
            final,
            "maximum_absolute_joint1_angular_velocity",
        ),
        mean_action_squared_norm=_float_metric(
            final,
            "mean_action_squared_norm",
        ),
        cumulative_reward_distance=_float_metric(
            final,
            "cumulative_reward_distance",
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
        raise ValueError(f"Reacher metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _distance_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
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
                    "initial_distance": (
                        diagnostics.initial_distance if diagnostics is not None else None
                    ),
                    "final_distance": (
                        diagnostics.final_distance if diagnostics is not None else None
                    ),
                    "minimum_distance": (
                        diagnostics.minimum_distance if diagnostics is not None else None
                    ),
                    "distance_reduction": (
                        diagnostics.distance_reduction if diagnostics is not None else None
                    ),
                    "closest_approach_step": (
                        diagnostics.closest_approach_step if diagnostics is not None else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
            next_observation = _trace_observation(transition.step.observation)
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "action_components": {
                            "joint0_control": action[0],
                            "joint1_control": action[1],
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
        raise ValueError("Reacher trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("Reacher trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError("Reacher trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("Reacher trace observation is invalid")
    traced: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Reacher trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Reacher trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps", "closest_approach_step"}:
            if type(value) is not int:
                raise ValueError("Reacher trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("Reacher trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("Reacher trace metrics are invalid")
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


__all__ = ["ReacherBenchmark"]
