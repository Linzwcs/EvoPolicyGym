"""A parameterized Pusher-v5 Benchmark with public traces."""

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

from .config import PusherConfig
from .environment import PusherEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-pusher/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 100
_MODEL_TIMESTEP_SECONDS = 0.01
_ACTUATOR_GEAR = 1.0
_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "upper_arm_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
)
_OBSERVATION_FIELDS = (
    *(f"{name}_angle" for name in _JOINT_NAMES),
    *(f"{name}_angular_velocity" for name in _JOINT_NAMES),
    "fingertip_x",
    "fingertip_y",
    "fingertip_z",
    "object_x",
    "object_y",
    "object_z",
    "goal_x",
    "goal_y",
    "goal_z",
)
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "seconds_per_step",
        "simulated_seconds",
        "action_squared_norm",
        "cumulative_action_squared_norm",
        "mean_action_squared_norm",
        "fingertip_object_delta_x",
        "fingertip_object_delta_y",
        "fingertip_object_delta_z",
        "fingertip_object_distance",
        "minimum_fingertip_object_distance",
        "object_goal_delta_x",
        "object_goal_delta_y",
        "object_goal_delta_z",
        "initial_object_goal_distance",
        "object_goal_distance",
        "minimum_object_goal_distance",
        "maximum_object_goal_distance",
        "object_goal_distance_reduction",
        "object_goal_fraction_remaining",
        "object_moved_toward_goal",
        "object_displacement_x",
        "object_displacement_y",
        "object_displacement_z",
        "object_displacement",
        "maximum_object_displacement",
        "reward_distance",
        "reward_near",
        "reward_control",
        "reward_from_public_terms",
        "cumulative_reward_distance",
        "cumulative_reward_near",
        "cumulative_reward_control",
        "cumulative_return",
        "terminal_reason",
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    initial_object_goal_distance: float
    final_object_goal_distance: float
    minimum_object_goal_distance: float
    maximum_object_goal_distance: float
    object_goal_distance_reduction: float
    object_goal_fraction_remaining: float
    final_fingertip_object_distance: float
    minimum_fingertip_object_distance: float
    final_object_displacement: float
    maximum_object_displacement: float
    mean_action_squared_norm: float
    cumulative_reward_distance: float
    cumulative_reward_near: float
    cumulative_reward_control: float
    outcome: str


class PusherBenchmark:
    """Mean Pusher return over deterministic Episode plans."""

    def __init__(self, config: PusherConfig | None = None) -> None:
        if config is None:
            config = PusherConfig()
        if type(config) is not PusherConfig:
            raise TypeError("config must be PusherConfig")
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
        return PusherEnvironment(episode, config=self._config)

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
        mean_final_distance = _mean_or_none(
            tuple(item.final_object_goal_distance for item in diagnostics)
        )
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"mean final object-goal distance "
                    f"{_distance_summary(mean_final_distance)}."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_initial_object_goal_distance": _mean_or_none(
                    tuple(item.initial_object_goal_distance for item in diagnostics)
                ),
                "mean_final_object_goal_distance": mean_final_distance,
                "mean_episode_minimum_object_goal_distance": _mean_or_none(
                    tuple(item.minimum_object_goal_distance for item in diagnostics)
                ),
                "mean_episode_maximum_object_goal_distance": _mean_or_none(
                    tuple(item.maximum_object_goal_distance for item in diagnostics)
                ),
                "mean_object_goal_distance_reduction": _mean_or_none(
                    tuple(item.object_goal_distance_reduction for item in diagnostics)
                ),
                "mean_object_goal_fraction_remaining": _mean_or_none(
                    tuple(item.object_goal_fraction_remaining for item in diagnostics)
                ),
                "mean_final_fingertip_object_distance": _mean_or_none(
                    tuple(item.final_fingertip_object_distance for item in diagnostics)
                ),
                "mean_episode_minimum_fingertip_object_distance": (
                    _mean_or_none(
                        tuple(item.minimum_fingertip_object_distance for item in diagnostics)
                    )
                ),
                "mean_final_object_displacement": _mean_or_none(
                    tuple(item.final_object_displacement for item in diagnostics)
                ),
                "mean_episode_maximum_object_displacement": _mean_or_none(
                    tuple(item.maximum_object_displacement for item in diagnostics)
                ),
                "mean_action_squared_norm": _mean_or_none(
                    tuple(item.mean_action_squared_norm for item in diagnostics)
                ),
                "mean_episode_distance_reward": _mean_or_none(
                    tuple(item.cumulative_reward_distance for item in diagnostics)
                ),
                "mean_episode_near_reward": _mean_or_none(
                    tuple(item.cumulative_reward_near for item in diagnostics)
                ),
                "mean_episode_control_reward": _mean_or_none(
                    tuple(item.cumulative_reward_control for item in diagnostics)
                ),
                "episodes_ending_closer_to_goal": sum(
                    item.object_goal_distance_reduction > 0.0 for item in diagnostics
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
    config: PusherConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {}
    for name in _JOINT_NAMES:
        fields[f"{name}_angle"] = {
            "type": "float",
            "unit": "radians",
            "meaning": f"Unclipped {name} hinge qpos.",
        }
    for name in _JOINT_NAMES:
        fields[f"{name}_angular_velocity"] = {
            "type": "float",
            "unit": "radians_per_second",
            "meaning": f"Unclipped {name} hinge qvel.",
        }
    for prefix in ("fingertip", "object", "goal"):
        for axis in ("x", "y", "z"):
            fields[f"{prefix}_{axis}"] = {
                "type": "float",
                "unit": "meters",
                "meaning": f"{axis}-coordinate of {prefix} body center of mass.",
            }
    return BenchmarkSpec(
        id="gymnasium/Pusher-v5/mean-return-v1",
        description=(
            "Apply seven direct joint torques (gear 1) to move a cylinder to "
            "a fixed goal. Reward is the negative weighted object-goal "
            "distance, fingertip-object distance, and squared torque norm."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float]",
            "source_dtype": "float64",
            "fields": fields,
        },
        action_space={
            "type": "array",
            "shape": [7],
            "items": {
                "type": "float",
                "minimum": -2.0,
                "maximum": 2.0,
            },
            "policy_carrier": "list[float]",
            "components": [f"{name}_torque" for name in _JOINT_NAMES],
            "actuator_gears": [_ACTUATOR_GEAR] * 7,
            "unit": "newton_meter",
            "meaning": "Seven MuJoCo motor controls with gear 1; no clipping or remapping.",
        },
        metadata={
            "environment": "Pusher-v5",
            "provider": "Gymnasium",
            "reward_threshold": 0.0,
            "official_model": "pusher_v5.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": _MODEL_TIMESTEP_SECONDS,
            "seconds_per_step": _MODEL_TIMESTEP_SECONDS * config.frame_skip,
            "actuator_gears": [_ACTUATOR_GEAR] * 7,
            "reward_near_weight": config.reward_near_weight,
            "reward_dist_weight": config.reward_dist_weight,
            "reward_control_weight": config.reward_control_weight,
            "reward_formula": (
                "-reward_dist_weight*object_goal_distance-"
                "reward_near_weight*fingertip_object_distance-"
                "reward_control_weight*sum(action^2)"
            ),
            "object_initial_xy_ranges": {
                "x": [-0.3, 0.0],
                "y": [-0.2, 0.2],
            },
            "minimum_initial_object_goal_planar_distance": 0.17,
            "natural_termination": "none",
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _failure_return(config: PusherConfig) -> float:
    return -1_000.0 * max(
        1.0,
        config.reward_near_weight,
        config.reward_dist_weight,
        3.0 * config.reward_control_weight,
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
        raise ValueError("Pusher terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("Pusher diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        initial_object_goal_distance=_float_metric(
            final,
            "initial_object_goal_distance",
        ),
        final_object_goal_distance=_float_metric(
            final,
            "object_goal_distance",
        ),
        minimum_object_goal_distance=_float_metric(
            final,
            "minimum_object_goal_distance",
        ),
        maximum_object_goal_distance=_float_metric(
            final,
            "maximum_object_goal_distance",
        ),
        object_goal_distance_reduction=_float_metric(
            final,
            "object_goal_distance_reduction",
        ),
        object_goal_fraction_remaining=_float_metric(
            final,
            "object_goal_fraction_remaining",
        ),
        final_fingertip_object_distance=_float_metric(
            final,
            "fingertip_object_distance",
        ),
        minimum_fingertip_object_distance=_float_metric(
            final,
            "minimum_fingertip_object_distance",
        ),
        final_object_displacement=_float_metric(
            final,
            "object_displacement",
        ),
        maximum_object_displacement=_float_metric(
            final,
            "maximum_object_displacement",
        ),
        mean_action_squared_norm=_float_metric(
            final,
            "mean_action_squared_norm",
        ),
        cumulative_reward_distance=_float_metric(
            final,
            "cumulative_reward_distance",
        ),
        cumulative_reward_near=_float_metric(
            final,
            "cumulative_reward_near",
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
        raise ValueError(f"Pusher metric {name} is invalid")
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
                    "initial_object_goal_distance": (
                        diagnostics.initial_object_goal_distance
                        if diagnostics is not None
                        else None
                    ),
                    "final_object_goal_distance": (
                        diagnostics.final_object_goal_distance if diagnostics is not None else None
                    ),
                    "minimum_object_goal_distance": (
                        diagnostics.minimum_object_goal_distance
                        if diagnostics is not None
                        else None
                    ),
                    "object_goal_distance_reduction": (
                        diagnostics.object_goal_distance_reduction
                        if diagnostics is not None
                        else None
                    ),
                    "minimum_fingertip_object_distance": (
                        diagnostics.minimum_fingertip_object_distance
                        if diagnostics is not None
                        else None
                    ),
                    "maximum_object_displacement": (
                        diagnostics.maximum_object_displacement if diagnostics is not None else None
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
                        "action_components": dict(
                            zip(
                                (f"{name}_torque" for name in _JOINT_NAMES),
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
    if type(action) is not list or len(action) != 7:
        raise ValueError("Pusher trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if type(value) is not float or not math.isfinite(value) or not -2.0 <= value <= 2.0:
            raise ValueError("Pusher trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError("Pusher trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("Pusher trace observation is invalid")
    traced: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Pusher trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Pusher trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("Pusher trace metrics are invalid")
        elif key == "object_moved_toward_goal":
            if type(value) is not bool:
                raise ValueError("Pusher trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("Pusher trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("Pusher trace metrics are invalid")
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


__all__ = ["PusherBenchmark"]
