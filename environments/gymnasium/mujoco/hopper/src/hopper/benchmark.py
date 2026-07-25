"""A parameterized Hopper-v5 Benchmark with public traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence

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
    "torso_angle",
    "thigh_angle",
    "leg_angle",
    "foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_angular_velocity",
    "thigh_angular_velocity",
    "leg_angular_velocity",
    "foot_angular_velocity",
)
_METRIC_FIELDS = (
    "x_position",
    "z_distance_from_origin",
    "x_velocity",
    "reward_forward",
    "reward_control",
    "reward_survive",
)


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
        }
    units = {
        "torso_z_position": "meters",
        "torso_angle": "radians",
        "thigh_angle": "radians",
        "leg_angle": "radians",
        "foot_angle": "radians",
        "torso_x_velocity": "meters_per_second",
        "torso_z_velocity": "meters_per_second",
        "torso_angular_velocity": "radians_per_second",
        "thigh_angular_velocity": "radians_per_second",
        "leg_angular_velocity": "radians_per_second",
        "foot_angular_velocity": "radians_per_second",
    }
    for name in _BODY_FIELDS:
        fields[name] = {"type": "float", "unit": units[name]}
    return BenchmarkSpec(
        id="gymnasium/Hopper-v5/mean-return-v1",
        description=(
            "Coordinate three joint torques to keep a one-legged Hopper "
            "healthy and move it in the positive x direction. Maximize mean "
            "Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space={
            "type": "array",
            "shape": [3],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": [
                "thigh_torque",
                "leg_torque",
                "foot_torque",
            ],
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
            "forward_reward_weight": config.forward_reward_weight,
            "ctrl_cost_weight": config.ctrl_cost_weight,
            "healthy_reward": config.healthy_reward,
            "terminate_when_unhealthy": config.terminate_when_unhealthy,
            "healthy_state_range": list(config.healthy_state_range),
            "healthy_z_range": list(config.healthy_z_range),
            "healthy_angle_range": list(config.healthy_angle_range),
            "reset_noise_scale": config.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                config.exclude_current_positions_from_observation
            ),
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


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


def _final_x_position(record: EpisodeRecord) -> float:
    metrics = record.transitions[-1].step.metrics
    traced = _trace_metrics(metrics)
    return traced["x_position"]


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
    if type(action) is not list or len(action) != 3:
        raise ValueError("Hopper trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
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


def _trace_metrics(metrics: PolicyValue) -> dict[str, float]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("Hopper trace metrics are invalid")
    traced: dict[str, float] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if type(value) is not float or not math.isfinite(value):
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
