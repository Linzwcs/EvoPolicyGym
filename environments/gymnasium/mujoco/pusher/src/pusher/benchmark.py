"""A parameterized Pusher-v5 Benchmark with public traces."""

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

from .config import PusherConfig
from .environment import PusherEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-pusher/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 100
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
        distances = tuple(
            _final_distance(record)
            for record in records
            if record.policy_failure is None
        )
        mean_final_distance = (
            statistics.fmean(distances) if distances else None
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
                "mean_final_object_goal_distance": mean_final_distance,
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
        }
    for name in _JOINT_NAMES:
        fields[f"{name}_angular_velocity"] = {
            "type": "float",
            "unit": "radians_per_second",
        }
    for prefix in ("fingertip", "object", "goal"):
        for axis in ("x", "y", "z"):
            fields[f"{prefix}_{axis}"] = {
                "type": "float",
                "unit": "meters",
            }
    return BenchmarkSpec(
        id="gymnasium/Pusher-v5/mean-return-v1",
        description=(
            "Apply seven arm joint torques to move a cylinder to a fixed goal. "
            "Balance object-goal distance, fingertip-object distance, and "
            "control effort. Maximize mean Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space={
            "type": "array",
            "shape": [7],
            "items": {
                "type": "float",
                "minimum": -2.0,
                "maximum": 2.0,
            },
            "components": [f"{name}_torque" for name in _JOINT_NAMES],
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
            "reward_near_weight": config.reward_near_weight,
            "reward_dist_weight": config.reward_dist_weight,
            "reward_control_weight": config.reward_control_weight,
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


def _final_distance(record: EpisodeRecord) -> float:
    observation = (
        record.transitions[-1].step.observation
        if record.transitions
        else record.initial_observation
    )
    traced = _trace_observation(observation)
    return math.sqrt(
        sum(
            (
                traced[f"object_{axis}"] - traced[f"goal_{axis}"]
            )
            ** 2
            for axis in ("x", "y", "z")
        )
    )


def _distance_summary(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
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
                    "final_object_goal_distance": (
                        _final_distance(record)
                        if record.policy_failure is None
                        else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
            next_observation = _trace_observation(
                transition.step.observation
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
                        "reward_terms": _trace_metrics(
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
    if type(action) is not list or len(action) != 7:
        raise ValueError("Pusher trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -2.0 <= value <= 2.0
        ):
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


def _trace_metrics(metrics: PolicyValue) -> dict[str, float]:
    if type(metrics) is not dict:
        raise ValueError("Pusher trace metrics are invalid")
    required = {
        "reward_distance",
        "reward_control",
        "reward_near",
    }
    if set(metrics) != required:
        raise ValueError("Pusher trace metrics are invalid")
    traced: dict[str, float] = {}
    for key in sorted(required):
        value = metrics[key]
        if type(value) is not float or not math.isfinite(value):
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
