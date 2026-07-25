"""A parameterized Reacher-v5 Benchmark with public traces."""

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

from .config import ReacherConfig
from .environment import ReacherEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-reacher/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 50
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
                    f"mean final distance "
                    f"{_distance_summary(mean_final_distance)}."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_final_distance": mean_final_distance,
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
        },
        "joint1_cos": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "joint0_sin": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "joint1_sin": {
            "type": "float",
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "target_x": {"type": "float", "unit": "meters"},
        "target_y": {"type": "float", "unit": "meters"},
        "joint0_angular_velocity": {
            "type": "float",
            "unit": "radians_per_second",
        },
        "joint1_angular_velocity": {
            "type": "float",
            "unit": "radians_per_second",
        },
        "fingertip_target_x": {"type": "float", "unit": "meters"},
        "fingertip_target_y": {"type": "float", "unit": "meters"},
    }
    return BenchmarkSpec(
        id="gymnasium/Reacher-v5/mean-return-v1",
        description=(
            "Apply two joint torques to move a planar arm fingertip toward a "
            "random target while minimizing control effort. Maximize mean "
            "Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space={
            "type": "array",
            "shape": [2],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": ["joint0_torque", "joint1_torque"],
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
            "reward_dist_weight": config.reward_dist_weight,
            "reward_control_weight": config.reward_control_weight,
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


def _final_distance(record: EpisodeRecord) -> float:
    observation = (
        record.transitions[-1].step.observation
        if record.transitions
        else record.initial_observation
    )
    traced = _trace_observation(observation)
    return math.hypot(
        traced["fingertip_target_x"],
        traced["fingertip_target_y"],
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
                    "final_distance": (
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
    if type(action) is not list or len(action) != 2:
        raise ValueError("Reacher trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
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


def _trace_metrics(metrics: PolicyValue) -> dict[str, float]:
    if type(metrics) is not dict:
        raise ValueError("Reacher trace metrics are invalid")
    if set(metrics) != {"reward_distance", "reward_control"}:
        raise ValueError("Reacher trace metrics are invalid")
    traced: dict[str, float] = {}
    for key in ("reward_distance", "reward_control"):
        value = metrics[key]
        if type(value) is not float or not math.isfinite(value):
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
