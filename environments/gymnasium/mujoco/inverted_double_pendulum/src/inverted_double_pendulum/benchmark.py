"""Parameterized InvertedDoublePendulum-v5 Benchmark and traces."""

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

from .config import InvertedDoublePendulumConfig
from .environment import InvertedDoublePendulumEnvironment

_EPISODE_SEED_DOMAIN = (
    b"evopolicygym-inverted-double-pendulum/episode-seed/v1\0"
)
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 1_000
_OBSERVATION_FIELDS = (
    "cart_position",
    "pole1_sin",
    "pole2_sin",
    "pole1_cos",
    "pole2_cos",
    "cart_velocity",
    "pole1_angular_velocity",
    "pole2_angular_velocity",
    "cart_constraint_force",
)


class InvertedDoublePendulumBenchmark:
    """Mean double-pendulum return over deterministic Episode plans."""

    def __init__(
        self,
        config: InvertedDoublePendulumConfig | None = None,
    ) -> None:
        if config is None:
            config = InvertedDoublePendulumConfig()
        if type(config) is not InvertedDoublePendulumConfig:
            raise TypeError(
                "config must be InvertedDoublePendulumConfig"
            )
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
        return InvertedDoublePendulumEnvironment(
            episode,
            config=self._config,
        )

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
        balanced = sum(_balanced_full_horizon(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{balanced} balanced for all 1000 steps."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "full_horizon_balances": balanced,
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
    config: InvertedDoublePendulumConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="gymnasium/InvertedDoublePendulum-v5/mean-return-v1",
        description=(
            "Apply horizontal force to a cart to balance two serial poles. "
            "Balance survival reward against free-tip distance and angular "
            "velocity penalties. Maximize mean Episode return."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "cart_position": {"type": "float", "unit": "meters"},
                "pole1_sin": {
                    "type": "float",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "pole2_sin": {
                    "type": "float",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "pole1_cos": {
                    "type": "float",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "pole2_cos": {
                    "type": "float",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "cart_velocity": {
                    "type": "float",
                    "unit": "meters_per_second",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
                "pole1_angular_velocity": {
                    "type": "float",
                    "unit": "radians_per_second",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
                "pole2_angular_velocity": {
                    "type": "float",
                    "unit": "radians_per_second",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
                "cart_constraint_force": {
                    "type": "float",
                    "unit": "newtons",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
            },
        },
        action_space={
            "type": "array",
            "shape": [1],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": ["cart_force"],
        },
        metadata={
            "environment": "InvertedDoublePendulum-v5",
            "provider": "Gymnasium",
            "reward_threshold": 9100.0,
            "unhealthy_tip_height": 1.0,
            "official_model": "inverted_double_pendulum.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "healthy_reward": config.healthy_reward,
            "reset_noise_scale": config.reset_noise_scale,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _failure_return(config: InvertedDoublePendulumConfig) -> float:
    return -1_000.0 * max(1.0, config.healthy_reward / 10.0)


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _balanced_full_horizon(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.steps == _MAX_EPISODE_STEPS
        and record.transitions
        and record.transitions[-1].step.truncated
        and not record.transitions[-1].step.terminated
    )


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
                    "balanced_full_horizon": _balanced_full_horizon(record),
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
    if type(action) is not list or len(action) != 1:
        raise ValueError(
            "InvertedDoublePendulum trace Action is invalid"
        )
    value = action[0]
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not -1.0 <= value <= 1.0
    ):
        raise ValueError(
            "InvertedDoublePendulum trace Action is invalid"
        )
    return [value]


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError(
            "InvertedDoublePendulum trace observation is invalid"
        )
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError(
            "InvertedDoublePendulum trace observation is invalid"
        )
    traced: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(
                "InvertedDoublePendulum trace observation is invalid"
            )
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, float]:
    if type(metrics) is not dict:
        raise ValueError(
            "InvertedDoublePendulum trace metrics are invalid"
        )
    required = {
        "reward_survive",
        "distance_penalty",
        "velocity_penalty",
    }
    if set(metrics) != required:
        raise ValueError(
            "InvertedDoublePendulum trace metrics are invalid"
        )
    traced: dict[str, float] = {}
    for key in sorted(required):
        value = metrics[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(
                "InvertedDoublePendulum trace metrics are invalid"
            )
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


__all__ = ["InvertedDoublePendulumBenchmark"]
