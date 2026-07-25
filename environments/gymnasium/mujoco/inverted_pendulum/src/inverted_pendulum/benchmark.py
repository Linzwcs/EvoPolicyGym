"""A parameterized InvertedPendulum-v5 Benchmark with public traces."""

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

from .config import InvertedPendulumConfig
from .environment import InvertedPendulumEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-inverted-pendulum/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 1_000
_FAILURE_RETURN = -1_000.0
_OBSERVATION_FIELDS = (
    "cart_position",
    "pole_angle",
    "cart_velocity",
    "pole_angular_velocity",
)


class InvertedPendulumBenchmark:
    """Mean survival return over deterministic Episode plans."""

    def __init__(
        self,
        config: InvertedPendulumConfig | None = None,
    ) -> None:
        if config is None:
            config = InvertedPendulumConfig()
        if type(config) is not InvertedPendulumConfig:
            raise TypeError("config must be InvertedPendulumConfig")
        self._config = config
        self._spec = _benchmark_spec(config)

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
        return InvertedPendulumEnvironment(
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
                else _FAILURE_RETURN
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
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: InvertedPendulumConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="gymnasium/InvertedPendulum-v5/mean-return-v1",
        description=(
            "Apply horizontal force to a MuJoCo cart to keep its pole upright. "
            "Each healthy step earns one point. Maximize mean Episode return."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "cart_position": {"type": "float", "unit": "meters"},
                "pole_angle": {"type": "float", "unit": "radians"},
                "cart_velocity": {
                    "type": "float",
                    "unit": "meters_per_second",
                },
                "pole_angular_velocity": {
                    "type": "float",
                    "unit": "radians_per_second",
                },
            },
        },
        action_space={
            "type": "array",
            "shape": [1],
            "items": {
                "type": "float",
                "minimum": -3.0,
                "maximum": 3.0,
            },
            "components": ["cart_force"],
        },
        metadata={
            "environment": "InvertedPendulum-v5",
            "provider": "Gymnasium",
            "reward_threshold": 950.0,
            "maximum_return": 1000.0,
            "healthy_angle_limit": 0.2,
            "official_model": "inverted_pendulum.xml",
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "reset_noise_scale": config.reset_noise_scale,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


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


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
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
                        else _FAILURE_RETURN
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
        raise ValueError("InvertedPendulum trace Action is invalid")
    value = action[0]
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not -3.0 <= value <= 3.0
    ):
        raise ValueError("InvertedPendulum trace Action is invalid")
    return [value]


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError(
            "InvertedPendulum trace observation is invalid"
        )
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError(
            "InvertedPendulum trace observation is invalid"
        )
    traced: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(
                "InvertedPendulum trace observation is invalid"
            )
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, float]:
    if type(metrics) is not dict or set(metrics) != {"reward_survive"}:
        raise ValueError("InvertedPendulum trace metrics are invalid")
    value = metrics["reward_survive"]
    if type(value) is not float or value not in {0.0, 1.0}:
        raise ValueError("InvertedPendulum trace metrics are invalid")
    return {"reward_survive": value}


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


__all__ = ["InvertedPendulumBenchmark"]
