"""A reproducible Pendulum Benchmark with bounded public traces."""

from __future__ import annotations

import hashlib
import json
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

from .environment import PendulumEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-pendulum/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -3300.0

_SPEC = BenchmarkSpec(
    id="gymnasium/Pendulum-v1/mean-return-v1",
    description=(
        "Control an inverted pendulum for 200 steps. Return one float from "
        "-2.0 to 2.0 as torque. Maximize mean Episode return by keeping the "
        "pendulum upright with low angular velocity and control effort."
    ),
    observation_space={
        "type": "object",
        "fields": {
            "cos_theta": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "sin_theta": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "theta_angular_velocity": {
                "type": "float",
                "minimum": -8.0,
                "maximum": 8.0,
            },
        },
    },
    action_space={
        "type": "float",
        "minimum": -2.0,
        "maximum": 2.0,
        "meaning": "torque",
    },
    metadata={
        "environment": "Pendulum-v1",
        "provider": "Gymnasium",
        "angle_cost_coefficient": 1.0,
        "angular_velocity_cost_coefficient": 0.1,
        "torque_cost_coefficient": 0.001,
        "minimum_step_reward": -16.2736044,
        "maximum_step_reward": 0.0,
        "failure_return": _FAILURE_RETURN,
        "reward_threshold": -200.0,
    },
    max_episode_steps=200,
    primary_metric="mean_return",
    score_direction="maximize",
)


class PendulumBenchmark:
    """Mean Pendulum return over deterministic Episode plans."""

    @property
    def spec(self) -> BenchmarkSpec:
        return _SPEC

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
        return PendulumEnvironment(episode)

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
        failures = sum(record.policy_failure is not None for record in records)
        completed = len(records) - failures
        truncated = sum(_truncated(record) for record in records)
        score = statistics.fmean(returns)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{failures} Policy failures."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "completed_episodes": completed,
                "truncated_episodes": truncated,
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
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
                    "ended_by_truncation": _truncated(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not float:
                raise ValueError("Pendulum trace Action is invalid")
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
                        "action": transition.action,
                        "reward": transition.step.reward,
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


def _trace_observation(observation: PolicyValue) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError("Pendulum trace observation is invalid")
    expected = {
        "cos_theta",
        "sin_theta",
        "theta_angular_velocity",
    }
    if set(observation) != expected:
        raise ValueError("Pendulum trace observation is invalid")
    traced: dict[str, float] = {}
    for key in (
        "cos_theta",
        "sin_theta",
        "theta_angular_velocity",
    ):
        value = observation[key]
        if type(value) is not float:
            raise ValueError("Pendulum trace observation is invalid")
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


__all__ = ["PendulumBenchmark"]
