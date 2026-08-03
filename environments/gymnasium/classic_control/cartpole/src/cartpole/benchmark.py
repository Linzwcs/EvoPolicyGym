"""A minimal reproducible CartPole Benchmark with public traces."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from typing import cast

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .environment import CartPoleEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-cartpole/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8

_SPEC = BenchmarkSpec(
    id="gymnasium/CartPole-v1/mean-return-v1",
    description=(
        "Balance the pole for 500 control steps of 0.02 seconds each. Action 0 "
        "applies 10 N left and Action 1 applies 10 N right. Every transition, "
        "including a terminating transition, rewards 1. The Episode terminates "
        "when absolute cart position exceeds 2.4 m or absolute pole angle exceeds "
        "12 degrees (0.20943951 rad), and truncates successfully after step 500. "
        "Maximize mean Episode return; return therefore equals survived steps."
    ),
    observation_space={
        "type": "vector",
        "policy_carrier": "list[float]",
        "source_dtype": "float32",
        "shape": [4],
        "components": [
            "cart_position",
            "cart_velocity",
            "pole_angle",
            "pole_angular_velocity",
        ],
        "component_meanings": {
            "cart_position": "Horizontal cart position in meters; termination limit is ±2.4.",
            "cart_velocity": "Horizontal cart velocity in meters per second; unbounded.",
            "pole_angle": (
                "Pole angle in radians from upright; positive tilts right and "
                "termination limit is ±0.20943951 (±12 degrees)."
            ),
            "pole_angular_velocity": (
                "Pole angular velocity in radians per second; positive rotates right."
            ),
        },
        "initial_sampling": "each component independently uniform in [-0.05, 0.05]",
    },
    action_space={
        "type": "discrete",
        "values": [0, 1],
        "component": "force_direction",
        "meaning": {
            "0": "push_left",
            "1": "push_right",
        },
    },
    metadata={
        "environment": "CartPole-v1",
        "provider": "Gymnasium",
        "reward_per_step": 1.0,
        "maximum_return": 500.0,
        "failure_return": 0.0,
    },
    environment_parameters={
        "cart_position_limit_meters": 2.4,
        "pole_angle_limit_radians": 12.0 * 3.141592653589793 / 180.0,
        "pole_angle_limit_degrees": 12.0,
        "force_magnitude_newtons": 10.0,
        "seconds_per_step": 0.02,
        "gravity_meters_per_second_squared": 9.8,
        "cart_mass_kilograms": 1.0,
        "pole_mass_kilograms": 0.1,
        "pole_half_length_meters": 0.5,
        "integrator": "explicit_euler",
        "initial_state_minimum": -0.05,
        "initial_state_maximum": 0.05,
        "reward_per_step_including_termination": 1.0,
        "time_limit": 500,
    },
    max_episode_steps=500,
    primary_metric="mean_return",
    score_direction="maximize",
)


class CartPoleBenchmark:
    """Mean CartPole return over deterministic Episode plans."""

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
        return CartPoleEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        failures = sum(record.policy_failure is not None for record in records)
        score = statistics.fmean(returns)
        mean_steps = statistics.fmean(record.steps for record in records)
        time_limit_successes = sum(
            bool(
                record.policy_failure is None
                and record.transitions
                and record.transitions[-1].step.truncated
                and not record.transitions[-1].step.terminated
            )
            for record in records
        )
        cart_limit_episodes = sum(
            "cart_position_limit" in _terminal_reason(record)
            for record in records
        )
        pole_limit_episodes = sum(
            "pole_angle_limit" in _terminal_reason(record)
            for record in records
        )
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "time_limit_successes": time_limit_successes,
                "cart_position_limit_episodes": cart_limit_episodes,
                "pole_angle_limit_episodes": pole_limit_episodes,
                "policy_failures": failures,
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
                    "return": record.total_reward,
                    "outcome": _episode_outcome(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int or transition.action not in {0, 1}:
                raise ValueError("CartPole trace Action is invalid")
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
                        "action_meaning": (
                            "push_left" if transition.action == 0 else "push_right"
                        ),
                        "reward": transition.step.reward,
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": transition.step.metrics,
                    }
                )
            )
            observation = next_observation
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _terminal_reason(record: EpisodeRecord) -> str:
    if not record.transitions:
        return ""
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        return ""
    reason = metrics.get("terminal_reason")
    return reason if type(reason) is str else ""


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    reason = _terminal_reason(record)
    if reason == "time_limit":
        return "balanced_to_time_limit"
    if reason:
        return reason
    return "incomplete"


def _trace_observation(observation: PolicyValue) -> list[float]:
    if type(observation) is not list or len(observation) != 4:
        raise ValueError("CartPole trace observation is invalid")
    if any(type(value) is not float for value in observation):
        raise ValueError("CartPole trace observation is invalid")
    return list(cast(list[float], observation))


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


__all__ = ["CartPoleBenchmark"]
