"""A reproducible Mountain Car Benchmark with bounded public traces."""

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

from .environment import MountainCarEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-mountain-car/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -200.0
_ACTION_MEANINGS = ("accelerate_left", "coast", "accelerate_right")
_GOAL_POSITION = 0.5

_SPEC = BenchmarkSpec(
    id="gymnasium/MountainCar-v0/mean-return-v1",
    description=(
        "Drive an underpowered car up the right hill. On each step, action "
        "0/1/2 adds -0.001/0/+0.001 to velocity before gravity adds "
        "-0.0025*cos(3*position); velocity is clipped to [-0.07, 0.07], "
        "then position is advanced and clipped to [-1.2, 0.6]. Hitting the "
        "left wall at -1.2 while moving left resets velocity to zero. Reach "
        "position >= 0.5 with velocity >= 0 within 200 steps. Every transition, "
        "including the successful one, rewards -1, so return equals negative "
        "Episode length and less-negative return means faster success."
    ),
    observation_space={
        "type": "object",
        "policy_carrier": "dict[str, float]",
        "source_dtype": "float32",
        "fields": {
            "position": {
                "type": "float",
                "minimum": -1.2,
                "maximum": 0.6,
                "unit": "track_position",
                "meaning": (
                    "Car position along the one-dimensional track; the goal "
                    "begins at position 0.5 on the right hill."
                ),
            },
            "velocity": {
                "type": "float",
                "minimum": -0.07,
                "maximum": 0.07,
                "unit": "track_position_per_step",
                "meaning": (
                    "Signed car velocity after clipping; negative moves left "
                    "and positive moves right."
                ),
            },
        },
    },
    action_space={
        "type": "discrete",
        "values": [0, 1, 2],
        "component": "engine_acceleration_direction",
        "meaning": {
            "0": "accelerate_left",
            "1": "coast_without_engine_acceleration",
            "2": "accelerate_right",
        },
    },
    metadata={
        "environment": "MountainCar-v0",
        "provider": "Gymnasium",
        "reward_per_transition": -1.0,
        "failure_return": _FAILURE_RETURN,
        "reward_threshold": -110.0,
    },
    environment_parameters={
        "engine_velocity_increment_per_action_unit": 0.001,
        "gravity_velocity_increment_formula": "-0.0025*cos(3*position)",
        "velocity_update_formula": (
            "clip(velocity+(action-1)*0.001-0.0025*cos(3*position),-0.07,0.07)"
        ),
        "position_update_formula": "clip(position+updated_velocity,-1.2,0.6)",
        "minimum_position": -1.2,
        "maximum_position": 0.6,
        "maximum_absolute_velocity": 0.07,
        "left_wall_velocity_reset": 0.0,
        "terrain_height_formula": "0.45*sin(3*position)+0.55",
        "initial_position_minimum": -0.6,
        "initial_position_maximum": -0.4,
        "initial_velocity": 0.0,
        "goal_position_minimum": _GOAL_POSITION,
        "goal_velocity_minimum": 0.0,
        "reward_per_transition": -1.0,
        "time_limit": 200,
    },
    max_episode_steps=200,
    primary_metric="mean_return",
    score_direction="maximize",
)


class MountainCarBenchmark:
    """Mean Mountain Car return over deterministic Episode plans."""

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
        return MountainCarEnvironment(episode)

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
        successes = sum(_successful(record) for record in records)
        time_limits = sum(
            bool(
                record.policy_failure is None
                and record.transitions
                and record.transitions[-1].step.truncated
                and not record.transitions[-1].step.terminated
            )
            for record in records
        )
        position_extrema = tuple(_episode_position_extrema(record) for record in records)
        episode_minimum_positions = tuple(extrema[0] for extrema in position_extrema)
        episode_maximum_positions = tuple(extrema[1] for extrema in position_extrema)
        score = statistics.fmean(returns)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{successes} reached the goal."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "successful_episodes": successes,
                "time_limit_episodes": time_limits,
                "mean_episode_minimum_position": statistics.fmean(
                    episode_minimum_positions
                ),
                "mean_episode_maximum_position": statistics.fmean(
                    episode_maximum_positions
                ),
                "maximum_position_reached": max(episode_maximum_positions),
                "mean_closest_goal_position_gap": statistics.fmean(
                    max(_GOAL_POSITION - position, 0.0)
                    for position in episode_maximum_positions
                ),
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


def _successful(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
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
                    "reached_goal": _successful(record),
                    "outcome": _episode_outcome(record),
                    "minimum_position": _episode_position_extrema(record)[0],
                    "maximum_position": _episode_position_extrema(record)[1],
                    "closest_goal_position_gap": max(
                        _GOAL_POSITION - _episode_position_extrema(record)[1],
                        0.0,
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int or not 0 <= transition.action < 3:
                raise ValueError("Mountain Car trace Action is invalid")
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
                        "action_meaning": _ACTION_MEANINGS[transition.action],
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


def _trace_observation(observation: PolicyValue) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError("Mountain Car trace observation is invalid")
    expected = {"position", "velocity"}
    if set(observation) != expected:
        raise ValueError("Mountain Car trace observation is invalid")
    traced: dict[str, float] = {}
    for key in ("position", "velocity"):
        value = observation[key]
        if type(value) is not float:
            raise ValueError("Mountain Car trace observation is invalid")
        traced[key] = value
    return traced


def _episode_position_extrema(record: EpisodeRecord) -> tuple[float, float]:
    observations = (
        record.initial_observation,
        *(transition.step.observation for transition in record.transitions),
    )
    positions = tuple(_trace_observation(observation)["position"] for observation in observations)
    return min(positions), max(positions)


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if _successful(record):
        return "goal_reached"
    if record.transitions and record.transitions[-1].step.truncated:
        return "time_limit"
    return "incomplete"


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


__all__ = ["MountainCarBenchmark"]
