"""A reproducible Continuous Mountain Car Benchmark with bounded traces."""

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

from .environment import MountainCarContinuousEnvironment

_EPISODE_SEED_DOMAIN = (
    b"evopolicygym-mountain-car-continuous/episode-seed/v1\0"
)
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -100.0
_GOAL_POSITION = 0.45
_ACTION_COST_COEFFICIENT = 0.1

_SPEC = BenchmarkSpec(
    id="gymnasium/MountainCarContinuous-v0/mean-return-v1",
    description=(
        "Drive an underpowered car up the right hill. Return one finite Python "
        "float in [-1,1]: negative force points left and positive force points "
        "right. Velocity updates by force*0.0015-0.0025*cos(3*position), is "
        "clipped to [-0.07,0.07], and then advances position within [-1.2,0.6]. "
        "The left wall resets leftward velocity to zero. Reach position >=0.45 "
        "with velocity >=0 within 999 steps. Each reward is -0.1*force^2 plus "
        "a 100-point bonus on success; maximize mean return by succeeding with "
        "low control effort."
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
                    "begins at position 0.45 on the right hill."
                ),
            },
            "velocity": {
                "type": "float",
                "minimum": -0.07,
                "maximum": 0.07,
                "unit": "track_position_per_step",
                "meaning": (
                    "Signed clipped velocity; negative moves left and positive moves right."
                ),
            },
        },
    },
    action_space={
        "type": "float",
        "minimum": -1.0,
        "maximum": 1.0,
        "policy_carrier": "float",
        "meaning": "signed_force: negative=left, zero=no engine force, positive=right",
    },
    metadata={
        "environment": "MountainCarContinuous-v0",
        "provider": "Gymnasium",
        "failure_return": _FAILURE_RETURN,
        "reward_threshold": 90.0,
    },
    environment_parameters={
        "engine_velocity_increment_per_unit_force": 0.0015,
        "gravity_velocity_increment_formula": "-0.0025*cos(3*position)",
        "velocity_update_formula": (
            "clip(velocity+force*0.0015-0.0025*cos(3*position),-0.07,0.07)"
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
        "action_cost_coefficient": _ACTION_COST_COEFFICIENT,
        "success_bonus": 100.0,
        "reward_formula": "success_bonus_if_terminated-0.1*force^2",
        "time_limit": 999,
    },
    max_episode_steps=999,
    primary_metric="mean_return",
    score_direction="maximize",
)


class MountainCarContinuousBenchmark:
    """Mean Continuous Mountain Car return over deterministic plans."""

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
        return MountainCarContinuousEnvironment(episode)

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
        episode_control_costs = tuple(_episode_control_cost(record) for record in records)
        episode_mean_absolute_forces = tuple(
            _episode_mean_absolute_force(record) for record in records
        )
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
                "mean_episode_control_cost": statistics.fmean(episode_control_costs),
                "mean_episode_absolute_force": statistics.fmean(
                    episode_mean_absolute_forces
                ),
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
        minimum_position, maximum_position = _episode_position_extrema(record)
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
                    "minimum_position": minimum_position,
                    "maximum_position": maximum_position,
                    "closest_goal_position_gap": max(
                        _GOAL_POSITION - maximum_position,
                        0.0,
                    ),
                    "control_cost": _episode_control_cost(record),
                    "mean_absolute_force": _episode_mean_absolute_force(record),
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
                        "action_meaning": "signed_force_left_negative_right_positive",
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
        raise ValueError(
            "Continuous Mountain Car trace observation is invalid"
        )
    expected = {"position", "velocity"}
    if set(observation) != expected:
        raise ValueError(
            "Continuous Mountain Car trace observation is invalid"
        )
    traced: dict[str, float] = {}
    for key in ("position", "velocity"):
        value = observation[key]
        if type(value) is not float:
            raise ValueError(
                "Continuous Mountain Car trace observation is invalid"
            )
        traced[key] = value
    return traced


def _trace_action(action: PolicyValue) -> float:
    if type(action) is not float or not math.isfinite(action) or not -1.0 <= action <= 1.0:
        raise ValueError("Continuous Mountain Car trace Action is invalid")
    return action


def _episode_position_extrema(record: EpisodeRecord) -> tuple[float, float]:
    observations = (
        record.initial_observation,
        *(transition.step.observation for transition in record.transitions),
    )
    positions = tuple(_trace_observation(observation)["position"] for observation in observations)
    return min(positions), max(positions)


def _episode_control_cost(record: EpisodeRecord) -> float:
    return math.fsum(
        _ACTION_COST_COEFFICIENT * _trace_action(transition.action) ** 2
        for transition in record.transitions
    )


def _episode_mean_absolute_force(record: EpisodeRecord) -> float:
    if not record.transitions:
        return 0.0
    return statistics.fmean(
        abs(_trace_action(transition.action)) for transition in record.transitions
    )


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


__all__ = ["MountainCarContinuousBenchmark"]
