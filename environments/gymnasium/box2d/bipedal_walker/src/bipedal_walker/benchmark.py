"""A parameterized BipedalWalker-v3 Benchmark with public traces."""

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

from .config import BipedalWalkerConfig
from .environment import BipedalWalkerEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-bipedal-walker/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 1_600
_FAILURE_RETURN = -1_000.0
_OBSERVATION_FIELDS = (
    "hull_angle",
    "hull_angular_velocity",
    "horizontal_velocity",
    "vertical_velocity",
    "left_hip_angle",
    "left_hip_angular_velocity",
    "left_knee_angle",
    "left_knee_angular_velocity",
    "left_foot_contact",
    "right_hip_angle",
    "right_hip_angular_velocity",
    "right_knee_angle",
    "right_knee_angular_velocity",
    "right_foot_contact",
    "lidar_ranges",
)
_CONTACT_FIELDS = frozenset(
    {"left_foot_contact", "right_foot_contact"}
)
_ACTION_COMPONENTS = (
    "left_hip",
    "left_knee",
    "right_hip",
    "right_knee",
)


class BipedalWalkerBenchmark:
    """Mean BipedalWalker return over deterministic Episode plans."""

    def __init__(self, config: BipedalWalkerConfig | None = None) -> None:
        if config is None:
            config = BipedalWalkerConfig()
        if type(config) is not BipedalWalkerConfig:
            raise TypeError("config must be BipedalWalkerConfig")
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
        return BipedalWalkerEnvironment(episode, config=self._config)

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
        courses = sum(_completed_course(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{courses} completed courses."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "completed_courses": courses,
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: BipedalWalkerConfig) -> BenchmarkSpec:
    scalar_fields: dict[str, PolicyValue] = {
        field: {
            "type": "boolean",
        }
        if field in _CONTACT_FIELDS
        else {
            "type": "float",
            "minimum": -5.0,
            "maximum": 5.0,
        }
        for field in _OBSERVATION_FIELDS[:-1]
    }
    scalar_fields["lidar_ranges"] = {
        "type": "array",
        "shape": [10],
        "items": {
            "type": "float",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "order": "downward_to_forward",
    }
    return BenchmarkSpec(
        id="gymnasium/BipedalWalker-v3/mean-return-v1",
        description=(
            "Control four hip and knee motors to walk across uneven terrain. "
            "Balance forward progress, upright posture, foot contacts, terrain "
            "sensing, and energy cost. Maximize mean Episode return."
        ),
        observation_space={
            "type": "object",
            "fields": scalar_fields,
            "notes": (
                "All continuous body and joint values are normalized by "
                "Gymnasium. Lidar values are fractions of maximum range."
            ),
        },
        action_space={
            "type": "array",
            "shape": [4],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": list(_ACTION_COMPONENTS),
            "notes": (
                "Sign selects motor direction and magnitude selects torque."
            ),
        },
        metadata={
            "environment": "BipedalWalker-v3",
            "provider": "Gymnasium",
            "reward_threshold": 300.0,
            "fall_terminal_reward": -100.0,
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={"hardcore": config.hardcore},
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


def _completed_course(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward > -100.0
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
                    "completed_course": _completed_course(record),
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
    if type(action) is not list or len(action) != 4:
        raise ValueError("BipedalWalker trace Action is invalid")
    traced: list[float] = []
    for value in action:
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValueError("BipedalWalker trace Action is invalid")
        traced.append(value)
    return traced


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise ValueError("BipedalWalker trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("BipedalWalker trace observation is invalid")
    traced: dict[str, PolicyValue] = {}
    for key in _OBSERVATION_FIELDS[:-1]:
        value = observation[key]
        if key in _CONTACT_FIELDS:
            if type(value) is not bool:
                raise ValueError(
                    "BipedalWalker trace observation is invalid"
                )
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("BipedalWalker trace observation is invalid")
        traced[key] = value
    lidar = observation["lidar_ranges"]
    if (
        type(lidar) is not list
        or len(lidar) != 10
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in lidar
        )
    ):
        raise ValueError("BipedalWalker trace observation is invalid")
    traced["lidar_ranges"] = list(lidar)
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


__all__ = ["BipedalWalkerBenchmark"]
