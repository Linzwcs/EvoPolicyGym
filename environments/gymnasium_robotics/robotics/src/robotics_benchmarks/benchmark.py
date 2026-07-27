"""Single-agent Gymnasium-Robotics profiles and public feedback."""

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

from .config import RoboticsConfig
from .environment import RoboticsEnvironment

_SEED_DOMAIN = b"evopolicygym-gymnasium-robotics/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_KITCHEN_GOALS: dict[str, PolicyValue] = {
    "type": "object",
    "fields": {
        "bottom burner": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "top burner": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "light switch": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "slide cabinet": {"type": "tensor", "dtype": "float64", "shape": [1]},
        "hinge cabinet": {"type": "tensor", "dtype": "float64", "shape": [2]},
        "microwave": {"type": "tensor", "dtype": "float64", "shape": [1]},
        "kettle": {"type": "tensor", "dtype": "float64", "shape": [7]},
    },
}


class RoboticsBenchmark:
    """Success rate for one fixed Gymnasium-Robotics profile."""

    def __init__(self, config: RoboticsConfig | None = None) -> None:
        if config is None:
            config = RoboticsConfig()
        if type(config) is not RoboticsConfig:
            raise TypeError("config must be RoboticsConfig")
        self._config = config
        self._spec = _spec(config)

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
            EpisodeSpec(environment_seed=_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return RoboticsEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        score = successes / len(records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Solved {successes}/{len(records)} "
                    f"{self._config.profile} Episodes "
                    f"({score:.3f} success rate)."
                ),
                "success_rate": score,
                "mean_return": statistics.fmean(
                    r.total_reward if r.policy_failure is None else 0.0
                    for r in records
                ),
                "mean_steps": statistics.fmean(r.steps for r in records),
                "episodes": len(records),
                "successful_episodes": successes,
                "terminated_episodes": sum(_terminated(r) for r in records),
                "truncated_episodes": sum(_truncated(r) for r in records),
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace(traced),),
        )


def _spec(config: RoboticsConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=f"gymnasium-robotics/{config.environment_id}/success-rate-v1",
        description=(
            f"Complete Gymnasium-Robotics' {config.profile} task. "
            "Maximize the fraction of Episodes that reach the public task "
            "success condition."
        ),
        observation_space=_observation_space(config),
        action_space={
            "type": "array",
            "shape": [config.action_size],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
        },
        metadata={
            "environment": config.environment_id,
            "provider": "Gymnasium-Robotics",
            "upstream_version": "1.4.2",
            "reward_mode": "upstream default",
        },
        environment_parameters={
            "profile": config.profile,
            "family": config.family,
            "continuous_actions": True,
            "action_size": config.action_size,
            "action_dtype": config.action_dtype,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="success_rate",
        score_direction="maximize",
    )


def _observation_space(config: RoboticsConfig) -> PolicyValue:
    state: PolicyValue = {
        "type": "tensor",
        "dtype": "float64",
        "shape": [config.observation_size],
    }
    if config.goal_size is None:
        return state
    goal: PolicyValue
    if config.goal_size == -1:
        goal = _KITCHEN_GOALS
    else:
        goal = {
            "type": "tensor",
            "dtype": "float64",
            "shape": [config.goal_size],
        }
    return {
        "type": "object",
        "fields": {
            "observation": state,
            "achieved_goal": goal,
            "desired_goal": goal,
        },
    }


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _success(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and any(
            type(item.step.metrics) is dict
            and item.step.metrics.get("success") is True
            for item in record.transitions
        )
    )


def _terminated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
    )


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _trace(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        lines.append(
            _json(
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
                    "success": _success(record),
                    "failure": record.policy_failure,
                }
            )
        )
        for step_index, transition in enumerate(record.transitions):
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "action": transition.action,
                        "reward": transition.step.reward,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": transition.step.metrics,
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _json(document: dict[str, object]) -> bytes:
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


__all__ = ["RoboticsBenchmark"]

