"""Canonical single-Policy Jumanji profiles with bounded public feedback."""

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

from .config import JumanjiConfig
from .environment import JumanjiEnvironment

_SEED_DOMAIN = b"evopolicygym-jumanji/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4


class JumanjiBenchmark:
    """Mean return for one fixed Jumanji profile."""

    def __init__(self, config: JumanjiConfig | None = None) -> None:
        if config is None:
            config = JumanjiConfig()
        if type(config) is not JumanjiConfig:
            raise TypeError("config must be JumanjiConfig")
        self._config = config
        self._spec = _spec(config)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(self, split: str, *, seed: int, count: int) -> Sequence[EpisodeSpec]:
        if type(split) is not str or split not in _SPLITS:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(EpisodeSpec(environment_seed=_seed(split, seed, index)) for index in range(count))

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return JumanjiEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        failure_return = -float(self._config.max_episode_steps)
        returns = tuple(
            record.total_reward if record.policy_failure is None else failure_return
            for record in records
        )
        score = statistics.fmean(returns)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} "
                    f"{self._config.profile} Episodes."
                ),
                "mean_return": score,
                "mean_steps": statistics.fmean(record.steps for record in records),
                "episodes": len(records),
                "completed_episodes": sum(_completed(record) for record in records),
                "policy_failures": sum(record.policy_failure is not None for record in records),
                "failure_return": failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace(traced),),
        )


def _spec(config: JumanjiConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.action_kind == "discrete":
        size = config.action_num_values[0]
        action_space = {
            "type": "discrete",
            "values": list(range(size)),
            "masked": config.has_action_mask,
        }
    else:
        action_space = {
            "type": "multi_discrete",
            "shape": [len(config.action_num_values)],
            "num_values": list(config.action_num_values),
            "masked": config.has_action_mask,
        }
    return BenchmarkSpec(
        id=f"jumanji/{config.environment_id}/mean-return-v1",
        description=(
            f"Solve Jumanji's {config.environment_id} {config.category} task. "
            "Maximize mean upstream return across independently seeded instances."
        ),
        observation_space={
            "type": "object",
            "encoding": "named fields with canonical TensorValue leaves",
            "includes_action_mask": config.has_action_mask,
        },
        action_space=action_space,
        metadata={
            "environment": config.environment_id,
            "provider": "Jumanji",
            "upstream_version": "1.1.1",
            "failure_return": -float(config.max_episode_steps),
        },
        environment_parameters={
            "profile": config.profile,
            "category": config.category,
            "environment_id": config.environment_id,
            "action_kind": config.action_kind,
            "action_num_values": list(config.action_num_values),
            "has_action_mask": config.has_action_mask,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _completed(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.done
    )


def _trace(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": "completed" if record.policy_failure is None else "policy_failed",
                    "steps": record.steps,
                    "return": record.total_reward,
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


__all__ = ["JumanjiBenchmark"]
