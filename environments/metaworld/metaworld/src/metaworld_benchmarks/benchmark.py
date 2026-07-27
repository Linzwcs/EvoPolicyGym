"""MetaWorld MT collections with deterministic Episode plans."""

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

from .config import MetaWorldConfig
from .environment import MetaWorldEnvironment

_SEED_DOMAIN = b"evopolicygym-metaworld/episode-seed/v1\0"
_TASK_DOMAIN = b"evopolicygym-metaworld/task-offset/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_EPISODE_STEPS = 500


class MetaWorldBenchmark:
    """Success rate for one fixed MetaWorld MT task collection."""

    def __init__(self, config: MetaWorldConfig | None = None) -> None:
        if config is None:
            config = MetaWorldConfig()
        if type(config) is not MetaWorldConfig:
            raise TypeError("config must be MetaWorldConfig")
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
        task_count = len(self._config.task_names)
        offset = _task_offset(split, seed, task_count)
        return tuple(
            EpisodeSpec(
                environment_seed=_seed(split, seed, index),
                scenario=(
                    None
                    if task_count == 1
                    else {"task_index": (offset + index) % task_count}
                ),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return MetaWorldEnvironment(episode, config=self._config)

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
                    f"Solved {successes}/{len(records)} MetaWorld Episodes "
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


def _spec(config: MetaWorldConfig) -> BenchmarkSpec:
    task_count = len(config.task_names)
    state: PolicyValue = {
        "type": "tensor",
        "dtype": "float64",
        "shape": [39],
        "goal_observable": True,
    }
    observation: PolicyValue
    if task_count == 1:
        observation = state
    else:
        observation = {
            "type": "object",
            "fields": {
                "state": state,
                "task": {
                    "type": "tensor",
                    "dtype": "bool",
                    "shape": [task_count],
                    "encoding": "one_hot",
                },
            },
        }
    benchmark_name = (
        f"MT1/{config.profile}"
        if config.collection_name == "mt1"
        else config.collection_name.upper()
    )
    return BenchmarkSpec(
        id=f"metaworld/{benchmark_name}/success-rate-v1",
        description=(
            f"Complete tasks from MetaWorld's {benchmark_name} collection. "
            "Maximize the fraction of Episodes reaching the public success "
            "condition."
        ),
        observation_space=observation,
        action_space={
            "type": "array",
            "shape": [4],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "meaning": [
                "end_effector_delta_x",
                "end_effector_delta_y",
                "end_effector_delta_z",
                "gripper_effort",
            ],
        },
        metadata={
            "environment": "Meta-World/MT1",
            "provider": "MetaWorld",
            "upstream_version": "3.1.1",
            "reward_function_version": "v2",
        },
        environment_parameters={
            "profile": config.profile,
            "collection": config.collection_name,
            "task_count": task_count,
            "task_index_to_environment": list(config.task_names),
            "goal_observable": True,
            "continuous_actions": True,
            "action_size": 4,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="success_rate",
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


def _task_offset(split: str, seed: int, task_count: int) -> int:
    if task_count == 1:
        return 0
    digest = hashlib.sha256()
    digest.update(_TASK_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big") % task_count


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


__all__ = ["MetaWorldBenchmark"]

