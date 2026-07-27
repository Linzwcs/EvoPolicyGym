"""The canonical HighwayEnv profiles with bounded public feedback."""

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

from .config import HighwayConfig
from .environment import HighwayEnvironment

_SEED_DOMAIN = b"evopolicygym-highway-env/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_DISCRETE_ACTIONS: dict[str, PolicyValue] = {
    "0": "lane_left_or_slower",
    "1": "idle",
    "2": "lane_right_or_faster",
    "3": "faster",
    "4": "slower",
}


class HighwayBenchmark:
    """Mean return for one fixed HighwayEnv profile."""

    def __init__(self, config: HighwayConfig | None = None) -> None:
        if config is None:
            config = HighwayConfig()
        if type(config) is not HighwayConfig:
            raise TypeError("config must be HighwayConfig")
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
        return HighwayEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        failure_return = -float(self._config.max_episode_steps)
        returns = tuple(
            record.total_reward
            if record.policy_failure is None
            else failure_return
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
                "mean_steps": statistics.fmean(r.steps for r in records),
                "episodes": len(records),
                "terminated_episodes": sum(_terminated(r) for r in records),
                "truncated_episodes": sum(_truncated(r) for r in records),
                "crashed_episodes": sum(_reached(r, "crashed") for r in records),
                "successful_episodes": sum(
                    _reached(r, "is_success") for r in records
                ),
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "failure_return": failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace(traced),),
        )


def _spec(config: HighwayConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.continuous:
        action_space = {
            "type": "array",
            "shape": [config.action_size],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "meaning": (
                ["acceleration", "steering"]
                if config.action_size == 2
                else ["steering"]
            ),
        }
    else:
        action_space = {
            "type": "discrete",
            "values": list(range(config.action_size)),
            "meaning": {
                key: value
                for key, value in _DISCRETE_ACTIONS.items()
                if int(key) < config.action_size
            },
            "notes": (
                "For longitudinal-only profiles, actions 0/1/2 mean "
                "slower/idle/faster. Otherwise they mean "
                "lane-left/idle/lane-right and actions 3/4 change speed."
            ),
        }
    return BenchmarkSpec(
        id=f"highway-env/{config.environment_id}/mean-return-v1",
        description=(
            f"Control the ego vehicle in HighwayEnv's {config.profile} task. "
            "Maximize mean return while following the selected task's "
            "driving objective."
        ),
        observation_space={
            "type": "tensor_or_object",
            "encoding": "canonical TensorValue leaves",
            "upstream_observation": config.observation_kind,
        },
        action_space=action_space,
        metadata={
            "environment": config.environment_id,
            "provider": "HighwayEnv",
            "upstream_version": "1.12",
            "failure_return": -float(config.max_episode_steps),
        },
        environment_parameters={
            "profile": config.profile,
            "observation_kind": config.observation_kind,
            "continuous_actions": config.continuous,
            "action_size": config.action_size,
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


def _reached(record: EpisodeRecord, name: str) -> bool:
    return any(
        type(item.step.metrics) is dict
        and item.step.metrics.get(name) is True
        for item in record.transitions
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


__all__ = ["HighwayBenchmark"]

