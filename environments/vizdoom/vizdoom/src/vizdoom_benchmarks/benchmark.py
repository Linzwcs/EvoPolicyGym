"""Bundled ViZDoom scenarios with bounded public traces."""

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

from .config import ViZDoomConfig
from .environment import ViZDoomEnvironment

_SEED_DOMAIN = b"evopolicygym-vizdoom/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4


class ViZDoomBenchmark:
    """Mean return for one fixed bundled ViZDoom scenario."""

    def __init__(self, config: ViZDoomConfig | None = None) -> None:
        if config is None:
            config = ViZDoomConfig()
        if type(config) is not ViZDoomConfig:
            raise TypeError("config must be ViZDoomConfig")
        self._config = config
        self._spec = _spec(config)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self, split: str, *, seed: int, count: int
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
        return ViZDoomEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        floor = -float(self._config.max_episode_steps)
        returns = tuple(
            r.total_reward if r.policy_failure is None else floor
            for r in records
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
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "failure_return": floor,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace(traced),),
        )


def _spec(config: ViZDoomConfig) -> BenchmarkSpec:
    fields: dict[str, PolicyValue] = {
        "screen": {
            "type": "tensor",
            "dtype": "uint8",
            "shape": [240, 320, 3],
        }
    }
    if config.audio:
        fields["audio"] = {
            "type": "tensor",
            "dtype": "int16",
            "shape": [1260, 2],
        }
    if config.notifications:
        fields["notifications"] = {"type": "string"}
    if config.game_variables:
        fields["gamevariables"] = {
            "type": "tensor",
            "dtype": "float32",
            "shape": [config.game_variables],
        }
    action: PolicyValue
    if config.hybrid_action:
        action = {
            "type": "object",
            "fields": {
                "binary": {
                    "type": "discrete",
                    "values": list(range(config.action_size)),
                },
                "continuous": {
                    "type": "array",
                    "shape": [3],
                    "items": {"type": "finite_float32"},
                },
            },
        }
    else:
        action = {
            "type": "discrete",
            "values": list(range(config.action_size)),
        }
    return BenchmarkSpec(
        id=f"vizdoom/{config.environment_id}/mean-return-v1",
        description=(
            f"Control the agent in ViZDoom's {config.profile} scenario. "
            "Maximize mean Episode return."
        ),
        observation_space={"type": "object", "fields": fields},
        action_space=action,
        metadata={
            "environment": config.environment_id,
            "provider": "ViZDoom",
            "upstream_version": "1.3.0",
            "failure_return": -float(config.max_episode_steps),
        },
        environment_parameters={
            "profile": config.profile,
            "action_size": config.action_size,
            "hybrid_action": config.hybrid_action,
            "rgb_resolution": [320, 240],
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


__all__ = ["ViZDoomBenchmark"]
