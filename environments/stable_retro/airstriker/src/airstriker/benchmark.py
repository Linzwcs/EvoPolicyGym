"""Redistributable Stable-Retro Airstriker Benchmark."""

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

from .config import AirstrikerConfig
from .environment import AirstrikerEnvironment

_SEED_DOMAIN = b"evopolicygym-stable-retro-airstriker/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_EPISODE_STEPS = 18_000
_MAX_TRACED_EPISODES = 4


class AirstrikerBenchmark:
    """Mean score delta on Stable-Retro's redistributable Airstriker game."""

    def __init__(self, config: AirstrikerConfig | None = None) -> None:
        if config is None:
            config = AirstrikerConfig()
        if type(config) is not AirstrikerConfig:
            raise TypeError("config must be AirstrikerConfig")
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
        return AirstrikerEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        floor = -float(_MAX_EPISODE_STEPS)
        returns = tuple(
            record.total_reward
            if record.policy_failure is None
            else floor
            for record in records
        )
        score = statistics.fmean(returns)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean Airstriker score delta {score:.3f} across "
                    f"{len(records)} Episodes."
                ),
                "mean_score_delta": score,
                "mean_steps": statistics.fmean(r.steps for r in records),
                "episodes": len(records),
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "failure_score": floor,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace(traced),),
        )


def _spec(config: AirstrikerConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="stable-retro/Airstriker-Genesis-v0/mean-score-delta-v1",
        description=(
            "Play the bundled Airstriker Level 1 from RGB observations using "
            "Stable-Retro's restricted discrete controller actions. Maximize "
            "mean score delta."
        ),
        observation_space={
            "type": "tensor",
            "dtype": "uint8",
            "shape": [224, 320, 3],
            "color_space": "RGB",
        },
        action_space={
            "type": "discrete",
            "start": 0,
            "count": 126,
            "controller_buttons": [
                "B",
                "A",
                "MODE",
                "START",
                "UP",
                "DOWN",
                "LEFT",
                "RIGHT",
                "C",
                "Y",
                "X",
                "Z",
            ],
            "restricted_actions": "DISCRETE",
        },
        metadata={
            "environment": config.game,
            "provider": "Stable-Retro",
            "upstream_version": "1.0.1",
            "failure_score": -float(_MAX_EPISODE_STEPS),
        },
        environment_parameters={
            "game": config.game,
            "state": config.state,
            "restricted_actions": "DISCRETE",
            "max_emulator_frames": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_score_delta",
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
                    "score_delta": record.total_reward,
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


__all__ = ["AirstrikerBenchmark"]
