"""A parameterized Blackjack-v1 Benchmark with public traces."""

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

from .config import BlackjackConfig
from .environment import BlackjackEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-blackjack/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 32
_MAX_EPISODE_STEPS = 32
_FAILURE_RETURN = -1.0


class BlackjackBenchmark:
    """Mean Blackjack reward over deterministic Episode plans."""

    def __init__(self, config: BlackjackConfig | None = None) -> None:
        if config is None:
            config = BlackjackConfig()
        if type(config) is not BlackjackConfig:
            raise TypeError("config must be BlackjackConfig")
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
        return BlackjackEnvironment(episode, config=self._config)

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
        completed_returns = tuple(
            record.total_reward
            for record in records
            if record.policy_failure is None
        )
        score = statistics.fmean(returns)
        wins = sum(value > 0.0 for value in completed_returns)
        draws = sum(value == 0.0 for value in completed_returns)
        losses = sum(value < 0.0 for value in completed_returns)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean reward {score:.3f} across {len(records)} Episodes; "
                    f"{wins} wins, {draws} draws, and {losses} losses."
                ),
                "mean_reward": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: BlackjackConfig) -> BenchmarkSpec:
    payout = 1.0 if config.sab or not config.natural else 1.5
    return BenchmarkSpec(
        id="gymnasium/Blackjack-v1/mean-reward-v1",
        description=(
            "Play one hand of infinite-deck Blackjack. Choose 0 to stick or "
            "1 to hit. Maximize mean terminal reward over many seeded hands."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "player_sum": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 31,
                },
                "dealer_showing": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
                "usable_ace": {
                    "type": "boolean",
                },
            },
        },
        action_space={
            "type": "discrete",
            "values": [0, 1],
            "meaning": {
                "0": "stick",
                "1": "hit",
            },
        },
        metadata={
            "environment": "Blackjack-v1",
            "provider": "Gymnasium",
            "deck": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10],
            "dealer_sticks_at": 17,
            "reward_schedule": {
                "win": 1.0,
                "natural_win": payout,
                "draw": 0.0,
                "loss": -1.0,
            },
            "failure_return": _FAILURE_RETURN,
            "natural_ignored_when_sab": True,
        },
        environment_parameters={
            "natural": config.natural,
            "sab": config.sab,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_reward",
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


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        episode_return = (
            record.total_reward
            if record.policy_failure is None
            else _FAILURE_RETURN
        )
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
                    "return": episode_return,
                    "outcome": _outcome(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int:
                raise ValueError("Blackjack trace Action is invalid")
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


def _outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if record.total_reward > 0.0:
        return "win"
    if record.total_reward < 0.0:
        return "loss"
    return "draw"


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise ValueError("Blackjack trace observation is invalid")
    if set(observation) != {
        "player_sum",
        "dealer_showing",
        "usable_ace",
    }:
        raise ValueError("Blackjack trace observation is invalid")
    if (
        type(observation["player_sum"]) is not int
        or not 0 <= observation["player_sum"] <= 31
    ):
        raise ValueError("Blackjack trace observation is invalid")
    if (
        type(observation["dealer_showing"]) is not int
        or not 1 <= observation["dealer_showing"] <= 10
    ):
        raise ValueError("Blackjack trace observation is invalid")
    if type(observation["usable_ace"]) is not bool:
        raise ValueError("Blackjack trace observation is invalid")
    return {
        "player_sum": observation["player_sum"],
        "dealer_showing": observation["dealer_showing"],
        "usable_ace": observation["usable_ace"],
    }


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


__all__ = ["BlackjackBenchmark"]
