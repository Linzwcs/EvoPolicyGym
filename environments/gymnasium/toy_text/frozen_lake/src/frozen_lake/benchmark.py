"""A parameterized FrozenLake Benchmark with bounded public traces."""

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

from .config import FrozenLakeConfig
from .environment import FrozenLakeEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-frozen-lake/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8


class FrozenLakeBenchmark:
    """Goal-reaching rate over configured standard FrozenLake maps."""

    def __init__(self, config: FrozenLakeConfig | None = None) -> None:
        if config is None:
            config = FrozenLakeConfig()
        if type(config) is not FrozenLakeConfig:
            raise TypeError("config must be FrozenLakeConfig")
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
        return FrozenLakeEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        successes = sum(_successful(record) for record in records)
        score = successes / len(records)
        mean_return = statistics.fmean(
            (
                record.total_reward
                if record.policy_failure is None
                else 0.0
            )
            for record in records
        )
        mean_steps = statistics.fmean(record.steps for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Reached the goal in {successes}/{len(records)} "
                    f"Episodes ({score:.3f} success rate)."
                ),
                "success_rate": score,
                "mean_return": mean_return,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "successful_episodes": successes,
                "policy_failures": failures,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: FrozenLakeConfig) -> BenchmarkSpec:
    rows = len(config.layout)
    columns = len(config.layout[0])
    return BenchmarkSpec(
        id="gymnasium/FrozenLake-v1/success-rate-v1",
        description=(
            f"Navigate the standard {config.map_name} FrozenLake map without "
            "falling into a hole. Choose 0, 1, 2, or 3 to move left, down, "
            "right, or up. Maximize goal-reaching rate."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "state": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": rows * columns - 1,
                },
                "row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": rows - 1,
                },
                "column": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": columns - 1,
                },
                "tile": {
                    "type": "string",
                    "values": ["S", "F", "H", "G"],
                },
            },
        },
        action_space={
            "type": "discrete",
            "values": [0, 1, 2, 3],
            "meaning": {
                "0": "move_left",
                "1": "move_down",
                "2": "move_right",
                "3": "move_up",
            },
        },
        metadata={
            "environment": config.environment_id,
            "provider": "Gymnasium",
            "goal_reward": 1.0,
            "hole_reward": 0.0,
            "frozen_reward": 0.0,
            "failure_return": 0.0,
        },
        environment_parameters={
            "map_name": config.map_name,
            "map": list(config.layout),
            "is_slippery": config.is_slippery,
            "success_rate": config.success_rate,
            "reward_schedule": {
                "goal": 1.0,
                "hole": 0.0,
                "frozen": 0.0,
            },
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="success_rate",
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


def _successful(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.total_reward > 0.0
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
                        else 0.0
                    ),
                    "reached_goal": _successful(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int:
                raise ValueError("FrozenLake trace Action is invalid")
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


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise ValueError("FrozenLake trace observation is invalid")
    if set(observation) != {"state", "row", "column", "tile"}:
        raise ValueError("FrozenLake trace observation is invalid")
    if any(type(observation[key]) is not int for key in ("state", "row", "column")):
        raise ValueError("FrozenLake trace observation is invalid")
    if (
        type(observation["tile"]) is not str
        or observation["tile"] not in {"S", "F", "H", "G"}
    ):
        raise ValueError("FrozenLake trace observation is invalid")
    return {
        "state": observation["state"],
        "row": observation["row"],
        "column": observation["column"],
        "tile": observation["tile"],
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


__all__ = ["FrozenLakeBenchmark"]
