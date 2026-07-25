"""A parameterized CliffWalking-v1 Benchmark with public traces."""

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

from .config import CliffWalkingConfig
from .environment import MAX_EPISODE_STEPS, CliffWalkingEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-cliff-walking/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -100.0 * MAX_EPISODE_STEPS
_MAP = (
    "............",
    "............",
    "............",
    "SCCCCCCCCCCG",
)


class CliffWalkingBenchmark:
    """Mean CliffWalking return over deterministic Episode plans."""

    def __init__(self, config: CliffWalkingConfig | None = None) -> None:
        if config is None:
            config = CliffWalkingConfig()
        if type(config) is not CliffWalkingConfig:
            raise TypeError("config must be CliffWalkingConfig")
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
        return CliffWalkingEnvironment(episode, config=self._config)

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
        successes = sum(_successful(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
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
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: CliffWalkingConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="gymnasium/CliffWalking-v1/mean-return-v1",
        description=(
            "Cross a 4 by 12 grid from the lower-left start to the lower-right "
            "goal without repeatedly falling from the intervening cliff. "
            "Choose up, right, down, or left and maximize mean return."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "state": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 47,
                },
                "row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                },
                "column": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 11,
                },
                "tile": {
                    "type": "string",
                    "values": ["start", "safe", "cliff", "goal"],
                },
            },
        },
        action_space={
            "type": "discrete",
            "values": [0, 1, 2, 3],
            "meaning": {
                "0": "move_up",
                "1": "move_right",
                "2": "move_down",
                "3": "move_left",
            },
        },
        metadata={
            "environment": "CliffWalking-v1",
            "provider": "Gymnasium",
            "map": list(_MAP),
            "start": [3, 0],
            "goal": [3, 11],
            "reward_schedule": {
                "ordinary_step": -1.0,
                "cliff_step": -100.0,
            },
            "slippery_direction_probability": 1.0 / 3.0,
            "failure_return": _FAILURE_RETURN,
            "benchmark_time_limit": MAX_EPISODE_STEPS,
        },
        environment_parameters={
            "is_slippery": config.is_slippery,
        },
        max_episode_steps=MAX_EPISODE_STEPS,
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


def _successful(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
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
                    "reached_goal": _successful(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int:
                raise ValueError("CliffWalking trace Action is invalid")
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
        raise ValueError("CliffWalking trace observation is invalid")
    if set(observation) != {"state", "row", "column", "tile"}:
        raise ValueError("CliffWalking trace observation is invalid")
    if any(
        type(observation[key]) is not int
        for key in ("state", "row", "column")
    ):
        raise ValueError("CliffWalking trace observation is invalid")
    if (
        type(observation["tile"]) is not str
        or observation["tile"]
        not in {"start", "safe", "cliff", "goal"}
    ):
        raise ValueError("CliffWalking trace observation is invalid")
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


__all__ = ["CliffWalkingBenchmark"]
