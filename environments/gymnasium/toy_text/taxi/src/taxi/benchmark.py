"""A parameterized Taxi-v4 Benchmark with bounded public traces."""

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

from .config import TaxiConfig
from .environment import TaxiEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-taxi/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_FAILURE_RETURN = -2_000.0
_MAP = (
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
)
_LANDMARKS: dict[str, PolicyValue] = {
    "red": [0, 0],
    "green": [0, 4],
    "yellow": [4, 0],
    "blue": [4, 3],
}


class TaxiBenchmark:
    """Mean Taxi return over deterministic Episode plans."""

    def __init__(self, config: TaxiConfig | None = None) -> None:
        if config is None:
            config = TaxiConfig()
        if type(config) is not TaxiConfig:
            raise TypeError("config must be TaxiConfig")
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
        return TaxiEnvironment(episode, config=self._config)

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
        successes = sum(_successful(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        score = statistics.fmean(returns)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{successes} passengers delivered."
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


def _benchmark_spec(config: TaxiConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="gymnasium/Taxi-v4/mean-return-v1",
        description=(
            "Navigate a taxi to a passenger, pick them up, and deliver them "
            "to the requested landmark within 200 steps. Choose among four "
            "movement Actions, pickup, and dropoff. Maximize mean return."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "state": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 499,
                },
                "taxi_row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "taxi_column": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "passenger_location": {
                    "type": "string",
                    "values": [
                        "red",
                        "green",
                        "yellow",
                        "blue",
                        "in_taxi",
                    ],
                },
                "destination": {
                    "type": "string",
                    "values": ["red", "green", "yellow", "blue"],
                },
                "legal_actions": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5,
                    },
                },
            },
        },
        action_space={
            "type": "discrete",
            "values": [0, 1, 2, 3, 4, 5],
            "meaning": {
                "0": "move_south",
                "1": "move_north",
                "2": "move_east",
                "3": "move_west",
                "4": "pickup",
                "5": "dropoff",
            },
        },
        metadata={
            "environment": "Taxi-v4",
            "provider": "Gymnasium",
            "map": list(_MAP),
            "landmarks": _LANDMARKS,
            "reward_schedule": {
                "step": -1.0,
                "successful_dropoff": 20.0,
                "illegal_pickup_or_dropoff": -10.0,
            },
            "failure_return": _FAILURE_RETURN,
            "reward_threshold": 8.0,
        },
        environment_parameters={
            "is_rainy": config.is_rainy,
            "fickle_passenger": config.fickle_passenger,
            "rainy_probability": config.rainy_probability,
            "fickle_probability": config.fickle_probability,
        },
        max_episode_steps=200,
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
        and record.transitions[-1].step.reward == 20.0
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
                    "passenger_delivered": _successful(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int:
                raise ValueError("Taxi trace Action is invalid")
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
        raise ValueError("Taxi trace observation is invalid")
    expected = {
        "state",
        "taxi_row",
        "taxi_column",
        "passenger_location",
        "destination",
        "legal_actions",
    }
    if set(observation) != expected:
        raise ValueError("Taxi trace observation is invalid")
    if any(
        type(observation[key]) is not int
        for key in ("state", "taxi_row", "taxi_column")
    ):
        raise ValueError("Taxi trace observation is invalid")
    if (
        type(observation["passenger_location"]) is not str
        or observation["passenger_location"]
        not in {"red", "green", "yellow", "blue", "in_taxi"}
    ):
        raise ValueError("Taxi trace observation is invalid")
    if (
        type(observation["destination"]) is not str
        or observation["destination"]
        not in {"red", "green", "yellow", "blue"}
    ):
        raise ValueError("Taxi trace observation is invalid")
    legal_actions = observation["legal_actions"]
    if (
        type(legal_actions) is not list
        or any(type(action) is not int or action not in range(6) for action in legal_actions)
        or len(set(legal_actions)) != len(legal_actions)
    ):
        raise ValueError("Taxi trace observation is invalid")
    return {
        "state": observation["state"],
        "taxi_row": observation["taxi_row"],
        "taxi_column": observation["taxi_column"],
        "passenger_location": observation["passenger_location"],
        "destination": observation["destination"],
        "legal_actions": list(legal_actions),
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


__all__ = ["TaxiBenchmark"]
