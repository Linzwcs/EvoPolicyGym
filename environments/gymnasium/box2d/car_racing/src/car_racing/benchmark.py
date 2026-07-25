"""A parameterized CarRacing-v3 Benchmark with lossless pixel traces."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
import zlib
from collections.abc import Sequence

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue

from .config import CarRacingConfig
from .environment import CarRacingEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-car-racing/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 1
_MAX_EPISODE_STEPS = 1_000
_FAILURE_RETURN = -1_000.0
_FRAME_SHAPE = (96, 96, 3)
_FRAME_BYTES = 96 * 96 * 3


class CarRacingBenchmark:
    """Mean CarRacing return over deterministic Episode plans."""

    def __init__(self, config: CarRacingConfig | None = None) -> None:
        if config is None:
            config = CarRacingConfig()
        if type(config) is not CarRacingConfig:
            raise TypeError("config must be CarRacingConfig")
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
        return CarRacingEnvironment(episode, config=self._config)

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
        laps = sum(_completed_lap(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{laps} completed laps."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "completed_laps": laps,
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
                "trace_frame_encoding": "zlib+base64",
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    continuous=self._config.continuous,
                ),
            ),
        )


def _benchmark_spec(config: CarRacingConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.continuous:
        action_space = {
            "type": "array",
            "shape": [3],
            "components": [
                {
                    "name": "steering",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                {
                    "name": "gas",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                {
                    "name": "brake",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            ],
            "items": {"type": "float"},
        }
    else:
        action_space = {
            "type": "discrete",
            "values": [0, 1, 2, 3, 4],
            "meaning": {
                "0": "do_nothing",
                "1": "steer_right",
                "2": "steer_left",
                "3": "gas",
                "4": "brake",
            },
        }
    return BenchmarkSpec(
        id="gymnasium/CarRacing-v3/mean-return-v1",
        description=(
            "Drive a rear-wheel-drive car around a procedurally generated "
            "track using exact RGB observations. Maximize tile coverage while "
            "finishing quickly and staying inside the playfield."
        ),
        observation_space={
            "type": "tensor",
            "dtype": "uint8",
            "shape": [96, 96, 3],
            "layout": "HWC",
            "channels": ["red", "green", "blue"],
            "minimum": 0,
            "maximum": 255,
        },
        action_space=action_space,
        metadata={
            "environment": "CarRacing-v3",
            "provider": "Gymnasium",
            "reward_threshold": 900.0,
            "off_playfield_terminal_reward": -100.0,
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={
            "continuous": config.continuous,
            "lap_complete_percent": config.lap_complete_percent,
            "domain_randomize": config.domain_randomize,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
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


def _completed_lap(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward > -100.0
    )


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    continuous: bool,
) -> Artifact:
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
                    "completed_lap": _completed_lap(record),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_frame(
                        record.initial_observation
                    ),
                }
            )
        )
        for step_index, transition in enumerate(record.transitions):
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "action": _trace_action(
                            transition.action,
                            continuous=continuous,
                        ),
                        "reward": transition.step.reward,
                        "next_observation": _trace_frame(
                            transition.step.observation
                        ),
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


def _trace_action(
    action: PolicyValue,
    *,
    continuous: bool,
) -> PolicyValue:
    if not continuous:
        if type(action) is not int or action not in {0, 1, 2, 3, 4}:
            raise ValueError("CarRacing trace Action is invalid")
        return action
    if type(action) is not list or len(action) != 3:
        raise ValueError("CarRacing trace Action is invalid")
    bounds = ((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    traced: list[PolicyValue] = []
    for value, (minimum, maximum) in zip(action, bounds, strict=True):
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError("CarRacing trace Action is invalid")
        traced.append(value)
    return traced


def _trace_frame(observation: PolicyValue) -> dict[str, object]:
    if (
        type(observation) is not TensorValue
        or observation.dtype != "uint8"
        or observation.shape != _FRAME_SHAPE
        or len(observation.data) != _FRAME_BYTES
    ):
        raise ValueError("CarRacing trace observation is invalid")
    compressed = zlib.compress(observation.data, level=9)
    return {
        "type": "tensor",
        "dtype": observation.dtype,
        "shape": list(observation.shape),
        "layout": "HWC",
        "encoding": "zlib+base64",
        "data": base64.b64encode(compressed).decode("ascii"),
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


__all__ = ["CarRacingBenchmark"]
