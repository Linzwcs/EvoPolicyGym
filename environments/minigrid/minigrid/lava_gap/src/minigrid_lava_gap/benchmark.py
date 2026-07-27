"""MiniGrid LavaGap with deterministic plans and safety traces."""

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
from evopolicygym.policy import PolicyValue, TensorValue

from .config import LavaGapConfig
from .environment import LavaGapEnvironment

_SEED_DOMAIN = b"evopolicygym-minigrid-lava_gap/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_OBJECTS = (
    "unseen",
    "empty",
    "wall",
    "floor",
    "door",
    "key",
    "ball",
    "box",
    "goal",
    "lava",
    "agent",
)
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_STATES = ("open", "closed", "locked")
_SYMBOLS = ("?", " ", "#", ".", "D", "K", "O", "B", "G", "L", "A")
_ACTIONS: dict[str, PolicyValue] = {
    "0": "turn_left",
    "1": "turn_right",
    "2": "move_forward",
    "3": "pick_up",
    "4": "drop",
    "5": "toggle",
    "6": "done",
}
_METRICS = {"goal_found", "hazard_entered", "success"}


class LavaGapBenchmark:
    """Safe-navigation success rate through one generated lava barrier."""

    def __init__(self, config: LavaGapConfig | None = None) -> None:
        if config is None:
            config = LavaGapConfig()
        if type(config) is not LavaGapConfig:
            raise TypeError("config must be LavaGapConfig")
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
        return LavaGapEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        score = successes / len(records)
        goal_found = sum(_reached(r, "goal_found") for r in records)
        hazards = sum(_reached(r, "hazard_entered") for r in records)
        traced = records[:4]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Reached the goal in {successes}/{len(records)} Episodes "
                    f"({score:.3f} success rate); {hazards} entered a hazard."
                ),
                "success_rate": score,
                "goal_found_rate": goal_found / len(records),
                "hazard_rate": hazards / len(records),
                "mean_return": statistics.fmean(
                    r.total_reward if r.policy_failure is None else 0.0
                    for r in records
                ),
                "mean_steps": statistics.fmean(r.steps for r in records),
                "episodes": len(records),
                "successful_episodes": successes,
                "hazard_episodes": hazards,
                "truncated_episodes": sum(_truncated(r) for r in records),
                "policy_failures": sum(
                    r.policy_failure is not None for r in records
                ),
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace(traced),),
        )


def _spec(config: LavaGapConfig) -> BenchmarkSpec:
    mission = "avoid the lava and get to the green goal square"
    return BenchmarkSpec(
        id="minigrid/LavaGap-v0/success-rate-v1",
        description=(
            "Discover the single opening in a generated vertical lava strip "
            "and safely reach the opposite-corner goal."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "image": {
                    "type": "tensor",
                    "dtype": "uint8",
                    "shape": [7, 7, 3],
                },
                "direction": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                },
                "mission": {"type": "string", "constant": mission},
            },
        },
        action_space={
            "type": "discrete",
            "values": list(range(7)),
            "meaning": _ACTIONS,
        },
        metadata={
            "environment": config.environment_id,
            "provider": "MiniGrid",
            "failure_return": 0.0,
            "partial_observability": True,
        },
        environment_parameters={
            "profile": config.profile,
            "size": config.size,
            "start_position": [1, 1],
            "goal_position": [config.size - 2, config.size - 2],
            "view_size": 7,
            "agent_view_position": [3, 6],
            "mission": mission,
            "object_encoding": {
                name: code for code, name in enumerate(_OBJECTS)
            },
            "color_encoding": {
                name: code for code, name in enumerate(_COLORS)
            },
            "state_encoding": {
                name: code for code, name in enumerate(_STATES)
            },
        },
        max_episode_steps=config.max_episode_steps,
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


def _success(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.total_reward > 0.0
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
                    "steps": record.steps,
                    "return": (
                        record.total_reward
                        if record.policy_failure is None
                        else 0.0
                    ),
                    "success": _success(record),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_observation(
                        record.initial_observation
                    ),
                }
            )
        )
        indices = (
            tuple(range(record.steps))
            if record.steps <= 160
            else (*range(128), *range(record.steps - 32, record.steps))
        )
        for index in indices:
            item = record.transitions[index]
            if type(item.action) is not int or not 0 <= item.action <= 6:
                raise ValueError("MiniGrid LavaGap trace Action is invalid")
            if (
                type(item.step.metrics) is not dict
                or set(item.step.metrics) != _METRICS
            ):
                raise ValueError("MiniGrid LavaGap trace metrics are invalid")
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": index,
                        "action": item.action,
                        "action_meaning": _ACTIONS[str(item.action)],
                        "reward": item.step.reward,
                        "next_observation": _trace_observation(
                            item.step.observation
                        ),
                        "terminated": item.step.terminated,
                        "truncated": item.step.truncated,
                        "metrics": item.step.metrics,
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_observation(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise ValueError("MiniGrid LavaGap trace observation is invalid")
    image = value.get("image")
    direction = value.get("direction")
    mission = value.get("mission")
    if (
        type(image) is not TensorValue
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
        or type(direction) is not int
        or type(mission) is not str
    ):
        raise ValueError("MiniGrid LavaGap trace observation is invalid")
    rows: list[PolicyValue] = []
    for y in range(7):
        rows.append(
            "".join(
                _SYMBOLS[image.data[(x * 7 + y) * 3]]
                for x in range(7)
            )
        )
    return {
        "direction": direction,
        "mission": mission,
        "grid_rows": rows,
    }


def _json(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


__all__ = ["LavaGapBenchmark"]
