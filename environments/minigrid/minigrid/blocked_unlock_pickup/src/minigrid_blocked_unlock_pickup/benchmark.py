"""BlockedUnlockPickup with deterministic plans and milestone traces."""

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

from .environment import BlockedUnlockPickupEnvironment

_EPISODE_SEED_DOMAIN = (
    b"evopolicygym-minigrid-blocked-unlock-pickup/episode-seed/v1\0"
)
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
_ENVIRONMENT_ID = "MiniGrid-BlockedUnlockPickup-v0"
_OBJECT_NAMES = (
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
_COLOR_NAMES = ("red", "green", "blue", "purple", "yellow", "grey")
_STATE_NAMES = ("open", "closed", "locked")
_OBJECT_SYMBOLS = ("?", " ", "#", ".", "D", "K", "O", "B", "G", "L", "A")
_MISSIONS = tuple(f"pick up the {color} box" for color in _COLOR_NAMES)
_ACTION_MEANING: dict[str, PolicyValue] = {
    "0": "turn_left",
    "1": "turn_right",
    "2": "move_forward",
    "3": "pick_up",
    "4": "drop",
    "5": "toggle",
    "6": "done",
}
_METRIC_FIELDS = {
    "blocker_found",
    "blocker_moved",
    "key_found",
    "key_picked_up",
    "door_opened",
    "target_found",
    "success",
}


class BlockedUnlockPickupBenchmark:
    """Success rate for the complete blocked-door manipulation chain."""

    def __init__(self) -> None:
        self._spec = _benchmark_spec()

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
            EpisodeSpec(environment_seed=_episode_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return BlockedUnlockPickupEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successful = tuple(record for record in records if _successful(record))
        successes = len(successful)
        score = successes / len(records)
        counts = {
            name: _milestone_count(records, name)
            for name in _METRIC_FIELDS - {"success"}
        }
        mean_return = statistics.fmean(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        mean_steps = statistics.fmean(record.steps for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        truncations = sum(_truncated(record) for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Moved the blocker, unlocked the room, and collected the "
                f"target in {successes}/{len(records)} Episodes "
                f"({score:.3f} success rate)."
            ),
            "success_rate": score,
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful)
                if successful
                else None
            ),
            "episodes": len(records),
            "successful_episodes": successes,
            "truncated_episodes": truncations,
            "policy_failures": failures,
            "traced_episodes": len(traced),
            "trace_episodes_omitted": len(records) - len(traced),
            "trace_prefix_steps": _TRACE_PREFIX_STEPS,
            "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
        }
        for name, value in counts.items():
            content[f"{name}_episodes"] = value
            content[f"{name}_rate"] = value / len(records)
        return Feedback(
            score=score,
            content=content,
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/BlockedUnlockPickup-v0/success-rate-v1",
        description=(
            "Move the ball obstructing a locked door, acquire the matching "
            "key, unlock the second room, and pick up the mission box."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "image": {
                    "type": "tensor",
                    "dtype": "uint8",
                    "shape": [7, 7, 3],
                    "layout": "XYC",
                    "channels": ["object", "color", "state"],
                },
                "direction": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                },
                "mission": {"type": "string", "values": list(_MISSIONS)},
            },
        },
        action_space={
            "type": "discrete",
            "values": list(range(7)),
            "meaning": _ACTION_MEANING,
        },
        metadata={
            "environment": _ENVIRONMENT_ID,
            "provider": "MiniGrid",
            "failure_return": 0.0,
            "partial_observability": True,
        },
        environment_parameters={
            "room_size": 6,
            "num_rows": 1,
            "num_columns": 2,
            "view_size": 7,
            "agent_view_position": [3, 6],
            "mission_template": "pick up the {color} box",
            "object_encoding": {
                name: code for code, name in enumerate(_OBJECT_NAMES)
            },
            "color_encoding": {
                name: code for code, name in enumerate(_COLOR_NAMES)
            },
            "state_encoding": {
                name: code for code, name in enumerate(_STATE_NAMES)
            },
            "reward": (
                "positive discounted reward for picking up the target box; "
                "zero for timeout"
            ),
        },
        max_episode_steps=16 * 6**2,
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


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _milestone_count(
    records: Sequence[EpisodeRecord],
    name: str,
) -> int:
    return sum(
        any(
            type(transition.step.metrics) is dict
            and transition.step.metrics.get(name) is True
            for transition in record.transitions
        )
        for record in records
    )


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        selected_steps = _trace_indices(record.steps)
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
                    "success": _successful(record),
                    "truncated": _truncated(record),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_observation(
                        record.initial_observation
                    ),
                    "traced_steps": len(selected_steps),
                    "omitted_steps": record.steps - len(selected_steps),
                }
            )
        )
        for step_index in selected_steps:
            transition = record.transitions[step_index]
            if type(transition.action) is not int or not 0 <= transition.action <= 6:
                raise ValueError(
                    "BlockedUnlockPickup trace Action is invalid"
                )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "action": transition.action,
                        "action_meaning": _ACTION_MEANING[
                            str(transition.action)
                        ],
                        "reward": transition.step.reward,
                        "next_observation": _trace_observation(
                            transition.step.observation
                        ),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": _trace_metrics(
                            transition.step.metrics
                        ),
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_indices(step_count: int) -> tuple[int, ...]:
    if step_count <= _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS:
        return tuple(range(step_count))
    return (
        *range(_TRACE_PREFIX_STEPS),
        *range(step_count - _TRACE_SUFFIX_STEPS, step_count),
    )


def _trace_metrics(metrics: PolicyValue) -> dict[str, PolicyValue]:
    if (
        type(metrics) is not dict
        or set(metrics) != _METRIC_FIELDS
        or any(type(value) is not bool for value in metrics.values())
    ):
        raise ValueError("BlockedUnlockPickup trace metrics are invalid")
    return dict(metrics)


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict or set(observation) != {
        "image",
        "direction",
        "mission",
    }:
        raise ValueError("BlockedUnlockPickup trace observation is invalid")
    image = observation["image"]
    direction = observation["direction"]
    mission = observation["mission"]
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
    ):
        raise ValueError("BlockedUnlockPickup trace image is invalid")
    if type(direction) is not int or not 0 <= direction <= 3:
        raise ValueError("BlockedUnlockPickup trace direction is invalid")
    if type(mission) is not str or mission not in _MISSIONS:
        raise ValueError("BlockedUnlockPickup trace mission is invalid")
    rows: list[PolicyValue] = []
    visible_objects: list[PolicyValue] = []
    for y in range(7):
        row: list[str] = []
        for x in range(7):
            offset = (x * 7 + y) * 3
            object_code, color_code, state_code = image.data[offset : offset + 3]
            if (
                object_code >= len(_OBJECT_NAMES)
                or color_code >= len(_COLOR_NAMES)
                or state_code >= len(_STATE_NAMES)
            ):
                raise ValueError(
                    "BlockedUnlockPickup trace image codes are invalid"
                )
            row.append(_OBJECT_SYMBOLS[object_code])
            if object_code not in {0, 1}:
                visible_objects.append(
                    {
                        "x": x,
                        "y": y,
                        "object": _OBJECT_NAMES[object_code],
                        "color": _COLOR_NAMES[color_code],
                        "state": _STATE_NAMES[state_code],
                    }
                )
        rows.append("".join(row))
    return {
        "direction": direction,
        "mission": mission,
        "grid_rows": rows,
        "visible_objects": visible_objects,
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


__all__ = ["BlockedUnlockPickupBenchmark"]
