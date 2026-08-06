"""MiniGrid Unlock with deterministic plans and milestone traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue

from .environment import UnlockEnvironment

_SEED_DOMAIN = b"evopolicygym-minigrid-unlock/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
_MISSION = "open the door"
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
_ACTION_NAMES = tuple(str(_ACTIONS[str(index)]) for index in range(7))
_METRICS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "key_visible",
        "key_found",
        "key_first_seen_step",
        "key_color_found",
        "key_picked_up_this_step",
        "key_picked_up",
        "key_picked_up_step",
        "key_dropped_this_step",
        "key_dropped",
        "carried_key_color",
        "carried_key_color_before_action",
        "matching_key_carried",
        "door_found",
        "door_first_seen_step",
        "door_color_found",
        "locked_door_visible",
        "open_door_visible",
        "front_object",
        "front_object_before_action",
        "door_opened_this_step",
        "door_opened",
        "door_opened_step",
        "pickup_attempt",
        "pickup_attempt_count",
        "failed_pickup",
        "failed_pickup_count",
        "drop_attempt",
        "drop_attempt_count",
        "failed_drop",
        "failed_drop_count",
        "toggle_attempt",
        "toggle_attempt_count",
        "failed_toggle",
        "failed_toggle_count",
        "task_stage",
        "observation_novel",
        "unique_observation_count",
        "observation_novelty_step_fraction",
        "ineffective_action",
        "ineffective_action_fraction",
        "success_reward_at_this_step",
        "cumulative_return",
        "success",
        "terminal_reason",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
)
_MILESTONES = (
    "key_found",
    "key_picked_up",
    "key_dropped",
    "door_found",
    "door_opened",
    "failed_pickup",
    "failed_drop",
    "failed_toggle",
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    key_first_seen_step: int
    key_picked_up_step: int
    door_first_seen_step: int
    door_opened_step: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    failed_pickup_count: int
    failed_drop_count: int
    failed_toggle_count: int
    action_counts: tuple[int, ...]
    key_color_found: str
    door_color_found: str
    front_object_before_action: str
    task_stage: str
    outcome: str


class UnlockBenchmark:
    """Mean upstream Episode return for this Benchmark."""

    def __init__(self) -> None:
        self._spec = _spec()

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
            EpisodeSpec(environment_seed=_seed(split, seed, index)) for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        return UnlockEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        success_rate = successes / len(records)
        mean_return = statistics.fmean(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        counts = {name: sum(_reached(record, name) for record in records) for name in _MILESTONES}
        successful = tuple(record for record in records if _success(record))
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        traced = records[:4]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Opened the locked door in {successes}/{len(records)} "
                f"Episodes ({success_rate:.3f} success rate)."
            ),
            "success_rate": success_rate,
            "mean_return": statistics.fmean(
                record.total_reward if record.policy_failure is None else 0.0 for record in records
            ),
            "mean_steps": statistics.fmean(record.steps for record in records),
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful) if successful else None
            ),
            "mean_key_first_seen_step": _mean_milestone(
                tuple(item.key_first_seen_step for item in diagnostics)
            ),
            "mean_key_picked_up_step": _mean_milestone(
                tuple(item.key_picked_up_step for item in diagnostics)
            ),
            "mean_door_first_seen_step": _mean_milestone(
                tuple(item.door_first_seen_step for item in diagnostics)
            ),
            "mean_door_opened_step": _mean_milestone(
                tuple(item.door_opened_step for item in diagnostics)
            ),
            "mean_unique_observation_count": _mean_or_none(
                tuple(float(item.unique_observation_count) for item in diagnostics)
            ),
            "mean_observation_novelty_step_fraction": _mean_or_none(
                tuple(item.observation_novelty_step_fraction for item in diagnostics)
            ),
            "mean_ineffective_action_fraction": _mean_or_none(
                tuple(item.ineffective_action_fraction for item in diagnostics)
            ),
            "mean_failed_pickup_count": _mean_or_none(
                tuple(float(item.failed_pickup_count) for item in diagnostics)
            ),
            "mean_failed_drop_count": _mean_or_none(
                tuple(float(item.failed_drop_count) for item in diagnostics)
            ),
            "mean_failed_toggle_count": _mean_or_none(
                tuple(float(item.failed_toggle_count) for item in diagnostics)
            ),
            "episodes": len(records),
            "successful_episodes": successes,
            "truncated_episodes": sum(_truncated(r) for r in records),
            "policy_failures": sum(record.policy_failure is not None for record in records),
            "traced_episodes": len(traced),
            "trace_episodes_omitted": len(records) - len(traced),
            "trace_prefix_steps": _TRACE_PREFIX_STEPS,
            "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
        }
        for name, value in counts.items():
            content[f"{name}_rate"] = value / len(records)
            content[f"{name}_episodes"] = value
        for action_index, name in enumerate(_ACTION_NAMES):
            content[f"mean_{name}_fraction"] = _mean_or_none(
                tuple(
                    item.action_counts[action_index] / sum(item.action_counts)
                    for item in diagnostics
                )
            )
        return Feedback(
            score=mean_return,
            content=content,
            artifacts=(_trace(traced),),
        )


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/Unlock-v0/mean-return-v1",
        description=(
            "Explore the first room, pick up the key matching the locked "
            "door, and unlock it. Maximize upstream Episode return."
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
                    "axis_order": ["view_x", "view_y", "channel"],
                    "meaning": (
                        "Agent-centric view: agent at (3,6), forward decreases "
                        "view_y, and right increases view_x. A carried key is "
                        "encoded at the agent position."
                    ),
                },
                "direction": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "meaning": {
                        "0": "east",
                        "1": "south",
                        "2": "west",
                        "3": "north",
                    },
                },
                "mission": {"type": "string", "constant": _MISSION},
            },
        },
        action_space={
            "type": "discrete",
            "values": list(range(7)),
            "meaning": _ACTIONS,
        },
        metadata={
            "environment": "MiniGrid-Unlock-v0",
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
            "view_forward_direction": "decreasing view_y",
            "view_right_direction": "increasing view_x",
            "image_axis_order": ["view_x", "view_y", "channel"],
            "image_channel_order": ["object", "color", "state"],
            "direction_encoding": {
                "east": 0,
                "south": 1,
                "west": 2,
                "north": 3,
            },
            "mission": _MISSION,
            "object_encoding": {name: code for code, name in enumerate(_OBJECTS)},
            "color_encoding": {name: code for code, name in enumerate(_COLORS)},
            "state_encoding": {name: code for code, name in enumerate(_STATES)},
            "success_reward_formula": ("1 - 0.9*step_count/max_episode_steps"),
            "non_success_reward": 0.0,
            "natural_termination": (
                "the toggle action terminates with success only when it "
                "opens the locked door; other actions do not naturally "
                "terminate"
            ),
            "time_limit": 8 * 6**2,
        },
        max_episode_steps=8 * 6**2,
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
        type(item.step.metrics) is dict and item.step.metrics.get(name) is True
        for item in record.transitions
    )


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if not record.transitions:
        return "incomplete"
    final = _trace_metrics(record.transitions[-1].step.metrics)
    reason = _string_metric(final, "terminal_reason")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("MiniGrid Unlock diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        key_first_seen_step=_int_metric(final, "key_first_seen_step"),
        key_picked_up_step=_int_metric(final, "key_picked_up_step"),
        door_first_seen_step=_int_metric(final, "door_first_seen_step"),
        door_opened_step=_int_metric(final, "door_opened_step"),
        unique_observation_count=_int_metric(
            final,
            "unique_observation_count",
        ),
        observation_novelty_step_fraction=_float_metric(
            final,
            "observation_novelty_step_fraction",
        ),
        ineffective_action_fraction=_float_metric(
            final,
            "ineffective_action_fraction",
        ),
        failed_pickup_count=_int_metric(final, "failed_pickup_count"),
        failed_drop_count=_int_metric(final, "failed_drop_count"),
        failed_toggle_count=_int_metric(final, "failed_toggle_count"),
        action_counts=tuple(_int_metric(final, f"{name}_count") for name in _ACTION_NAMES),
        key_color_found=_string_metric(final, "key_color_found"),
        door_color_found=_string_metric(final, "door_color_found"),
        front_object_before_action=_string_metric(
            final,
            "front_object_before_action",
        ),
        task_stage=_string_metric(final, "task_stage"),
        outcome=_episode_outcome(record),
    )


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"MiniGrid Unlock metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"MiniGrid Unlock metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"MiniGrid Unlock metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_milestone(values: tuple[int, ...]) -> float | None:
    reached = tuple(value for value in values if value >= 0)
    return statistics.fmean(reached) if reached else None


def _trace(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        diagnostics = (
            _episode_diagnostics(record)
            if record.policy_failure is None and record.transitions
            else None
        )
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "steps": record.steps,
                    "return": (record.total_reward if record.policy_failure is None else 0.0),
                    "key_found": _reached(record, "key_found"),
                    "key_picked_up": _reached(record, "key_picked_up"),
                    "key_dropped": _reached(record, "key_dropped"),
                    "door_found": _reached(record, "door_found"),
                    "door_opened": _reached(record, "door_opened"),
                    "success": _success(record),
                    "truncated": _truncated(record),
                    "outcome": (
                        diagnostics.outcome if diagnostics is not None else _episode_outcome(record)
                    ),
                    "task_stage": (diagnostics.task_stage if diagnostics is not None else None),
                    "key_color_found": (
                        diagnostics.key_color_found if diagnostics is not None else None
                    ),
                    "door_color_found": (
                        diagnostics.door_color_found if diagnostics is not None else None
                    ),
                    "key_first_seen_step": (
                        diagnostics.key_first_seen_step if diagnostics is not None else None
                    ),
                    "key_picked_up_step": (
                        diagnostics.key_picked_up_step if diagnostics is not None else None
                    ),
                    "door_first_seen_step": (
                        diagnostics.door_first_seen_step if diagnostics is not None else None
                    ),
                    "door_opened_step": (
                        diagnostics.door_opened_step if diagnostics is not None else None
                    ),
                    "front_object_before_action": (
                        diagnostics.front_object_before_action if diagnostics is not None else None
                    ),
                    "failed_pickup_count": (
                        diagnostics.failed_pickup_count if diagnostics is not None else None
                    ),
                    "failed_drop_count": (
                        diagnostics.failed_drop_count if diagnostics is not None else None
                    ),
                    "failed_toggle_count": (
                        diagnostics.failed_toggle_count if diagnostics is not None else None
                    ),
                    "unique_observation_count": (
                        diagnostics.unique_observation_count if diagnostics is not None else None
                    ),
                    "ineffective_action_fraction": (
                        diagnostics.ineffective_action_fraction if diagnostics is not None else None
                    ),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_observation(record.initial_observation),
                    "traced_steps": min(
                        record.steps,
                        _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS,
                    ),
                    "omitted_steps": max(
                        0,
                        record.steps - _TRACE_PREFIX_STEPS - _TRACE_SUFFIX_STEPS,
                    ),
                }
            )
        )
        indices = (
            tuple(range(record.steps))
            if record.steps <= _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS
            else (
                *range(_TRACE_PREFIX_STEPS),
                *range(record.steps - _TRACE_SUFFIX_STEPS, record.steps),
            )
        )
        for index in indices:
            item = record.transitions[index]
            if type(item.action) is not int or not 0 <= item.action <= 6:
                raise ValueError("MiniGrid Unlock trace Action is invalid")
            lines.append(
                _json(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": index,
                        "action": item.action,
                        "action_meaning": _ACTIONS[str(item.action)],
                        "reward": item.step.reward,
                        "next_observation": _trace_observation(item.step.observation),
                        "terminated": item.step.terminated,
                        "truncated": item.step.truncated,
                        "metrics": _trace_metrics(item.step.metrics),
                    }
                )
            )
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_metrics(metrics: PolicyValue) -> dict[str, PolicyValue]:
    if type(metrics) is not dict or set(metrics) != set(_METRICS):
        raise ValueError("MiniGrid Unlock trace metrics are invalid")
    boolean_fields = {
        "key_visible",
        "key_found",
        "key_picked_up_this_step",
        "key_picked_up",
        "key_dropped_this_step",
        "key_dropped",
        "matching_key_carried",
        "door_found",
        "locked_door_visible",
        "open_door_visible",
        "door_opened_this_step",
        "door_opened",
        "pickup_attempt",
        "failed_pickup",
        "drop_attempt",
        "failed_drop",
        "toggle_attempt",
        "failed_toggle",
        "observation_novel",
        "ineffective_action",
        "success",
    }
    integer_fields = {
        "step_count",
        "remaining_steps",
        "key_first_seen_step",
        "key_picked_up_step",
        "door_first_seen_step",
        "door_opened_step",
        "pickup_attempt_count",
        "failed_pickup_count",
        "drop_attempt_count",
        "failed_drop_count",
        "toggle_attempt_count",
        "failed_toggle_count",
        "unique_observation_count",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
    string_fields = {
        "key_color_found",
        "carried_key_color",
        "carried_key_color_before_action",
        "door_color_found",
        "front_object",
        "front_object_before_action",
        "task_stage",
        "terminal_reason",
    }
    traced: dict[str, PolicyValue] = {}
    for key in _METRICS:
        value = metrics[key]
        if key in boolean_fields:
            if type(value) is not bool:
                raise ValueError("MiniGrid Unlock trace metrics are invalid")
        elif key in integer_fields:
            if type(value) is not int:
                raise ValueError("MiniGrid Unlock trace metrics are invalid")
        elif key in string_fields:
            if type(value) is not str:
                raise ValueError("MiniGrid Unlock trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("MiniGrid Unlock trace metrics are invalid")
        traced[key] = value
    return traced


def _trace_observation(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise ValueError("MiniGrid Unlock trace observation is invalid")
    image = value.get("image")
    direction = value.get("direction")
    mission = value.get("mission")
    if (
        type(image) is not TensorValue
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
        or type(direction) is not int
        or type(mission) is not str
        or mission != _MISSION
    ):
        raise ValueError("MiniGrid Unlock trace observation is invalid")
    rows: list[PolicyValue] = []
    visible_objects: list[PolicyValue] = []
    for y in range(7):
        row: list[str] = []
        for x in range(7):
            offset = (x * 7 + y) * 3
            object_code, color_code, state_code = image.data[offset : offset + 3]
            if (
                object_code >= len(_OBJECTS)
                or color_code >= len(_COLORS)
                or state_code >= len(_STATES)
            ):
                raise ValueError("MiniGrid Unlock trace image codes are invalid")
            row.append(_SYMBOLS[object_code])
            if object_code not in {0, 1}:
                visible_objects.append(
                    {
                        "x": x,
                        "y": y,
                        "object": _OBJECTS[object_code],
                        "color": _COLORS[color_code],
                        "state": _STATES[state_code],
                    }
                )
        rows.append("".join(row))
    return {
        "direction": direction,
        "mission": mission,
        "grid_rows": rows,
        "visible_objects": visible_objects,
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


__all__ = ["UnlockBenchmark"]
