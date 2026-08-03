"""MiniGrid Playground room coverage with bounded actionable traces."""

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

from .environment import PlaygroundEnvironment

_SEED_DOMAIN = b"evopolicygym-minigrid-playground/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
_MAX_EPISODE_STEPS = 1000
_ROOMS = 9
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
_LABEL_METRICS = frozenset(
    {
        "carried_object",
        "carried_object_before_action",
        "front_object",
        "front_object_before_action",
        "task_stage",
        "terminal_reason",
    }
)
_BOOL_METRICS = frozenset(
    {
        "new_room",
        "door_found",
        "door_opened_this_step",
        "door_closed_this_step",
        "door_crossed_this_step",
        "object_picked_up_this_step",
        "object_dropped_this_step",
        "box_destroyed_this_step",
        "pickup_attempt",
        "failed_pickup",
        "drop_attempt",
        "failed_drop",
        "toggle_attempt",
        "toggle_effective",
        "failed_toggle",
        "blocked_forward",
        "done_action",
        "observation_novel",
        "ineffective_action",
        "success",
    }
)
_INT_METRICS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "rooms_visited",
        "rooms_remaining",
        "new_room_entry_count",
        "last_new_room_step",
        "steps_since_new_room",
        "visible_door_count",
        "visible_closed_door_count",
        "visible_open_door_count",
        "door_open_event_count",
        "door_close_event_count",
        "door_crossing_event_count",
        "visible_portable_object_count",
        "pickup_event_count",
        "drop_event_count",
        "box_destroy_event_count",
        "pickup_attempt_count",
        "failed_pickup_count",
        "drop_attempt_count",
        "failed_drop_count",
        "toggle_attempt_count",
        "failed_toggle_count",
        "blocked_forward_count",
        "done_action_count",
        "unique_observation_count",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
)
_MILESTONE_INT_METRICS = frozenset(
    {
        "door_first_seen_step",
        *(f"room_{room}_first_entry_step" for room in range(2, _ROOMS + 1)),
    }
)
_FLOAT_METRICS = frozenset(
    {
        "room_coverage",
        "coverage_gain",
        "observation_novelty_step_fraction",
        "ineffective_action_fraction",
        "cumulative_return",
    }
)
_METRICS = _LABEL_METRICS | _BOOL_METRICS | _INT_METRICS | _MILESTONE_INT_METRICS | _FLOAT_METRICS
_MILESTONES = (
    "door_found",
    "door_opened_this_step",
    "door_crossed_this_step",
    "object_picked_up_this_step",
    "object_dropped_this_step",
    "box_destroyed_this_step",
    "failed_pickup",
    "failed_drop",
    "failed_toggle",
    "blocked_forward",
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    room_first_entry_steps: tuple[int, ...]
    door_first_seen_step: int
    door_open_event_count: int
    door_close_event_count: int
    door_crossing_event_count: int
    pickup_event_count: int
    drop_event_count: int
    box_destroy_event_count: int
    failed_pickup_count: int
    failed_drop_count: int
    failed_toggle_count: int
    blocked_forward_count: int
    done_action_count: int
    steps_since_new_room: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    action_counts: tuple[int, ...]
    task_stage: str
    outcome: str


class PlaygroundBenchmark:
    """Coverage score for a generated nine-room playground."""

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
        return PlaygroundEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successes = sum(_success(record) for record in records)
        coverages = tuple(_final_coverage(record) for record in records)
        score = statistics.fmean(coverages)
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        milestones = {
            name: sum(_reached(record, name) for record in records) for name in _MILESTONES
        }
        traced = records[:4]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Mean room coverage was {score:.3f}; all nine rooms were "
                f"visited in {successes}/{len(records)} Episodes."
            ),
            "mean_room_coverage": score,
            "full_coverage_rate": successes / len(records),
            "mean_rooms_visited": statistics.fmean(coverage * _ROOMS for coverage in coverages),
            "mean_return": statistics.fmean(
                record.total_reward if record.policy_failure is None else 0.0 for record in records
            ),
            "mean_steps": statistics.fmean(record.steps for record in records),
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in records if _success(record))
                if successes
                else None
            ),
            "mean_door_open_event_count": _mean_int(diagnostics, "door_open_event_count"),
            "mean_door_close_event_count": _mean_int(diagnostics, "door_close_event_count"),
            "mean_door_crossing_event_count": _mean_int(diagnostics, "door_crossing_event_count"),
            "mean_pickup_event_count": _mean_int(diagnostics, "pickup_event_count"),
            "mean_drop_event_count": _mean_int(diagnostics, "drop_event_count"),
            "mean_box_destroy_event_count": _mean_int(diagnostics, "box_destroy_event_count"),
            "mean_failed_pickup_count": _mean_int(diagnostics, "failed_pickup_count"),
            "mean_failed_drop_count": _mean_int(diagnostics, "failed_drop_count"),
            "mean_failed_toggle_count": _mean_int(diagnostics, "failed_toggle_count"),
            "mean_blocked_forward_count": _mean_int(diagnostics, "blocked_forward_count"),
            "mean_done_action_count": _mean_int(diagnostics, "done_action_count"),
            "mean_steps_since_last_new_room": _mean_int(diagnostics, "steps_since_new_room"),
            "mean_unique_observation_count": _mean_int(diagnostics, "unique_observation_count"),
            "mean_observation_novelty_step_fraction": _mean_float(
                diagnostics, "observation_novelty_step_fraction"
            ),
            "mean_ineffective_action_fraction": _mean_float(
                diagnostics, "ineffective_action_fraction"
            ),
            "episodes": len(records),
            "successful_episodes": successes,
            "truncated_episodes": sum(_truncated(record) for record in records),
            "policy_failures": sum(record.policy_failure is not None for record in records),
            "traced_episodes": len(traced),
            "trace_episodes_omitted": len(records) - len(traced),
            "trace_prefix_steps": _TRACE_PREFIX_STEPS,
            "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
        }
        for room_count in range(2, _ROOMS + 1):
            values = tuple(item.room_first_entry_steps[room_count - 2] for item in diagnostics)
            reached = sum(value >= 0 for value in values)
            content[f"room_{room_count}_reached_rate"] = reached / len(records)
            content[f"mean_room_{room_count}_first_entry_step"] = _mean_milestone(values)
        for name, count in milestones.items():
            label = name.removesuffix("_this_step")
            content[f"{label}_episodes"] = count
            content[f"{label}_rate"] = count / len(records)
        for action_index, name in enumerate(_ACTION_NAMES):
            content[f"mean_{name}_fraction"] = _mean_or_none(
                tuple(
                    item.action_counts[action_index] / sum(item.action_counts)
                    for item in diagnostics
                )
            )
        return Feedback(score=score, content=content, artifacts=(_trace(traced),))


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/Playground-v0/room-coverage-v1",
        description=(
            "Open generated doors and visit all nine geometric rooms in an "
            "otherwise goal-free playground containing colored portable objects."
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
                        "view_y, and right increases view_x. A carried object "
                        "is encoded at the agent position."
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
                "mission": {"type": "string", "constant": ""},
            },
        },
        action_space={
            "type": "discrete",
            "values": list(range(7)),
            "meaning": _ACTIONS,
        },
        metadata={
            "environment": "MiniGrid-Playground-v0",
            "provider": "MiniGrid",
            "failure_return": 0.0,
            "partial_observability": True,
        },
        environment_parameters={
            "size": 19,
            "rooms": 9,
            "room_layout": [3, 3],
            "room_cell_span": 6,
            "random_portable_objects": 12,
            "upstream_has_goal": False,
            "upstream_has_reward": False,
            "upstream_default_max_episode_steps": 100,
            "benchmark_time_limit": _MAX_EPISODE_STEPS,
            "coverage_definition": (
                "the 19x19 grid is partitioned into its generated 3x3 rooms; "
                "coverage counts distinct rooms entered without exposing room "
                "coordinates to the Policy"
            ),
            "initial_room_counts_toward_coverage": True,
            "initial_room_reward": 0.0,
            "new_room_reward": 1.0,
            "maximum_episode_return": 8.0,
            "natural_termination": (
                "the wrapper terminates when all nine rooms have been entered; "
                "the upstream Playground never terminates naturally"
            ),
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
            "mission": "",
            "object_encoding": {name: code for code, name in enumerate(_OBJECTS)},
            "color_encoding": {name: code for code, name in enumerate(_COLORS)},
            "state_encoding": {name: code for code, name in enumerate(_STATES)},
            "action_notes": {
                "pick_up": "picks up a key, ball, or box when hands are empty",
                "drop": "drops a carried object into an empty front cell",
                "toggle": (
                    "opens or closes a door; toggling an empty Playground box destroys that box"
                ),
                "done": "unused no-op",
            },
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_room_coverage",
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
    if record.policy_failure is not None or not record.transitions:
        return False
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return bool(
        record.transitions[-1].step.terminated
        and final["success"] is True
        and final["room_coverage"] == 1.0
    )


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _final_coverage(record: EpisodeRecord) -> float:
    if record.policy_failure is not None or not record.transitions:
        return 0.0
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _float_metric(final, "room_coverage")


def _reached(record: EpisodeRecord, name: str) -> bool:
    return any(
        type(transition.step.metrics) is dict and transition.step.metrics.get(name) is True
        for transition in record.transitions
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
        raise ValueError("MiniGrid Playground diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        room_first_entry_steps=tuple(
            _int_metric(final, f"room_{room}_first_entry_step") for room in range(2, _ROOMS + 1)
        ),
        door_first_seen_step=_int_metric(final, "door_first_seen_step"),
        door_open_event_count=_int_metric(final, "door_open_event_count"),
        door_close_event_count=_int_metric(final, "door_close_event_count"),
        door_crossing_event_count=_int_metric(final, "door_crossing_event_count"),
        pickup_event_count=_int_metric(final, "pickup_event_count"),
        drop_event_count=_int_metric(final, "drop_event_count"),
        box_destroy_event_count=_int_metric(final, "box_destroy_event_count"),
        failed_pickup_count=_int_metric(final, "failed_pickup_count"),
        failed_drop_count=_int_metric(final, "failed_drop_count"),
        failed_toggle_count=_int_metric(final, "failed_toggle_count"),
        blocked_forward_count=_int_metric(final, "blocked_forward_count"),
        done_action_count=_int_metric(final, "done_action_count"),
        steps_since_new_room=_int_metric(final, "steps_since_new_room"),
        unique_observation_count=_int_metric(final, "unique_observation_count"),
        observation_novelty_step_fraction=_float_metric(final, "observation_novelty_step_fraction"),
        ineffective_action_fraction=_float_metric(final, "ineffective_action_fraction"),
        action_counts=tuple(_int_metric(final, f"{name}_count") for name in _ACTION_NAMES),
        task_stage=_string_metric(final, "task_stage"),
        outcome=_episode_outcome(record),
    )


def _mean_int(
    diagnostics: tuple[_EpisodeDiagnostics, ...],
    field: str,
) -> float | None:
    return _mean_or_none(tuple(float(getattr(item, field)) for item in diagnostics))


def _mean_float(
    diagnostics: tuple[_EpisodeDiagnostics, ...],
    field: str,
) -> float | None:
    return _mean_or_none(tuple(float(getattr(item, field)) for item in diagnostics))


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
        indices = _trace_indices(record.steps)
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "return": (record.total_reward if record.policy_failure is None else 0.0),
                    "room_coverage": _final_coverage(record),
                    "room_first_entry_steps": (
                        list(diagnostics.room_first_entry_steps) if diagnostics else []
                    ),
                    "outcome": _episode_outcome(record),
                    "success": _success(record),
                    "truncated": _truncated(record),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_observation(record.initial_observation),
                    "traced_steps": len(indices),
                    "omitted_steps": record.steps - len(indices),
                }
            )
        )
        for index in indices:
            item = record.transitions[index]
            if type(item.action) is not int or not 0 <= item.action <= 6:
                raise ValueError("MiniGrid Playground trace Action is invalid")
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


def _trace_indices(step_count: int) -> tuple[int, ...]:
    if step_count <= _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS:
        return tuple(range(step_count))
    return (
        *range(_TRACE_PREFIX_STEPS),
        *range(step_count - _TRACE_SUFFIX_STEPS, step_count),
    )


def _trace_metrics(metrics: PolicyValue) -> dict[str, PolicyValue]:
    if type(metrics) is not dict or set(metrics) != _METRICS:
        raise ValueError("MiniGrid Playground trace metrics are invalid")
    for name in _LABEL_METRICS:
        if type(metrics[name]) is not str:
            raise ValueError("MiniGrid Playground trace metrics are invalid")
    for name in _BOOL_METRICS:
        if type(metrics[name]) is not bool:
            raise ValueError("MiniGrid Playground trace metrics are invalid")
    for name in _INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < 0:
            raise ValueError("MiniGrid Playground trace metrics are invalid")
    for name in _MILESTONE_INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < -1:
            raise ValueError("MiniGrid Playground trace metrics are invalid")
    for name in _FLOAT_METRICS:
        value = metrics[name]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("MiniGrid Playground trace metrics are invalid")
    coverage = metrics["room_coverage"]
    if not isinstance(coverage, float) or not 0.0 <= coverage <= 1.0:
        raise ValueError("MiniGrid Playground trace metrics are invalid")
    return dict(metrics)


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"MiniGrid Playground metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"MiniGrid Playground metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"MiniGrid Playground metric {name} is invalid")
    return value


def _trace_observation(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict or set(value) != {"image", "direction", "mission"}:
        raise ValueError("MiniGrid Playground trace observation is invalid")
    image = value["image"]
    direction = value["direction"]
    mission = value["mission"]
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
        or type(direction) is not int
        or not 0 <= direction <= 3
        or type(mission) is not str
        or mission != ""
    ):
        raise ValueError("MiniGrid Playground trace observation is invalid")
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
                raise ValueError("MiniGrid Playground trace image codes are invalid")
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
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


__all__ = ["PlaygroundBenchmark"]
