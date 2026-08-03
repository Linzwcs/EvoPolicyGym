"""MiniGrid MultiRoom with deterministic plans and actionable traces."""

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

from .config import MultiRoomConfig
from .environment import MultiRoomEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-minigrid-multiroom/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
_MISSION = "traverse the rooms to get to the goal"
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
_ACTION_MEANING: dict[str, PolicyValue] = {
    "0": "turn_left",
    "1": "turn_right",
    "2": "move_forward",
    "3": "pick_up",
    "4": "drop",
    "5": "toggle",
    "6": "done",
}
_ACTION_NAMES = tuple(str(_ACTION_MEANING[str(index)]) for index in range(7))
_METRICS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "required_connecting_door_count",
        "visible_door_count",
        "visible_closed_door_count",
        "visible_open_door_count",
        "door_visible",
        "door_found",
        "door_first_seen_step",
        "front_object",
        "front_object_before_action",
        "closed_door_in_front",
        "closed_door_in_front_before_action",
        "open_door_in_front",
        "open_door_in_front_before_action",
        "toggle_attempt",
        "toggle_effective",
        "toggle_attempt_count",
        "failed_toggle",
        "failed_toggle_count",
        "door_opened_this_step",
        "door_closed_this_step",
        "door_open_event_count",
        "door_close_event_count",
        "first_door_opened_step",
        "door_crossed_this_step",
        "door_crossing_event_count",
        "goal_visible",
        "goal_found",
        "goal_first_seen_step",
        "goal_in_front",
        "goal_in_front_before_action",
        "forward_attempt",
        "forward_attempt_count",
        "blocked_forward",
        "blocked_forward_count",
        "unused_action",
        "unused_action_count",
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
_BOOL_METRICS = frozenset(
    {
        "door_visible",
        "door_found",
        "closed_door_in_front",
        "closed_door_in_front_before_action",
        "open_door_in_front",
        "open_door_in_front_before_action",
        "toggle_attempt",
        "toggle_effective",
        "failed_toggle",
        "door_opened_this_step",
        "door_closed_this_step",
        "door_crossed_this_step",
        "goal_visible",
        "goal_found",
        "goal_in_front",
        "goal_in_front_before_action",
        "forward_attempt",
        "blocked_forward",
        "unused_action",
        "observation_novel",
        "ineffective_action",
        "success",
    }
)
_INT_METRICS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "required_connecting_door_count",
        "visible_door_count",
        "visible_closed_door_count",
        "visible_open_door_count",
        "toggle_attempt_count",
        "failed_toggle_count",
        "door_open_event_count",
        "door_close_event_count",
        "door_crossing_event_count",
        "forward_attempt_count",
        "blocked_forward_count",
        "unused_action_count",
        "unique_observation_count",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
)
_MILESTONE_INT_METRICS = frozenset(
    {"door_first_seen_step", "first_door_opened_step", "goal_first_seen_step"}
)
_FLOAT_METRICS = frozenset(
    {
        "observation_novelty_step_fraction",
        "ineffective_action_fraction",
        "success_reward_at_this_step",
        "cumulative_return",
    }
)
_MILESTONES = {
    "door_found": "door_found",
    "door_opened": "door_opened_this_step",
    "door_crossed": "door_crossed_this_step",
    "goal_found": "goal_found",
    "failed_toggle": "failed_toggle",
    "blocked_forward": "blocked_forward",
}


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    door_first_seen_step: int
    first_door_opened_step: int
    goal_first_seen_step: int
    door_open_event_count: int
    door_close_event_count: int
    door_crossing_event_count: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    failed_toggle_count: int
    blocked_forward_count: int
    unused_action_count: int
    action_counts: tuple[int, ...]
    front_object_before_action: str
    task_stage: str
    outcome: str


class MultiRoomBenchmark:
    """Success rate on a partially observable connected-room task."""

    def __init__(self, config: MultiRoomConfig | None = None) -> None:
        if config is None:
            config = MultiRoomConfig()
        if type(config) is not MultiRoomConfig:
            raise TypeError("config must be MultiRoomConfig")
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
            EpisodeSpec(environment_seed=_episode_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return MultiRoomEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        successful = tuple(record for record in records if _successful(record))
        successes = len(successful)
        score = successes / len(records)
        milestones = {
            name: sum(_reached_bool_milestone(record, metric) for record in records)
            for name, metric in _MILESTONES.items()
        }
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        traced = records[:_MAX_TRACED_EPISODES]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Reached the final-room goal in {successes}/{len(records)} "
                f"Episodes ({score:.3f} success rate); recorded "
                f"{sum(item.door_open_event_count for item in diagnostics)} "
                "door-opening events."
            ),
            "success_rate": score,
            "mean_door_open_event_count": _mean_or_none(
                tuple(float(item.door_open_event_count) for item in diagnostics)
            ),
            "mean_door_close_event_count": _mean_or_none(
                tuple(float(item.door_close_event_count) for item in diagnostics)
            ),
            "mean_door_crossing_event_count": _mean_or_none(
                tuple(float(item.door_crossing_event_count) for item in diagnostics)
            ),
            "mean_return": statistics.fmean(
                record.total_reward if record.policy_failure is None else 0.0 for record in records
            ),
            "mean_steps": statistics.fmean(record.steps for record in records),
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful) if successful else None
            ),
            "mean_door_first_seen_step": _mean_milestone(
                tuple(item.door_first_seen_step for item in diagnostics)
            ),
            "mean_first_door_opened_step": _mean_milestone(
                tuple(item.first_door_opened_step for item in diagnostics)
            ),
            "mean_goal_first_seen_step": _mean_milestone(
                tuple(item.goal_first_seen_step for item in diagnostics)
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
            "mean_failed_toggle_count": _mean_or_none(
                tuple(float(item.failed_toggle_count) for item in diagnostics)
            ),
            "mean_blocked_forward_count": _mean_or_none(
                tuple(float(item.blocked_forward_count) for item in diagnostics)
            ),
            "mean_unused_action_count": _mean_or_none(
                tuple(float(item.unused_action_count) for item in diagnostics)
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
        for name, count in milestones.items():
            content[f"{name}_rate"] = count / len(records)
            content[f"{name}_episodes"] = count
        for action_index, name in enumerate(_ACTION_NAMES):
            content[f"mean_{name}_fraction"] = _mean_or_none(
                tuple(
                    item.action_counts[action_index] / sum(item.action_counts)
                    for item in diagnostics
                )
            )
        return Feedback(
            score=score,
            content=content,
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: MultiRoomConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/MultiRoom-v0/success-rate-v1",
        description=(
            "Explore a chain of partially observable rooms, open successive "
            "doors, and reach the green goal in the final room. Maximize "
            "success rate."
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
                        "view_y, and right increases view_x."
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
            "meaning": _ACTION_MEANING,
        },
        metadata={
            "environment": config.environment_id,
            "provider": "MiniGrid",
            "failure_return": 0.0,
            "partial_observability": True,
        },
        environment_parameters={
            "profile": config.profile,
            "minimum_rooms": config.minimum_rooms,
            "maximum_rooms": config.maximum_rooms,
            "required_connecting_doors": config.maximum_rooms - 1,
            "maximum_room_size": config.maximum_room_size,
            "grid_width": 25,
            "grid_height": 25,
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
            "object_encoding": {name: code for code, name in enumerate(_OBJECT_NAMES)},
            "color_encoding": {name: code for code, name in enumerate(_COLOR_NAMES)},
            "state_encoding": {name: code for code, name in enumerate(_STATE_NAMES)},
            "unused_actions": [3, 4, 6],
            "door_color_rule": (
                "adjacent connecting doors have different colors; colors may "
                "repeat later and do not identify physical doors"
            ),
            "success_reward_formula": "1 - 0.9*step_count/max_episode_steps",
            "non_success_reward": 0.0,
            "natural_termination": (
                "moving forward onto the final green goal terminates with success; "
                "opening or crossing a connecting door is intermediate"
            ),
            "time_limit": config.max_episode_steps,
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


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _reached_bool_milestone(record: EpisodeRecord, name: str) -> bool:
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
    terminal_reason = _string_metric(final, "terminal_reason")
    return terminal_reason if terminal_reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("MiniGrid MultiRoom diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        door_first_seen_step=_int_metric(final, "door_first_seen_step"),
        first_door_opened_step=_int_metric(final, "first_door_opened_step"),
        goal_first_seen_step=_int_metric(final, "goal_first_seen_step"),
        door_open_event_count=_int_metric(final, "door_open_event_count"),
        door_close_event_count=_int_metric(final, "door_close_event_count"),
        door_crossing_event_count=_int_metric(final, "door_crossing_event_count"),
        unique_observation_count=_int_metric(final, "unique_observation_count"),
        observation_novelty_step_fraction=_float_metric(final, "observation_novelty_step_fraction"),
        ineffective_action_fraction=_float_metric(final, "ineffective_action_fraction"),
        failed_toggle_count=_int_metric(final, "failed_toggle_count"),
        blocked_forward_count=_int_metric(final, "blocked_forward_count"),
        unused_action_count=_int_metric(final, "unused_action_count"),
        action_counts=tuple(_int_metric(final, f"{name}_count") for name in _ACTION_NAMES),
        front_object_before_action=_string_metric(final, "front_object_before_action"),
        task_stage=_string_metric(final, "task_stage"),
        outcome=_episode_outcome(record),
    )


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_milestone(values: tuple[int, ...]) -> float | None:
    reached = tuple(value for value in values if value >= 0)
    return statistics.fmean(reached) if reached else None


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        selected_steps = _trace_indices(record.steps)
        diagnostics = (
            _episode_diagnostics(record)
            if record.policy_failure is None and record.transitions
            else None
        )
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "return": (record.total_reward if record.policy_failure is None else 0.0),
                    "door_open_event_count": (
                        diagnostics.door_open_event_count if diagnostics else 0
                    ),
                    "door_close_event_count": (
                        diagnostics.door_close_event_count if diagnostics else 0
                    ),
                    "door_crossing_event_count": (
                        diagnostics.door_crossing_event_count if diagnostics else 0
                    ),
                    "goal_found": _reached_bool_milestone(record, "goal_found"),
                    "outcome": _episode_outcome(record),
                    "success": _successful(record),
                    "truncated": _truncated(record),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_observation(record.initial_observation),
                    "traced_steps": len(selected_steps),
                    "omitted_steps": record.steps - len(selected_steps),
                }
            )
        )
        for step_index in selected_steps:
            transition = record.transitions[step_index]
            if type(transition.action) is not int or not 0 <= transition.action <= 6:
                raise ValueError("MiniGrid MultiRoom trace Action is invalid")
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "action": transition.action,
                        "action_meaning": _ACTION_MEANING[str(transition.action)],
                        "reward": transition.step.reward,
                        "next_observation": _trace_observation(transition.step.observation),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                        "metrics": _trace_metrics(transition.step.metrics),
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
        raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    for name in _BOOL_METRICS:
        if type(metrics[name]) is not bool:
            raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    for name in _INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < 0:
            raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    for name in _MILESTONE_INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < -1:
            raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    for name in _FLOAT_METRICS:
        value = metrics[name]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    if type(metrics["front_object"]) is not str:
        raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    if type(metrics["front_object_before_action"]) is not str:
        raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    if type(metrics["task_stage"]) is not str:
        raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    if type(metrics["terminal_reason"]) is not str:
        raise ValueError("MiniGrid MultiRoom trace metrics are invalid")
    return dict(metrics)


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"MiniGrid MultiRoom metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"MiniGrid MultiRoom metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"MiniGrid MultiRoom metric {name} is invalid")
    return value


def _trace_observation(observation: PolicyValue) -> dict[str, PolicyValue]:
    if type(observation) is not dict or set(observation) != {
        "image",
        "direction",
        "mission",
    }:
        raise ValueError("MiniGrid MultiRoom trace observation is invalid")
    image = observation["image"]
    direction = observation["direction"]
    mission = observation["mission"]
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
    ):
        raise ValueError("MiniGrid MultiRoom trace image is invalid")
    if type(direction) is not int or not 0 <= direction <= 3:
        raise ValueError("MiniGrid MultiRoom trace direction is invalid")
    if type(mission) is not str or mission != _MISSION:
        raise ValueError("MiniGrid MultiRoom trace mission is invalid")

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
                raise ValueError("MiniGrid MultiRoom trace image codes are invalid")
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


__all__ = ["MultiRoomBenchmark"]
