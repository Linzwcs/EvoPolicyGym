"""BlockedUnlockPickup with deterministic plans and milestone traces."""

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

from .environment import BlockedUnlockPickupEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-minigrid-blocked-unlock-pickup/episode-seed/v1\0"
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
_ACTION_NAMES = tuple(str(_ACTION_MEANING[str(index)]) for index in range(7))
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "target_color",
        "target_type",
        "target_label",
        "target_visible",
        "target_found",
        "target_first_seen_step",
        "target_in_front",
        "target_in_front_before_action",
        "target_destroyed_this_step",
        "target_destroyed",
        "target_destroyed_step",
        "blocker_visible",
        "blocker_found",
        "blocker_first_seen_step",
        "blocker_picked_up_this_step",
        "blocker_moved",
        "blocker_moved_step",
        "blocker_dropped_this_step",
        "blocker_dropped",
        "blocker_dropped_step",
        "key_visible",
        "key_found",
        "key_first_seen_step",
        "key_picked_up_this_step",
        "key_picked_up",
        "key_picked_up_step",
        "key_dropped_this_step",
        "key_dropped",
        "door_found",
        "door_first_seen_step",
        "door_color_found",
        "locked_door_visible",
        "open_door_visible",
        "door_opened_this_step",
        "door_opened",
        "door_opened_step",
        "visible_task_object_count",
        "front_object",
        "front_object_before_action",
        "carried_object",
        "carried_object_before_action",
        "pickup_attempt",
        "pickup_succeeded",
        "picked_up_label",
        "pickup_attempt_count",
        "failed_pickup",
        "failed_pickup_count",
        "drop_attempt",
        "drop_succeeded",
        "dropped_label",
        "drop_attempt_count",
        "failed_drop",
        "failed_drop_count",
        "toggle_attempt",
        "toggle_effective",
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
    "blocker_found",
    "blocker_moved",
    "blocker_dropped",
    "key_found",
    "key_picked_up",
    "key_dropped",
    "door_found",
    "door_opened",
    "target_found",
    "target_destroyed",
    "failed_pickup",
    "failed_drop",
    "failed_toggle",
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    blocker_first_seen_step: int
    blocker_moved_step: int
    blocker_dropped_step: int
    key_first_seen_step: int
    key_picked_up_step: int
    door_first_seen_step: int
    door_opened_step: int
    target_first_seen_step: int
    target_destroyed_step: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    failed_pickup_count: int
    failed_drop_count: int
    failed_toggle_count: int
    action_counts: tuple[int, ...]
    front_object_before_action: str
    carried_object: str
    task_stage: str
    outcome: str


class BlockedUnlockPickupBenchmark:
    """Mean upstream Episode return for this Benchmark."""

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
        success_rate = successes / len(records)
        counts = {name: _milestone_count(records, name) for name in _MILESTONES}
        mean_return = statistics.fmean(
            record.total_reward if record.policy_failure is None else 0.0 for record in records
        )
        mean_steps = statistics.fmean(record.steps for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        truncations = sum(_truncated(record) for record in records)
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        traced = records[:_MAX_TRACED_EPISODES]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Moved the blocker, unlocked the room, and collected the "
                f"target in {successes}/{len(records)} Episodes "
                f"({success_rate:.3f} success rate); "
                f"{counts['target_destroyed']} destroyed the target box."
            ),
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful) if successful else None
            ),
            "mean_blocker_first_seen_step": _mean_milestone(
                tuple(item.blocker_first_seen_step for item in diagnostics)
            ),
            "mean_blocker_moved_step": _mean_milestone(
                tuple(item.blocker_moved_step for item in diagnostics)
            ),
            "mean_blocker_dropped_step": _mean_milestone(
                tuple(item.blocker_dropped_step for item in diagnostics)
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
            "mean_target_first_seen_step": _mean_milestone(
                tuple(item.target_first_seen_step for item in diagnostics)
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
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/BlockedUnlockPickup-v0/mean-return-v1",
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
            "mission_template": "pick up the {color} box",
            "object_encoding": {name: code for code, name in enumerate(_OBJECT_NAMES)},
            "color_encoding": {name: code for code, name in enumerate(_COLOR_NAMES)},
            "state_encoding": {name: code for code, name in enumerate(_STATE_NAMES)},
            "success_reward_formula": ("1 - 0.9*step_count/max_episode_steps"),
            "non_success_reward": 0.0,
            "natural_termination": (
                "only picking up the mission box terminates; toggling the "
                "mission box destroys it without terminating; timeout "
                "truncates the Episode"
            ),
            "time_limit": 16 * 6**2,
        },
        max_episode_steps=16 * 6**2,
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
    return sum(_reached_milestone(record, name) for record in records)


def _reached_milestone(record: EpisodeRecord, name: str) -> bool:
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
    if reason != "none":
        return reason
    if _boolean_metric(final, "target_destroyed"):
        return "target_destroyed"
    return "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("BlockedUnlockPickup diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        blocker_first_seen_step=_int_metric(
            final,
            "blocker_first_seen_step",
        ),
        blocker_moved_step=_int_metric(final, "blocker_moved_step"),
        blocker_dropped_step=_int_metric(final, "blocker_dropped_step"),
        key_first_seen_step=_int_metric(final, "key_first_seen_step"),
        key_picked_up_step=_int_metric(final, "key_picked_up_step"),
        door_first_seen_step=_int_metric(final, "door_first_seen_step"),
        door_opened_step=_int_metric(final, "door_opened_step"),
        target_first_seen_step=_int_metric(final, "target_first_seen_step"),
        target_destroyed_step=_int_metric(final, "target_destroyed_step"),
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
        front_object_before_action=_string_metric(
            final,
            "front_object_before_action",
        ),
        carried_object=_string_metric(final, "carried_object"),
        task_stage=_string_metric(final, "task_stage"),
        outcome=_episode_outcome(record),
    )


def _boolean_metric(metrics: dict[str, PolicyValue], name: str) -> bool:
    value = metrics.get(name)
    if type(value) is not bool:
        raise ValueError(f"BlockedUnlockPickup metric {name} is invalid")
    return value


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"BlockedUnlockPickup metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"BlockedUnlockPickup metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"BlockedUnlockPickup metric {name} is invalid")
    return value


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
                    "blocker_found": _reached_milestone(
                        record,
                        "blocker_found",
                    ),
                    "blocker_moved": _reached_milestone(
                        record,
                        "blocker_moved",
                    ),
                    "blocker_dropped": _reached_milestone(
                        record,
                        "blocker_dropped",
                    ),
                    "key_found": _reached_milestone(record, "key_found"),
                    "key_picked_up": _reached_milestone(
                        record,
                        "key_picked_up",
                    ),
                    "door_found": _reached_milestone(record, "door_found"),
                    "door_opened": _reached_milestone(
                        record,
                        "door_opened",
                    ),
                    "target_found": _reached_milestone(
                        record,
                        "target_found",
                    ),
                    "target_destroyed": _reached_milestone(
                        record,
                        "target_destroyed",
                    ),
                    "success": _successful(record),
                    "outcome": (
                        diagnostics.outcome if diagnostics is not None else _episode_outcome(record)
                    ),
                    "task_stage": (diagnostics.task_stage if diagnostics is not None else None),
                    "blocker_moved_step": (
                        diagnostics.blocker_moved_step if diagnostics is not None else None
                    ),
                    "blocker_dropped_step": (
                        diagnostics.blocker_dropped_step if diagnostics is not None else None
                    ),
                    "key_picked_up_step": (
                        diagnostics.key_picked_up_step if diagnostics is not None else None
                    ),
                    "door_opened_step": (
                        diagnostics.door_opened_step if diagnostics is not None else None
                    ),
                    "target_first_seen_step": (
                        diagnostics.target_first_seen_step if diagnostics is not None else None
                    ),
                    "target_destroyed_step": (
                        diagnostics.target_destroyed_step if diagnostics is not None else None
                    ),
                    "front_object_before_action": (
                        diagnostics.front_object_before_action if diagnostics is not None else None
                    ),
                    "carried_object": (
                        diagnostics.carried_object if diagnostics is not None else None
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
                raise ValueError("BlockedUnlockPickup trace Action is invalid")
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
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("BlockedUnlockPickup trace metrics are invalid")
    boolean_fields = {
        "target_visible",
        "target_found",
        "target_in_front",
        "target_in_front_before_action",
        "target_destroyed_this_step",
        "target_destroyed",
        "blocker_visible",
        "blocker_found",
        "blocker_picked_up_this_step",
        "blocker_moved",
        "blocker_dropped_this_step",
        "blocker_dropped",
        "key_visible",
        "key_found",
        "key_picked_up_this_step",
        "key_picked_up",
        "key_dropped_this_step",
        "key_dropped",
        "door_found",
        "locked_door_visible",
        "open_door_visible",
        "door_opened_this_step",
        "door_opened",
        "pickup_attempt",
        "pickup_succeeded",
        "failed_pickup",
        "drop_attempt",
        "drop_succeeded",
        "failed_drop",
        "toggle_attempt",
        "toggle_effective",
        "failed_toggle",
        "observation_novel",
        "ineffective_action",
        "success",
    }
    integer_fields = {
        "step_count",
        "remaining_steps",
        "target_first_seen_step",
        "target_destroyed_step",
        "blocker_first_seen_step",
        "blocker_moved_step",
        "blocker_dropped_step",
        "key_first_seen_step",
        "key_picked_up_step",
        "door_first_seen_step",
        "door_opened_step",
        "visible_task_object_count",
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
        "target_color",
        "target_type",
        "target_label",
        "door_color_found",
        "front_object",
        "front_object_before_action",
        "carried_object",
        "carried_object_before_action",
        "picked_up_label",
        "dropped_label",
        "task_stage",
        "terminal_reason",
    }
    traced: dict[str, PolicyValue] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in boolean_fields:
            if type(value) is not bool:
                raise ValueError("BlockedUnlockPickup trace metrics are invalid")
        elif key in integer_fields:
            if type(value) is not int:
                raise ValueError("BlockedUnlockPickup trace metrics are invalid")
        elif key in string_fields:
            if type(value) is not str:
                raise ValueError("BlockedUnlockPickup trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("BlockedUnlockPickup trace metrics are invalid")
        traced[key] = value
    return traced


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
                raise ValueError("BlockedUnlockPickup trace image codes are invalid")
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
