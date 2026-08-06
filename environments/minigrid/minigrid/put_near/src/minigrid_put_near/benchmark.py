"""MiniGrid PutNear with deterministic plans and actionable placement traces."""

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

from .config import PutNearConfig
from .environment import PutNearEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-minigrid-put-near/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
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
_DESCRIPTORS = tuple(f"{color} {kind}" for color in _COLOR_NAMES for kind in ("key", "ball", "box"))
_MISSIONS = tuple(
    f"put the {move} near the {target}"
    for move in _DESCRIPTORS
    for target in _DESCRIPTORS
    if move != target
)
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
_LABEL_METRICS = frozenset(
    {
        "move_object_label",
        "target_object_label",
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
        "move_object_visible",
        "move_object_found",
        "move_object_in_front",
        "move_object_in_front_before_action",
        "target_object_visible",
        "target_object_found",
        "target_object_in_front",
        "carrying_move_object",
        "correct_object_picked_up_this_step",
        "correct_object_picked_up",
        "wrong_object_picked_up",
        "front_cell_empty",
        "front_cell_near_target",
        "valid_success_drop_available",
        "valid_success_drop_before_action",
        "terminal_drop_attempt",
        "object_dropped_this_step",
        "misplaced_drop",
        "blocked_terminal_drop",
        "box_destroyed_this_step",
        "move_object_destroyed_this_step",
        "move_object_destroyed",
        "target_object_destroyed_this_step",
        "target_object_destroyed",
        "mission_object_destroyed",
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
        "visible_portable_object_count",
        "box_destroy_event_count",
        "pickup_attempt_count",
        "pickup_event_count",
        "failed_pickup_count",
        "drop_attempt_count",
        "drop_event_count",
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
        "move_object_first_seen_step",
        "target_object_first_seen_step",
        "correct_object_pickup_step",
        "mission_object_destroyed_step",
    }
)
_FLOAT_METRICS = frozenset(
    {
        "observation_novelty_step_fraction",
        "ineffective_action_fraction",
        "success_reward_at_this_step",
        "cumulative_return",
    }
)
_METRICS = _LABEL_METRICS | _BOOL_METRICS | _INT_METRICS | _MILESTONE_INT_METRICS | _FLOAT_METRICS
_MILESTONES = (
    "move_object_found",
    "target_object_found",
    "correct_object_picked_up",
    "wrong_object_picked_up",
    "misplaced_drop",
    "blocked_terminal_drop",
    "mission_object_destroyed",
    "failed_pickup",
    "failed_drop",
    "failed_toggle",
    "blocked_forward",
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    move_object_first_seen_step: int
    target_object_first_seen_step: int
    correct_object_pickup_step: int
    mission_object_destroyed_step: int
    box_destroy_event_count: int
    pickup_event_count: int
    drop_event_count: int
    failed_pickup_count: int
    failed_drop_count: int
    failed_toggle_count: int
    blocked_forward_count: int
    done_action_count: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    action_counts: tuple[int, ...]
    move_object_label: str
    target_object_label: str
    carried_object: str
    task_stage: str
    outcome: str


class PutNearBenchmark:
    """Mean upstream Episode return for this Benchmark."""

    def __init__(self, config: PutNearConfig | None = None) -> None:
        if config is None:
            config = PutNearConfig()
        if type(config) is not PutNearConfig:
            raise TypeError("config must be PutNearConfig")
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
        return PutNearEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successful = tuple(record for record in records if _successful(record))
        successes = len(successful)
        success_rate = successes / len(records)
        mean_return = statistics.fmean(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        counts = {name: _milestone_count(records, name) for name in _MILESTONES}
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        traced = records[:_MAX_TRACED_EPISODES]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Placed the requested object near its target in "
                f"{successes}/{len(records)} Episodes ({success_rate:.3f} success rate); "
                f"{counts['wrong_object_picked_up']} wrong pickups, "
                f"{counts['misplaced_drop']} misplaced drops, and "
                f"{counts['blocked_terminal_drop']} blocked terminal drops."
            ),
            "success_rate": success_rate,
            "mean_return": statistics.fmean(
                record.total_reward if record.policy_failure is None else 0.0 for record in records
            ),
            "mean_steps": statistics.fmean(record.steps for record in records),
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful) if successful else None
            ),
            "mean_move_object_first_seen_step": _mean_milestone(
                tuple(item.move_object_first_seen_step for item in diagnostics)
            ),
            "mean_target_object_first_seen_step": _mean_milestone(
                tuple(item.target_object_first_seen_step for item in diagnostics)
            ),
            "mean_correct_object_pickup_step": _mean_milestone(
                tuple(item.correct_object_pickup_step for item in diagnostics)
            ),
            "mean_mission_object_destroyed_step": _mean_milestone(
                tuple(item.mission_object_destroyed_step for item in diagnostics)
            ),
            "mean_box_destroy_event_count": _mean_int(diagnostics, "box_destroy_event_count"),
            "mean_pickup_event_count": _mean_int(diagnostics, "pickup_event_count"),
            "mean_drop_event_count": _mean_int(diagnostics, "drop_event_count"),
            "mean_failed_pickup_count": _mean_int(diagnostics, "failed_pickup_count"),
            "mean_failed_drop_count": _mean_int(diagnostics, "failed_drop_count"),
            "mean_failed_toggle_count": _mean_int(diagnostics, "failed_toggle_count"),
            "mean_blocked_forward_count": _mean_int(diagnostics, "blocked_forward_count"),
            "mean_done_action_count": _mean_int(diagnostics, "done_action_count"),
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


def _benchmark_spec(config: PutNearConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/PutNear-v0/mean-return-v1",
        description=(
            "Parse a relational mission, pick up exactly the requested "
            "colored object, and drop it in an empty cell adjacent to the "
            "named target object. Maximize upstream Episode return."
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
            "environment": config.environment_id,
            "provider": "MiniGrid",
            "failure_return": 0.0,
            "partial_observability": True,
        },
        environment_parameters={
            "profile": config.profile,
            "size": config.size,
            "object_count": config.object_count,
            "see_through_walls": True,
            "initial_objects_are_non_adjacent": True,
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
            "mission_template": (
                "put the {move_color} {move_type} near the {target_color} {target_type}"
            ),
            "object_types": ["key", "ball", "box"],
            "object_encoding": {name: code for code, name in enumerate(_OBJECT_NAMES)},
            "color_encoding": {name: code for code, name in enumerate(_COLOR_NAMES)},
            "state_encoding": {name: code for code, name in enumerate(_STATE_NAMES)},
            "action_notes": {
                "pick_up": (
                    "picking up a non-mission object terminates immediately with zero reward"
                ),
                "drop": (
                    "any drop attempted while carrying terminates; success "
                    "requires an actual drop into an empty front cell within "
                    "Chebyshev distance one of the named target"
                ),
                "toggle": (
                    "documented upstream as unused, but toggling an empty box "
                    "destroys it without terminating"
                ),
                "done": "unused no-op",
            },
            "success_reward_formula": "1 - 0.9*step_count/max_episode_steps",
            "non_success_reward": 0.0,
            "natural_termination": (
                "wrong-object pickup terminates with failure; every drop while "
                "carrying terminates, succeeding only when the requested object "
                "is actually dropped adjacent to the named target"
            ),
            "time_limit": config.max_episode_steps,
        },
        max_episode_steps=config.max_episode_steps,
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


def _milestone_count(records: Sequence[EpisodeRecord], name: str) -> int:
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
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("MiniGrid PutNear diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        move_object_first_seen_step=_int_metric(final, "move_object_first_seen_step"),
        target_object_first_seen_step=_int_metric(final, "target_object_first_seen_step"),
        correct_object_pickup_step=_int_metric(final, "correct_object_pickup_step"),
        mission_object_destroyed_step=_int_metric(final, "mission_object_destroyed_step"),
        box_destroy_event_count=_int_metric(final, "box_destroy_event_count"),
        pickup_event_count=_int_metric(final, "pickup_event_count"),
        drop_event_count=_int_metric(final, "drop_event_count"),
        failed_pickup_count=_int_metric(final, "failed_pickup_count"),
        failed_drop_count=_int_metric(final, "failed_drop_count"),
        failed_toggle_count=_int_metric(final, "failed_toggle_count"),
        blocked_forward_count=_int_metric(final, "blocked_forward_count"),
        done_action_count=_int_metric(final, "done_action_count"),
        unique_observation_count=_int_metric(final, "unique_observation_count"),
        observation_novelty_step_fraction=_float_metric(final, "observation_novelty_step_fraction"),
        ineffective_action_fraction=_float_metric(final, "ineffective_action_fraction"),
        action_counts=tuple(_int_metric(final, f"{name}_count") for name in _ACTION_NAMES),
        move_object_label=_string_metric(final, "move_object_label"),
        target_object_label=_string_metric(final, "target_object_label"),
        carried_object=_string_metric(final, "carried_object"),
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
                    "move_object": (diagnostics.move_object_label if diagnostics else None),
                    "target_object": (diagnostics.target_object_label if diagnostics else None),
                    "move_object_found": _reached_milestone(record, "move_object_found"),
                    "target_object_found": _reached_milestone(record, "target_object_found"),
                    "correct_object_picked_up": _reached_milestone(
                        record, "correct_object_picked_up"
                    ),
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
                raise ValueError("MiniGrid PutNear trace Action is invalid")
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
        raise ValueError("MiniGrid PutNear trace metrics are invalid")
    for name in _LABEL_METRICS:
        if type(metrics[name]) is not str:
            raise ValueError("MiniGrid PutNear trace metrics are invalid")
    for name in _BOOL_METRICS:
        if type(metrics[name]) is not bool:
            raise ValueError("MiniGrid PutNear trace metrics are invalid")
    for name in _INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < 0:
            raise ValueError("MiniGrid PutNear trace metrics are invalid")
    for name in _MILESTONE_INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < -1:
            raise ValueError("MiniGrid PutNear trace metrics are invalid")
    for name in _FLOAT_METRICS:
        value = metrics[name]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("MiniGrid PutNear trace metrics are invalid")
    return dict(metrics)


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"MiniGrid PutNear metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"MiniGrid PutNear metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"MiniGrid PutNear metric {name} is invalid")
    return value


def _trace_observation(observation: PolicyValue) -> dict[str, PolicyValue]:
    if type(observation) is not dict or set(observation) != {
        "image",
        "direction",
        "mission",
    }:
        raise ValueError("MiniGrid PutNear trace observation is invalid")
    image = observation["image"]
    direction = observation["direction"]
    mission = observation["mission"]
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
    ):
        raise ValueError("MiniGrid PutNear trace image is invalid")
    if type(direction) is not int or not 0 <= direction <= 3:
        raise ValueError("MiniGrid PutNear trace direction is invalid")
    if type(mission) is not str or mission not in _MISSIONS:
        raise ValueError("MiniGrid PutNear trace mission is invalid")
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
                raise ValueError("MiniGrid PutNear trace image codes are invalid")
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
    move_object, target_object = _mission_objects(mission)
    return {
        "direction": direction,
        "mission": mission,
        "move_object": move_object,
        "target_object": target_object,
        "grid_rows": rows,
        "visible_objects": visible_objects,
    }


def _mission_objects(mission: str) -> tuple[str, str]:
    prefix = "put the "
    separator = " near the "
    if not mission.startswith(prefix) or mission.count(separator) != 1:
        raise ValueError("MiniGrid PutNear trace mission is invalid")
    parts = mission.removeprefix(prefix).split(separator)
    if len(parts) != 2 or any(part not in _DESCRIPTORS for part in parts):
        raise ValueError("MiniGrid PutNear trace mission is invalid")
    return parts[0], parts[1]


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


__all__ = ["PutNearBenchmark"]
