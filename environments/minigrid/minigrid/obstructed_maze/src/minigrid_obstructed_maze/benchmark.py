"""ObstructedMaze with deterministic plans and actionable milestone traces."""

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

from .config import ObstructedMazeConfig
from .environment import ObstructedMazeEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-minigrid-obstructed-maze/episode-seed/v1\0"
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
_MISSION = "pick up the blue ball"
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
        "target_label",
        "blocker_label",
        "key_box_label",
        "carried_object",
        "carried_object_before_action",
        "front_object",
        "front_object_before_action",
        "task_stage",
        "terminal_reason",
    }
)
_LIST_METRICS = frozenset({"visible_key_labels", "visible_locked_door_labels"})
_BOOL_METRICS = frozenset(
    {
        "box_visible",
        "box_found",
        "box_opened_this_step",
        "box_opened",
        "blocker_visible",
        "blocker_found",
        "blocker_picked_up_this_step",
        "blocker_picked_up",
        "blocker_dropped_this_step",
        "blocker_relocated",
        "key_found",
        "key_picked_up_this_step",
        "key_picked_up",
        "key_dropped_this_step",
        "locked_door_found",
        "door_opened_this_step",
        "locked_door_opened_this_step",
        "locked_door_opened",
        "unlocked_door_opened_this_step",
        "door_closed_this_step",
        "door_crossed_this_step",
        "target_visible",
        "target_found",
        "target_in_front",
        "target_in_front_before_action",
        "target_picked_up_this_step",
        "matching_key_for_front_locked_door_carried",
        "pickup_attempt",
        "failed_pickup",
        "target_pickup_blocked_by_carried_object",
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
        "required_locked_door_count",
        "required_unlocked_door_count",
        "box_open_event_count",
        "blocker_pickup_event_count",
        "blocker_drop_event_count",
        "visible_key_count",
        "key_pickup_event_count",
        "key_drop_event_count",
        "visible_door_count",
        "visible_locked_door_count",
        "door_open_event_count",
        "locked_door_open_event_count",
        "unlocked_door_open_event_count",
        "door_close_event_count",
        "door_crossing_event_count",
        "pickup_attempt_count",
        "failed_pickup_count",
        "target_pickup_blocked_count",
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
        "box_first_seen_step",
        "first_box_opened_step",
        "blocker_first_seen_step",
        "first_blocker_pickup_step",
        "first_blocker_relocated_step",
        "key_first_seen_step",
        "first_key_pickup_step",
        "locked_door_first_seen_step",
        "first_locked_door_opened_step",
        "target_first_seen_step",
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
_METRICS = (
    _LABEL_METRICS
    | _LIST_METRICS
    | _BOOL_METRICS
    | _INT_METRICS
    | _MILESTONE_INT_METRICS
    | _FLOAT_METRICS
)
_MILESTONES = (
    "box_found",
    "box_opened",
    "blocker_found",
    "blocker_picked_up",
    "blocker_relocated",
    "key_found",
    "key_picked_up",
    "locked_door_found",
    "locked_door_opened",
    "target_found",
    "target_pickup_blocked_by_carried_object",
    "failed_pickup",
    "failed_drop",
    "failed_toggle",
    "blocked_forward",
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    box_first_seen_step: int
    first_box_opened_step: int
    blocker_first_seen_step: int
    first_blocker_pickup_step: int
    first_blocker_relocated_step: int
    key_first_seen_step: int
    first_key_pickup_step: int
    locked_door_first_seen_step: int
    first_locked_door_opened_step: int
    target_first_seen_step: int
    box_open_event_count: int
    blocker_pickup_event_count: int
    blocker_drop_event_count: int
    key_pickup_event_count: int
    locked_door_open_event_count: int
    unlocked_door_open_event_count: int
    door_crossing_event_count: int
    failed_pickup_count: int
    failed_drop_count: int
    failed_toggle_count: int
    target_pickup_blocked_count: int
    blocked_forward_count: int
    done_action_count: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    action_counts: tuple[int, ...]
    task_stage: str
    outcome: str


class ObstructedMazeBenchmark:
    """Mean upstream Episode return for this Benchmark."""

    def __init__(self, config: ObstructedMazeConfig | None = None) -> None:
        if config is None:
            config = ObstructedMazeConfig()
        if type(config) is not ObstructedMazeConfig:
            raise TypeError("config must be ObstructedMazeConfig")
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
        return ObstructedMazeEnvironment(episode, config=self._config)

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
                f"Collected the blue target ball in {successes}/{len(records)} "
                f"Episodes ({success_rate:.3f} success rate)."
            ),
            "success_rate": success_rate,
            "profile_requires_key_box_opening": self._config.key_in_box,
            "profile_requires_blocker_relocation": self._config.blocked,
            "upstream_key_blocker_overlap_possible": (self._config.key_blocker_overlap_possible),
            "mean_return": statistics.fmean(
                record.total_reward if record.policy_failure is None else 0.0 for record in records
            ),
            "mean_steps": statistics.fmean(record.steps for record in records),
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful) if successful else None
            ),
            "mean_box_open_event_count": _mean_int(diagnostics, "box_open_event_count"),
            "mean_blocker_pickup_event_count": _mean_int(diagnostics, "blocker_pickup_event_count"),
            "mean_blocker_drop_event_count": _mean_int(diagnostics, "blocker_drop_event_count"),
            "mean_key_pickup_event_count": _mean_int(diagnostics, "key_pickup_event_count"),
            "mean_locked_door_open_event_count": _mean_int(
                diagnostics, "locked_door_open_event_count"
            ),
            "mean_unlocked_door_open_event_count": _mean_int(
                diagnostics, "unlocked_door_open_event_count"
            ),
            "mean_door_crossing_event_count": _mean_int(diagnostics, "door_crossing_event_count"),
            "mean_failed_pickup_count": _mean_int(diagnostics, "failed_pickup_count"),
            "mean_failed_drop_count": _mean_int(diagnostics, "failed_drop_count"),
            "mean_failed_toggle_count": _mean_int(diagnostics, "failed_toggle_count"),
            "mean_target_pickup_blocked_count": _mean_int(
                diagnostics, "target_pickup_blocked_count"
            ),
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
        for field in (
            "box_first_seen_step",
            "first_box_opened_step",
            "blocker_first_seen_step",
            "first_blocker_pickup_step",
            "first_blocker_relocated_step",
            "key_first_seen_step",
            "first_key_pickup_step",
            "locked_door_first_seen_step",
            "first_locked_door_opened_step",
            "target_first_seen_step",
        ):
            content[f"mean_{field}"] = _mean_milestone(
                tuple(getattr(item, field) for item in diagnostics)
            )
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


def _benchmark_spec(config: ObstructedMazeConfig) -> BenchmarkSpec:
    generation_warning: PolicyValue = (
        "This legacy v0 registration can place a blocking ball over a key, "
        "making some generated tasks structurally unsolvable; prefer the "
        "corresponding v1 profile."
        if config.key_blocker_overlap_possible
        else None
    )
    return BenchmarkSpec(
        id="minigrid/ObstructedMaze-v0/mean-return-v1",
        description=(
            "Navigate a room maze, optionally open grey boxes containing "
            "keys and relocate green balls obstructing doors, unlock the "
            "needed rooms, and pick up the blue target ball."
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
            "room_size": 6,
            "num_rows": config.rows,
            "num_columns": config.columns,
            "num_quarters": config.quarters,
            "upstream_horizon_room_factor": config.horizon_rooms,
            "key_in_box": config.key_in_box,
            "doors_blocked_by_ball": config.blocked,
            "locked_door_count": config.locked_doors,
            "unlocked_door_count": config.unlocked_doors,
            "key_blocker_overlap_possible": config.key_blocker_overlap_possible,
            "generation_warning": generation_warning,
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
            "target_object": "blue_ball",
            "door_blocker_object": "green_ball",
            "key_container_object": "grey_box",
            "object_encoding": {name: code for code, name in enumerate(_OBJECT_NAMES)},
            "color_encoding": {name: code for code, name in enumerate(_COLOR_NAMES)},
            "state_encoding": {name: code for code, name in enumerate(_STATE_NAMES)},
            "action_notes": {
                "pick_up": "picks up a front object only when hands are empty",
                "drop": (
                    "drops the carried object into an empty front cell; required "
                    "to relocate blockers and free hands"
                ),
                "toggle": "opens key boxes and toggles doors",
                "done": "unused no-op",
            },
            "success_reward_formula": "1 - 0.9*step_count/max_episode_steps",
            "non_success_reward": 0.0,
            "natural_termination": (
                "picking up the blue mission ball terminates with success; "
                "opening boxes or doors and moving blockers are intermediate"
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
    return sum(
        any(
            type(transition.step.metrics) is dict and transition.step.metrics.get(name) is True
            for transition in record.transitions
        )
        for record in records
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
        raise ValueError("ObstructedMaze diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        box_first_seen_step=_int_metric(final, "box_first_seen_step"),
        first_box_opened_step=_int_metric(final, "first_box_opened_step"),
        blocker_first_seen_step=_int_metric(final, "blocker_first_seen_step"),
        first_blocker_pickup_step=_int_metric(final, "first_blocker_pickup_step"),
        first_blocker_relocated_step=_int_metric(final, "first_blocker_relocated_step"),
        key_first_seen_step=_int_metric(final, "key_first_seen_step"),
        first_key_pickup_step=_int_metric(final, "first_key_pickup_step"),
        locked_door_first_seen_step=_int_metric(final, "locked_door_first_seen_step"),
        first_locked_door_opened_step=_int_metric(final, "first_locked_door_opened_step"),
        target_first_seen_step=_int_metric(final, "target_first_seen_step"),
        box_open_event_count=_int_metric(final, "box_open_event_count"),
        blocker_pickup_event_count=_int_metric(final, "blocker_pickup_event_count"),
        blocker_drop_event_count=_int_metric(final, "blocker_drop_event_count"),
        key_pickup_event_count=_int_metric(final, "key_pickup_event_count"),
        locked_door_open_event_count=_int_metric(final, "locked_door_open_event_count"),
        unlocked_door_open_event_count=_int_metric(final, "unlocked_door_open_event_count"),
        door_crossing_event_count=_int_metric(final, "door_crossing_event_count"),
        failed_pickup_count=_int_metric(final, "failed_pickup_count"),
        failed_drop_count=_int_metric(final, "failed_drop_count"),
        failed_toggle_count=_int_metric(final, "failed_toggle_count"),
        target_pickup_blocked_count=_int_metric(final, "target_pickup_blocked_count"),
        blocked_forward_count=_int_metric(final, "blocked_forward_count"),
        done_action_count=_int_metric(final, "done_action_count"),
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
                    "outcome": _episode_outcome(record),
                    "box_open_event_count": (
                        diagnostics.box_open_event_count if diagnostics else 0
                    ),
                    "blocker_drop_event_count": (
                        diagnostics.blocker_drop_event_count if diagnostics else 0
                    ),
                    "key_pickup_event_count": (
                        diagnostics.key_pickup_event_count if diagnostics else 0
                    ),
                    "locked_door_open_event_count": (
                        diagnostics.locked_door_open_event_count if diagnostics else 0
                    ),
                    "target_found": _reached(record, "target_found"),
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
                raise ValueError("ObstructedMaze trace Action is invalid")
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
        raise ValueError("ObstructedMaze trace metrics are invalid")
    for name in _LABEL_METRICS:
        if type(metrics[name]) is not str:
            raise ValueError("ObstructedMaze trace metrics are invalid")
    for name in _LIST_METRICS:
        value = metrics[name]
        if type(value) is not list or any(type(item) is not str for item in value):
            raise ValueError("ObstructedMaze trace metrics are invalid")
    for name in _BOOL_METRICS:
        if type(metrics[name]) is not bool:
            raise ValueError("ObstructedMaze trace metrics are invalid")
    for name in _INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < 0:
            raise ValueError("ObstructedMaze trace metrics are invalid")
    for name in _MILESTONE_INT_METRICS:
        value = metrics[name]
        if type(value) is not int or value < -1:
            raise ValueError("ObstructedMaze trace metrics are invalid")
    for name in _FLOAT_METRICS:
        value = metrics[name]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("ObstructedMaze trace metrics are invalid")
    return dict(metrics)


def _reached(record: EpisodeRecord, name: str) -> bool:
    return any(
        type(transition.step.metrics) is dict and transition.step.metrics.get(name) is True
        for transition in record.transitions
    )


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"ObstructedMaze metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"ObstructedMaze metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"ObstructedMaze metric {name} is invalid")
    return value


def _trace_observation(observation: PolicyValue) -> dict[str, PolicyValue]:
    if type(observation) is not dict or set(observation) != {
        "image",
        "direction",
        "mission",
    }:
        raise ValueError("ObstructedMaze trace observation is invalid")
    image = observation["image"]
    direction = observation["direction"]
    mission = observation["mission"]
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
    ):
        raise ValueError("ObstructedMaze trace image is invalid")
    if type(direction) is not int or not 0 <= direction <= 3:
        raise ValueError("ObstructedMaze trace direction is invalid")
    if type(mission) is not str or mission != _MISSION:
        raise ValueError("ObstructedMaze trace mission is invalid")
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
                raise ValueError("ObstructedMaze trace image codes are invalid")
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


__all__ = ["ObstructedMazeBenchmark"]
