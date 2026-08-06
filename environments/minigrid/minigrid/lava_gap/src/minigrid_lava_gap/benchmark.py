"""MiniGrid LavaGap with deterministic plans and safety traces."""

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

from .config import LavaGapConfig
from .environment import LavaGapEnvironment

_SEED_DOMAIN = b"evopolicygym-minigrid-lava_gap/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_TRACE_PREFIX_STEPS = 128
_TRACE_SUFFIX_STEPS = 32
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
        "goal_visible",
        "goal_found",
        "goal_first_seen_step",
        "hazard_visible",
        "hazard_found",
        "hazard_first_seen_step",
        "observation_novel",
        "unique_observation_count",
        "observation_novelty_step_fraction",
        "ineffective_action",
        "ineffective_action_fraction",
        "success_reward_at_this_step",
        "cumulative_return",
        "hazard_entered",
        "success",
        "terminal_reason",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    goal_first_seen_step: int
    hazard_first_seen_step: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    action_counts: tuple[int, ...]
    outcome: str


class LavaGapBenchmark:
    """Mean upstream Episode return for this Benchmark."""

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
            EpisodeSpec(environment_seed=_seed(split, seed, index)) for index in range(count)
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
        success_rate = successes / len(records)
        mean_return = statistics.fmean(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        goal_found = sum(_reached(r, "goal_found") for r in records)
        hazard_found = sum(_reached(r, "hazard_found") for r in records)
        hazards = sum(_reached(r, "hazard_entered") for r in records)
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        traced = records[:4]
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Reached the goal in {successes}/{len(records)} Episodes "
                f"({success_rate:.3f} success rate); {hazard_found} saw lava "
                f"and {hazards} entered it."
            ),
            "success_rate": success_rate,
            "goal_found_rate": goal_found / len(records),
            "hazard_found_rate": hazard_found / len(records),
            "hazard_entry_rate": hazards / len(records),
            "hazard_rate": hazards / len(records),
            "mean_return": statistics.fmean(
                r.total_reward if r.policy_failure is None else 0.0 for r in records
            ),
            "mean_steps": statistics.fmean(r.steps for r in records),
            "mean_goal_first_seen_step": _mean_milestone(
                tuple(item.goal_first_seen_step for item in diagnostics)
            ),
            "mean_hazard_first_seen_step": _mean_milestone(
                tuple(item.hazard_first_seen_step for item in diagnostics)
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
            "episodes_goal_found_but_not_reached": sum(
                _reached(record, "goal_found") and not _success(record) for record in records
            ),
            "episodes": len(records),
            "successful_episodes": successes,
            "hazard_found_episodes": hazard_found,
            "hazard_episodes": hazards,
            "truncated_episodes": sum(_truncated(r) for r in records),
            "policy_failures": sum(r.policy_failure is not None for r in records),
            "traced_episodes": len(traced),
            "trace_episodes_omitted": len(records) - len(traced),
            "trace_prefix_steps": _TRACE_PREFIX_STEPS,
            "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
        }
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


def _spec(config: LavaGapConfig) -> BenchmarkSpec:
    mission = "avoid the lava and get to the green goal square"
    return BenchmarkSpec(
        id="minigrid/LavaGap-v0/mean-return-v1",
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
            "mission": mission,
            "object_encoding": {name: code for code, name in enumerate(_OBJECTS)},
            "color_encoding": {name: code for code, name in enumerate(_COLORS)},
            "state_encoding": {name: code for code, name in enumerate(_STATES)},
            "success_reward_formula": ("1 - 0.9*step_count/max_episode_steps"),
            "non_success_reward": 0.0,
            "natural_termination": "enter goal cell or enter lava",
            "time_limit": config.max_episode_steps,
        },
        max_episode_steps=config.max_episode_steps,
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
    metrics = _trace_metrics(record.transitions[-1].step.metrics)
    reason = metrics["terminal_reason"]
    if type(reason) is not str:
        raise ValueError("MiniGrid LavaGap terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("MiniGrid LavaGap diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        goal_first_seen_step=_int_metric(final, "goal_first_seen_step"),
        hazard_first_seen_step=_int_metric(final, "hazard_first_seen_step"),
        unique_observation_count=_int_metric(final, "unique_observation_count"),
        observation_novelty_step_fraction=_float_metric(
            final,
            "observation_novelty_step_fraction",
        ),
        ineffective_action_fraction=_float_metric(
            final,
            "ineffective_action_fraction",
        ),
        action_counts=tuple(_int_metric(final, f"{name}_count") for name in _ACTION_NAMES),
        outcome=_episode_outcome(record),
    )


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"MiniGrid LavaGap metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"MiniGrid LavaGap metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_milestone(values: tuple[int, ...]) -> float | None:
    reached = tuple(value for value in values if value >= 0)
    return statistics.fmean(reached) if reached else None


def _trace(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        indices = _trace_indices(record.steps)
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
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "return": (record.total_reward if record.policy_failure is None else 0.0),
                    "success": _success(record),
                    "outcome": (
                        diagnostics.outcome if diagnostics is not None else _episode_outcome(record)
                    ),
                    "goal_found": _reached(record, "goal_found"),
                    "hazard_found": _reached(record, "hazard_found"),
                    "hazard_entered": _reached(record, "hazard_entered"),
                    "goal_first_seen_step": (
                        diagnostics.goal_first_seen_step if diagnostics is not None else None
                    ),
                    "hazard_first_seen_step": (
                        diagnostics.hazard_first_seen_step if diagnostics is not None else None
                    ),
                    "unique_observation_count": (
                        diagnostics.unique_observation_count if diagnostics is not None else None
                    ),
                    "observation_novelty_step_fraction": (
                        diagnostics.observation_novelty_step_fraction
                        if diagnostics is not None
                        else None
                    ),
                    "ineffective_action_fraction": (
                        diagnostics.ineffective_action_fraction if diagnostics is not None else None
                    ),
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
                raise ValueError("MiniGrid LavaGap trace Action is invalid")
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
    if type(metrics) is not dict or set(metrics) != set(_METRICS):
        raise ValueError("MiniGrid LavaGap trace metrics are invalid")
    boolean_fields = {
        "goal_visible",
        "goal_found",
        "hazard_visible",
        "hazard_found",
        "observation_novel",
        "ineffective_action",
        "hazard_entered",
        "success",
    }
    integer_fields = {
        "step_count",
        "remaining_steps",
        "goal_first_seen_step",
        "hazard_first_seen_step",
        "unique_observation_count",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
    traced: dict[str, PolicyValue] = {}
    for key in _METRICS:
        value = metrics[key]
        if key in boolean_fields:
            if type(value) is not bool:
                raise ValueError("MiniGrid LavaGap trace metrics are invalid")
        elif key in integer_fields:
            if type(value) is not int:
                raise ValueError("MiniGrid LavaGap trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("MiniGrid LavaGap trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("MiniGrid LavaGap trace metrics are invalid")
        traced[key] = value
    return traced


def _trace_observation(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise ValueError("MiniGrid LavaGap trace observation is invalid")
    image = value.get("image")
    direction = value.get("direction")
    mission = value.get("mission")
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
        or type(direction) is not int
        or not 0 <= direction <= 3
        or type(mission) is not str
        or mission != "avoid the lava and get to the green goal square"
    ):
        raise ValueError("MiniGrid LavaGap trace observation is invalid")
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
                raise ValueError("MiniGrid trace image codes are invalid")
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


__all__ = ["LavaGapBenchmark"]
