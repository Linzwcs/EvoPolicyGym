"""MiniGrid GoToObject with deterministic plans and bounded semantic traces."""

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

from .config import GoToObjectConfig
from .environment import GoToObjectEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-minigrid-go_to_object/episode-seed/v1\0"
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
_MISSIONS = tuple(
    f"go to the {color} {kind}" for color in _COLOR_NAMES for kind in ("key", "ball", "box")
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
        "target_adjacent_visible",
        "target_adjacent_visible_before_action",
        "visible_candidate_count",
        "max_visible_candidate_count",
        "unique_candidate_count_found",
        "front_candidate_before_action",
        "adjacent_candidate_before_action",
        "adjacent_candidate_count_before_action",
        "completion_action",
        "completion_step",
        "premature_toggle",
        "wrong_done",
        "wrong_completion",
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


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    target_first_seen_step: int
    completion_step: int
    unique_candidate_count_found: int
    max_visible_candidate_count: int
    unique_observation_count: int
    observation_novelty_step_fraction: float
    ineffective_action_fraction: float
    action_counts: tuple[int, ...]
    adjacent_candidate_before_action: str
    task_stage: str
    outcome: str


class GoToObjectBenchmark:
    """Mean upstream Episode return for this Benchmark."""

    def __init__(self, config: GoToObjectConfig | None = None) -> None:
        if config is None:
            config = GoToObjectConfig()
        if type(config) is not GoToObjectConfig:
            raise TypeError("config must be GoToObjectConfig")
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
        return GoToObjectEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        successful = tuple(record for record in records if _successful(record))
        successes = len(successful)
        success_rate = successes / len(records)
        found_targets = _milestone_count(records, "target_found")
        wrong_completions = _milestone_count(records, "wrong_completion")
        premature_toggles = _milestone_count(records, "premature_toggle")
        wrong_done = _milestone_count(records, "wrong_done")
        mean_return = statistics.fmean(
            record.total_reward if record.policy_failure is None else 0.0 for record in records
        )
        mean_steps = statistics.fmean(record.steps for record in records)
        mean_success_steps: PolicyValue = (
            statistics.fmean(record.steps for record in successful) if successful else None
        )
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
                f"Declared completion beside the mission target in "
                f"{successes}/{len(records)} Episodes ({success_rate:.3f} success "
                f"rate); {premature_toggles} terminated with toggle and "
                f"{wrong_done} issued done from a wrong location."
            ),
            "success_rate": success_rate,
            "target_found_rate": found_targets / len(records),
            "wrong_completion_rate": wrong_completions / len(records),
            "premature_toggle_rate": premature_toggles / len(records),
            "wrong_done_rate": wrong_done / len(records),
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "mean_steps_on_success": mean_success_steps,
            "mean_target_first_seen_step": _mean_milestone(
                tuple(item.target_first_seen_step for item in diagnostics)
            ),
            "mean_completion_step": _mean_milestone(
                tuple(item.completion_step for item in diagnostics)
            ),
            "mean_unique_candidate_count_found": _mean_or_none(
                tuple(float(item.unique_candidate_count_found) for item in diagnostics)
            ),
            "mean_max_visible_candidate_count": _mean_or_none(
                tuple(float(item.max_visible_candidate_count) for item in diagnostics)
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
            "episodes_target_found_but_not_completed": sum(
                _reached_milestone(record, "target_found")
                and not _successful(record)
                and not _reached_milestone(record, "wrong_completion")
                for record in records
            ),
            "episodes": len(records),
            "successful_episodes": successes,
            "target_found_episodes": found_targets,
            "wrong_completion_episodes": wrong_completions,
            "premature_toggle_episodes": premature_toggles,
            "wrong_done_episodes": wrong_done,
            "truncated_episodes": truncations,
            "policy_failures": failures,
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
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: GoToObjectConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="minigrid/GoToObject-v0/mean-return-v1",
        description=(
            "Read the natural-language mission, explore the partially "
            "observable room, stand next to the matching colored key, ball, "
            "or box, and declare completion. Maximize upstream Episode return."
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
                "mission": {
                    "type": "string",
                    "values": list(_MISSIONS),
                },
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
            "mission_syntax": "go to the {color} {object}",
            "target_types": ["key", "ball", "box"],
            "object_encoding": {name: code for code, name in enumerate(_OBJECT_NAMES)},
            "color_encoding": {name: code for code, name in enumerate(_COLOR_NAMES)},
            "state_encoding": {name: code for code, name in enumerate(_STATE_NAMES)},
            "success_reward_formula": ("1 - 0.9*step_count/max_episode_steps"),
            "non_success_reward": 0.0,
            "natural_termination": (
                "toggle always terminates with zero; done always terminates "
                "and succeeds only when adjacent to the mission target"
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


def _milestone_count(
    records: Sequence[EpisodeRecord],
    name: str,
) -> int:
    return sum(_reached_milestone(record, name) for record in records)


def _reached_milestone(record: EpisodeRecord, name: str) -> bool:
    for transition in record.transitions:
        metrics = transition.step.metrics
        if type(metrics) is dict and metrics.get(name) is True:
            return True
    return False


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if not record.transitions:
        return "incomplete"
    metrics = _trace_metrics(record.transitions[-1].step.metrics)
    reason = metrics["terminal_reason"]
    if type(reason) is not str:
        raise ValueError("MiniGrid GoToObject terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("MiniGrid GoToObject diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    adjacent = _string_metric(final, "adjacent_candidate_before_action")
    stage = _string_metric(final, "task_stage")
    return _EpisodeDiagnostics(
        target_first_seen_step=_int_metric(final, "target_first_seen_step"),
        completion_step=_int_metric(final, "completion_step"),
        unique_candidate_count_found=_int_metric(
            final,
            "unique_candidate_count_found",
        ),
        max_visible_candidate_count=_int_metric(
            final,
            "max_visible_candidate_count",
        ),
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
        action_counts=tuple(_int_metric(final, f"{name}_count") for name in _ACTION_NAMES),
        adjacent_candidate_before_action=adjacent,
        task_stage=stage,
        outcome=_episode_outcome(record),
    )


def _int_metric(metrics: dict[str, PolicyValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError(f"MiniGrid GoToObject metric {name} is invalid")
    return value


def _float_metric(metrics: dict[str, PolicyValue], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"MiniGrid GoToObject metric {name} is invalid")
    return value


def _string_metric(metrics: dict[str, PolicyValue], name: str) -> str:
    value = metrics.get(name)
    if type(value) is not str:
        raise ValueError(f"MiniGrid GoToObject metric {name} is invalid")
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
                    "target_found": _reached_milestone(
                        record,
                        "target_found",
                    ),
                    "wrong_completion": _reached_milestone(
                        record,
                        "wrong_completion",
                    ),
                    "premature_toggle": _reached_milestone(
                        record,
                        "premature_toggle",
                    ),
                    "wrong_done": _reached_milestone(
                        record,
                        "wrong_done",
                    ),
                    "success": _successful(record),
                    "outcome": (
                        diagnostics.outcome if diagnostics is not None else _episode_outcome(record)
                    ),
                    "task_stage": (diagnostics.task_stage if diagnostics is not None else None),
                    "target_first_seen_step": (
                        diagnostics.target_first_seen_step if diagnostics is not None else None
                    ),
                    "completion_step": (
                        diagnostics.completion_step if diagnostics is not None else None
                    ),
                    "unique_candidate_count_found": (
                        diagnostics.unique_candidate_count_found
                        if diagnostics is not None
                        else None
                    ),
                    "adjacent_candidate_before_action": (
                        diagnostics.adjacent_candidate_before_action
                        if diagnostics is not None
                        else None
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
                raise ValueError("MiniGrid GoToObject trace Action is invalid")
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
        raise ValueError("MiniGrid GoToObject trace metrics are invalid")
    boolean_fields = {
        "target_visible",
        "target_found",
        "target_adjacent_visible",
        "target_adjacent_visible_before_action",
        "premature_toggle",
        "wrong_done",
        "wrong_completion",
        "observation_novel",
        "ineffective_action",
        "success",
    }
    integer_fields = {
        "step_count",
        "remaining_steps",
        "target_first_seen_step",
        "visible_candidate_count",
        "max_visible_candidate_count",
        "unique_candidate_count_found",
        "adjacent_candidate_count_before_action",
        "completion_step",
        "unique_observation_count",
        *(f"{name}_count" for name in _ACTION_NAMES),
    }
    string_fields = {
        "target_color",
        "target_type",
        "target_label",
        "front_candidate_before_action",
        "adjacent_candidate_before_action",
        "completion_action",
        "task_stage",
        "terminal_reason",
    }
    traced: dict[str, PolicyValue] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in boolean_fields:
            if type(value) is not bool:
                raise ValueError("MiniGrid GoToObject trace metrics are invalid")
        elif key in integer_fields:
            if type(value) is not int:
                raise ValueError("MiniGrid GoToObject trace metrics are invalid")
        elif key in string_fields:
            if type(value) is not str:
                raise ValueError("MiniGrid GoToObject trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("MiniGrid GoToObject trace metrics are invalid")
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
        raise ValueError("MiniGrid GoToObject trace observation is invalid")
    image = observation["image"]
    direction = observation["direction"]
    mission = observation["mission"]
    if (
        type(image) is not TensorValue
        or image.dtype != "uint8"
        or image.shape != (7, 7, 3)
        or len(image.data) != 147
    ):
        raise ValueError("MiniGrid GoToObject trace image is invalid")
    if type(direction) is not int or not 0 <= direction <= 3:
        raise ValueError("MiniGrid GoToObject trace direction is invalid")
    if type(mission) is not str or mission not in _MISSIONS:
        raise ValueError("MiniGrid GoToObject trace mission is invalid")

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
                raise ValueError("MiniGrid GoToObject trace image codes are invalid")
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
    target_color, target_type = _target(mission)
    return {
        "direction": direction,
        "mission": mission,
        "target_color": target_color,
        "target_type": target_type,
        "grid_rows": rows,
        "visible_objects": visible_objects,
    }


def _target(mission: str) -> tuple[str, str]:
    prefix = "go to the "
    if not mission.startswith(prefix):
        raise ValueError("MiniGrid GoToObject trace mission is invalid")
    remainder = mission.removeprefix(prefix)
    parts = remainder.split(" ")
    if len(parts) != 2 or parts[0] not in _COLOR_NAMES or parts[1] not in {"key", "ball", "box"}:
        raise ValueError("MiniGrid GoToObject trace mission is invalid")
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


__all__ = ["GoToObjectBenchmark"]
