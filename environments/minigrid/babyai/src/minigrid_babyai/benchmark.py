"""BabyAI task families with deterministic plans and semantic traces."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from typing import cast

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue

from .config import BabyAIConfig
from .environment import BabyAIEnvironment

_SEED_DOMAIN = b"evopolicygym-minigrid-babyai/episode-seed/v1\0"
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
_METRICS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "front_object",
        "front_object_before_action",
        "carried_object",
        "carried_object_before_action",
        "visible_object_labels",
        "visible_object_count",
        "newly_discovered_object_labels",
        "discovered_object_labels",
        "discovered_object_label_count",
        "pickup_attempt",
        "object_picked_up_this_step",
        "failed_pickup",
        "pickup_attempt_count",
        "pickup_event_count",
        "failed_pickup_count",
        "first_pickup_step",
        "drop_attempt",
        "object_dropped_this_step",
        "failed_drop",
        "drop_attempt_count",
        "drop_event_count",
        "failed_drop_count",
        "first_drop_step",
        "toggle_attempt",
        "toggle_effective",
        "door_opened_this_step",
        "door_closed_this_step",
        "box_opened_this_step",
        "failed_toggle",
        "toggle_attempt_count",
        "door_open_event_count",
        "door_close_event_count",
        "box_open_event_count",
        "failed_toggle_count",
        "first_door_open_step",
        "first_box_open_step",
        "blocked_forward",
        "blocked_forward_count",
        "done_action",
        "done_action_count",
        "observation_novel",
        "unique_observation_count",
        "observation_novelty_step_fraction",
        "ineffective_action",
        "ineffective_action_fraction",
        "instruction_failure",
        "task_stage",
        "success_reward_at_this_step",
        "cumulative_return",
        "success",
        "terminal_reason",
        *(f"{name}_count" for name in _ACTIONS.values()),
    }
)
_EVENT_METRICS = (
    "object_picked_up_this_step",
    "object_dropped_this_step",
    "door_opened_this_step",
    "door_closed_this_step",
    "box_opened_this_step",
    "failed_pickup",
    "failed_drop",
    "failed_toggle",
    "blocked_forward",
    "instruction_failure",
)
_FINAL_COUNT_METRICS = (
    "pickup_event_count",
    "drop_event_count",
    "door_open_event_count",
    "door_close_event_count",
    "box_open_event_count",
    "failed_pickup_count",
    "failed_drop_count",
    "failed_toggle_count",
    "blocked_forward_count",
    "done_action_count",
    "discovered_object_label_count",
    "unique_observation_count",
)


class BabyAIBenchmark:
    """Success rate for one Host-selected BabyAI task profile."""

    def __init__(self, config: BabyAIConfig | None = None) -> None:
        if config is None:
            config = BabyAIConfig()
        if type(config) is not BabyAIConfig:
            raise TypeError("config must be BabyAIConfig")
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
        return BabyAIEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")
        successful = tuple(record for record in records if _success(record))
        successes = len(successful)
        score = successes / len(records)
        traced = records[:4]
        event_counts = {
            name: sum(_reached(record, name) for record in records) for name in _EVENT_METRICS
        }
        final_metrics = tuple(
            metrics for record in records if (metrics := _final_metrics(record)) is not None
        )
        content: dict[str, PolicyValue] = {
            "summary": (
                f"Completed the instruction in {successes}/{len(records)} "
                f"Episodes ({score:.3f} success rate); "
                f"{event_counts['instruction_failure']} instruction failures "
                f"and {sum(_truncated(record) for record in records)} timeouts."
            ),
            "success_rate": score,
            "mean_return": statistics.fmean(
                record.total_reward if record.policy_failure is None else 0.0 for record in records
            ),
            "mean_steps": statistics.fmean(record.steps for record in records),
            "mean_steps_on_success": (
                statistics.fmean(record.steps for record in successful) if successful else None
            ),
            "mean_observation_novelty_step_fraction": _mean_final_float(
                final_metrics,
                "observation_novelty_step_fraction",
            ),
            "mean_ineffective_action_fraction": _mean_final_float(
                final_metrics,
                "ineffective_action_fraction",
            ),
            "episodes": len(records),
            "successful_episodes": successes,
            "instruction_failure_episodes": event_counts["instruction_failure"],
            "truncated_episodes": sum(_truncated(r) for r in records),
            "policy_failures": sum(r.policy_failure is not None for r in records),
            "traced_episodes": len(traced),
            "trace_episodes_omitted": len(records) - len(traced),
            "trace_prefix_steps": _TRACE_PREFIX_STEPS,
            "trace_suffix_steps": _TRACE_SUFFIX_STEPS,
        }
        for name, count in event_counts.items():
            if name == "instruction_failure":
                continue
            content[f"{name}_episodes"] = count
            content[f"{name}_rate"] = count / len(records)
        for name in _FINAL_COUNT_METRICS:
            content[f"mean_{name}"] = _mean_final_int(final_metrics, name)
        return Feedback(
            score=score,
            content=content,
            artifacts=(_trace(traced),),
        )


def _spec(config: BabyAIConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=f"minigrid/BabyAI-{config.family}-v0/success-rate-v1",
        description=(
            "Interpret the public BabyAI instruction and complete the selected task profile."
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
                "mission": {"type": "string"},
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
            "procedural_generation": True,
        },
        environment_parameters={
            "family": config.family,
            "profile": config.profile,
            "view_size": 7,
            "agent_view_position": [3, 6],
            "mission_source": "upstream generated instruction",
            "profile_max_episode_steps": config.max_episode_steps,
            "task_conditioned_horizon": (
                "The actual upstream horizon may be shorter than the profile "
                "maximum for a simpler generated composite instruction."
            ),
            "success_reward_formula": ("1 - 0.9*step_count/max_episode_steps"),
            "natural_termination": (
                "The selected non-debug profiles terminate only on verified "
                "instruction success; wrong or out-of-order actions normally "
                "continue, and the actual task horizon truncates separately."
            ),
            "action_notes": {
                "pick_up": (
                    "Acts on the front cell. A wrong pickup does not terminate "
                    "these non-debug profiles, but may obstruct later steps."
                ),
                "drop": (
                    "Acts on the front cell; PutNext instructions complete "
                    "only after the requested placement."
                ),
                "toggle": (
                    "Opens/closes doors or opens boxes; locked doors require "
                    "a carried key of the matching color."
                ),
                "done": (
                    "No-op in this upstream configuration; instructions are "
                    "verified directly after every action. The Host rejects "
                    "BABYAI_DONE_ACTIONS overrides to keep this stable."
                ),
            },
            "object_encoding": {name: code for code, name in enumerate(_OBJECTS)},
            "color_encoding": {name: code for code, name in enumerate(_COLORS)},
            "state_encoding": {name: code for code, name in enumerate(_STATES)},
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
        type(item.step.metrics) is dict and item.step.metrics.get(name) is True
        for item in record.transitions
    )


def _final_metrics(record: EpisodeRecord) -> dict[str, PolicyValue] | None:
    if record.policy_failure is not None or not record.transitions:
        return None
    value = record.transitions[-1].step.metrics
    if type(value) is not dict or set(value) != _METRICS:
        raise ValueError("MiniGrid BabyAI metrics are invalid")
    return value


def _mean_final_int(
    metrics: Sequence[dict[str, PolicyValue]],
    name: str,
) -> float | None:
    values = tuple(item[name] for item in metrics)
    if any(type(value) is not int for value in values):
        raise ValueError("MiniGrid BabyAI integer metric is invalid")
    return statistics.fmean(cast(int, value) for value in values) if values else None


def _mean_final_float(
    metrics: Sequence[dict[str, PolicyValue]],
    name: str,
) -> float | None:
    values = tuple(item[name] for item in metrics)
    if any(type(value) is not float for value in values):
        raise ValueError("MiniGrid BabyAI float metric is invalid")
    return statistics.fmean(cast(float, value) for value in values) if values else None


def _trace(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        lines.append(
            _json(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "steps": record.steps,
                    "return": (record.total_reward if record.policy_failure is None else 0.0),
                    "success": _success(record),
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
                raise ValueError("MiniGrid BabyAI trace Action is invalid")
            if type(item.step.metrics) is not dict or set(item.step.metrics) != _METRICS:
                raise ValueError("MiniGrid BabyAI trace metrics are invalid")
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
        raise ValueError("MiniGrid BabyAI trace observation is invalid")
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
        raise ValueError("MiniGrid BabyAI trace observation is invalid")
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


__all__ = ["BabyAIBenchmark"]
