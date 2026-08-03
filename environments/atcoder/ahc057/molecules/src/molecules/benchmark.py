"""The independent interactive AHC057 Molecules Benchmark."""

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
from evopolicygym.policy import PolicyValue, copy_policy_value

from .environment import MoleculesEnvironment
from .simulation import (
    GENERATOR_ID,
    POINTS,
    SPACE_SIZE,
    TARGET_COMPONENTS,
    TARGET_SIZE,
    TURNS,
)

_EPISODE_SEED_DOMAIN = b"evopolicygym-molecules/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_BOND_EVENTS = 256

_SPEC = BenchmarkSpec(
    id="atcoder/AHC057/Molecules/mean-log-cost-score-v1",
    description=(
        f"Bond {POINTS} moving points into {TARGET_COMPONENTS} components of "
        f"size {TARGET_SIZE} over {TURNS} turns. Maximize mean logarithmic "
        "distance-cost score."
    ),
    observation_space={
        "type": "object",
        "fields": {
            "turn": {"type": "integer", "minimum": 0, "maximum": TURNS},
            "turns_remaining": {"type": "integer", "minimum": 0, "maximum": TURNS},
            "positions": {"type": "float_matrix", "shape": [POINTS, 2]},
            "velocities": {"type": "float_matrix", "shape": [POINTS, 2]},
            "components": {"type": "integer_vector", "length": POINTS},
            "component_count": {
                "type": "integer",
                "minimum": TARGET_COMPONENTS,
                "maximum": POINTS,
            },
            "total_cost": {"type": "integer", "minimum": 0},
            "initial": {
                "one_of": ["task_constants", "null"],
                "notes": "Present only in the first observation.",
            },
        },
        "notes": "The Episode seed and split identity are never exposed.",
    },
    action_space={
        "type": "object",
        "fields": {
            "bonds": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": ["point_i", "point_j"],
                    "length": 2,
                },
                "max_items": POINTS - TARGET_COMPONENTS,
            },
        },
        "required": ["bonds"],
        "additional_fields": False,
        "notes": "Every bond must join two different current components; the turn is atomic.",
    },
    metadata={
        "environment": "Molecules",
        "provider": "AtCoder",
        "contest": "AtCoder Heuristic Contest 057",
        "task": "AHC057 A",
        "upstream_specification": ("https://atcoder.jp/contests/ahc057/tasks/ahc057_a"),
        "upstream_specification_revision": "2025-11-29",
        "upstream_tool_archive_revision": "BJTm8xSg",
        "upstream_tool_license": "not declared; not redistributed",
        "implementation": "independent",
        "upstream_code_included": False,
        "upstream_inputs_included": False,
        "upstream_assets_included": False,
        "failure_score": 0.0,
    },
    environment_parameters={
        "generator": GENERATOR_ID,
        "points": POINTS,
        "turns": TURNS,
        "space_size": SPACE_SIZE,
        "target_components": TARGET_COMPONENTS,
        "target_size": TARGET_SIZE,
        "reward_semantics": (
            "Reward is 0 for turns 1-999 and the complete official score is "
            "emitted exactly once on the mandatory 1000th turn."
        ),
        "bond_atomicity": (
            "Every submitted bond list is validated and applied atomically; "
            "cycles, oversize components, and invalid final partitions are rejected."
        ),
        "score_formula": (
            "round(1_000_000*log2(space_size*(points-target_components)/(total_cost+1)))"
        ),
    },
    max_episode_steps=TURNS,
    primary_metric="mean_log_cost_score",
    score_direction="maximize",
)


class MoleculesBenchmark:
    """Mean official logarithmic cost score over generated point systems."""

    @property
    def spec(self) -> BenchmarkSpec:
        return _SPEC

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
        return MoleculesEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        scores = tuple(_episode_score(record) for record in records)
        score = statistics.fmean(scores)
        completed = sum(record.policy_failure is None for record in records)
        costs = tuple(
            _final_integer(record, "total_cost")
            for record in records
            if record.policy_failure is None
        )
        trace, bond_events, omitted_events = _trace_artifact(records)
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean log cost score {score:.3f}; "
                    f"{completed}/{len(records)} target partitions completed."
                ),
                "mean_log_cost_score": score,
                "mean_total_cost": (statistics.fmean(costs) if costs else None),
                "mean_total_bonds": _mean_final_number(
                    records,
                    "total_bonds",
                ),
                "mean_cost_per_bond": _mean_final_number(
                    records,
                    "mean_cost_per_bond",
                ),
                "mean_maximum_bond_cost": _mean_final_number(
                    records,
                    "maximum_bond_cost",
                ),
                "mean_bond_action_count": _mean_final_number(
                    records,
                    "bond_action_count",
                ),
                "mean_empty_bond_action_fraction": _mean_final_number(
                    records,
                    "empty_bond_action_fraction",
                ),
                "mean_target_partition_first_ready_turn": _mean_final_number(
                    records,
                    "target_partition_first_ready_turn",
                    exclude_negative=True,
                ),
                "episodes": len(records),
                "completed": completed,
                "policy_failures": len(records) - completed,
                "failure_score": 0.0,
                "traced_episodes": min(len(records), _MAX_TRACED_EPISODES),
                "trace_episodes_omitted": max(
                    0,
                    len(records) - _MAX_TRACED_EPISODES,
                ),
                "bond_events": bond_events,
                "bond_events_omitted": omitted_events,
            },
            artifacts=(trace,),
        )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _episode_score(record: EpisodeRecord) -> float:
    if record.policy_failure is not None:
        return 0.0
    if len(record.transitions) != TURNS or not record.transitions[-1].step.terminated:
        raise ValueError("completed Molecules Episode evidence is invalid")
    return float(_final_integer(record, "final_score"))


def _final_integer(record: EpisodeRecord, name: str) -> int:
    if not record.transitions:
        raise ValueError("Molecules terminal evidence is missing")
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Molecules terminal metrics are invalid")
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError("Molecules terminal metrics are invalid")
    return value


def _final_number(
    record: EpisodeRecord,
    name: str,
) -> float | int | None:
    if record.policy_failure is not None or not record.transitions:
        return None
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Molecules terminal metrics are invalid")
    value = metrics.get(name)
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise ValueError("Molecules terminal metrics are invalid")
    return cast(float | int, value)


def _mean_final_number(
    records: Sequence[EpisodeRecord],
    name: str,
    *,
    exclude_negative: bool = False,
) -> float | None:
    values = tuple(
        value
        for record in records
        if (value := _final_number(record, name)) is not None
        and (not exclude_negative or value >= 0)
    )
    return statistics.fmean(values) if values else None


def _trace_artifact(
    records: Sequence[EpisodeRecord],
) -> tuple[Artifact, int, int]:
    lines: list[bytes] = []
    bond_events = 0
    total_bond_events = sum(_record_bond_events(record) for record in records)
    for episode_index, record in enumerate(records[:_MAX_TRACED_EPISODES]):
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "score": _episode_score(record),
                    "failure": record.policy_failure,
                    "initial_observation": _trace_initial(record.initial_observation),
                }
            )
        )
        for step_index, transition in enumerate(record.transitions):
            bonds = _trace_bonds(transition.action)
            if not bonds:
                continue
            if bond_events >= _MAX_BOND_EVENTS:
                break
            observation = _trace_state(transition.step.observation)
            metrics = transition.step.metrics
            if type(metrics) is not dict:
                raise ValueError("Molecules trace metrics are invalid")
            lines.append(
                _json_line(
                    {
                        "type": "bond_event",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "bonds": bonds,
                        "next_state": observation,
                        "action_cost": metrics.get("action_cost"),
                        "bond_count": metrics.get("bond_count_this_turn"),
                        "action_cost_per_bond": metrics.get("action_cost_per_bond"),
                        "minimum_bond_cost": metrics.get("minimum_bond_cost_this_turn"),
                        "maximum_bond_cost": metrics.get("maximum_bond_cost_this_turn"),
                        "total_cost": metrics.get("total_cost"),
                        "mean_cost_per_bond": metrics.get("mean_cost_per_bond"),
                        "score_upper_bound_if_no_further_cost": metrics.get(
                            "score_upper_bound_if_no_further_cost"
                        ),
                        "required_bonds_remaining": metrics.get("required_bonds_remaining"),
                        "component_size_histogram": metrics.get("component_size_histogram"),
                        "target_partition_ready": metrics.get("target_partition_ready"),
                    }
                )
            )
            bond_events += 1
    return (
        Artifact(
            name="trace.jsonl",
            media_type="application/x-ndjson",
            content=b"".join(lines),
        ),
        bond_events,
        total_bond_events - bond_events,
    )


def _record_bond_events(record: EpisodeRecord) -> int:
    return sum(bool(_trace_bonds(item.action)) for item in record.transitions)


def _trace_bonds(action: PolicyValue) -> list[list[int]]:
    if type(action) is not dict or set(action) != {"bonds"}:
        raise ValueError("Molecules trace Action is invalid")
    raw_bonds = action["bonds"]
    if type(raw_bonds) is not list:
        raise ValueError("Molecules trace Action is invalid")
    bonds: list[list[int]] = []
    for raw_bond in raw_bonds:
        if (
            type(raw_bond) is not list
            or len(raw_bond) != 2
            or any(type(value) is not int for value in raw_bond)
        ):
            raise ValueError("Molecules trace Action is invalid")
        bonds.append([cast(int, raw_bond[0]), cast(int, raw_bond[1])])
    return bonds


def _trace_initial(observation: PolicyValue) -> PolicyValue:
    traced = _trace_state(observation)
    if type(observation) is not dict or observation["initial"] is None:
        raise ValueError("Molecules initial trace is invalid")
    assert isinstance(traced, dict)
    traced["velocities"] = copy_policy_value(observation["velocities"])
    traced["initial"] = copy_policy_value(observation["initial"])
    return traced


def _trace_state(observation: PolicyValue) -> PolicyValue:
    if type(observation) is not dict or set(observation) != {
        "turn",
        "turns_remaining",
        "positions",
        "velocities",
        "components",
        "component_count",
        "total_cost",
        "initial",
    }:
        raise ValueError("Molecules trace observation is invalid")
    return {
        "turn": copy_policy_value(observation["turn"]),
        "positions": copy_policy_value(observation["positions"]),
        "components": copy_policy_value(observation["components"]),
        "component_count": copy_policy_value(observation["component_count"]),
        "total_cost": copy_policy_value(observation["total_cost"]),
        "initial": None,
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


__all__ = ["MoleculesBenchmark"]
