"""The independent interactive Treant's Forest Benchmark."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, copy_policy_value

from .environment import MAX_EPISODE_STEPS, TreantsForestEnvironment
from .simulation import GENERATOR_ID, MAX_SIZE, MIN_SIZE

_EPISODE_SEED_DOMAIN = b"evopolicygym-treants-forest/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_TRANSITIONS = 4_096

_SPEC = BenchmarkSpec(
    id="atcoder/AHC054/TreantsForest/capped-mean-turns-v1",
    description=(
        "Place permanent Treants on unseen empty cells while preserving a "
        "route to the flower. Delay the deterministic adventurer for as many "
        f"as {MAX_EPISODE_STEPS} turns."
    ),
    observation_space={
        "type": "object",
        "fields": {
            "turn": {"type": "integer", "minimum": 0},
            "adventurer": {"type": "coordinate"},
            "newly_revealed": {
                "type": "array",
                "items": {"type": "coordinate"},
            },
            "revealed_cells": {"type": "integer", "minimum": 1},
            "placed_treants": {"type": "integer", "minimum": 0},
            "initial": {
                "one_of": ["initial_forest", "null"],
                "notes": "Present only in the first observation.",
            },
        },
        "initial_forest": {
            "fields": ["size", "entrance", "flower", "trees"],
        },
        "notes": (
            "The adventurer's private target order and the Episode seed are "
            "never exposed. The Policy may retain state within the Episode."
        ),
    },
    action_space={
        "type": "object",
        "fields": {
            "placements": {
                "type": "array",
                "items": {"type": "coordinate"},
                "max_items": MAX_SIZE * MAX_SIZE,
            },
        },
        "required": ["placements"],
        "additional_fields": False,
        "notes": (
            "Every coordinate must be distinct, in bounds, unseen, empty, and "
            "different from the flower. Placements are atomic and must retain "
            "a path from both the entrance and adventurer to the flower."
        ),
    },
    metadata={
        "environment": "Treant's Forest",
        "provider": "AtCoder",
        "contest": "AtCoder Heuristic Contest 054",
        "task": "AHC054 A",
        "upstream_specification": (
            "https://atcoder.jp/contests/ahc054/tasks/ahc054_a"
        ),
        "upstream_specification_revision": "2025-09-21",
        "upstream_tool_archive_revision": "YDAxDRZr_v2",
        "upstream_tool_license": "not declared; not redistributed",
        "implementation": "independent",
        "upstream_code_included": False,
        "upstream_inputs_included": False,
        "upstream_assets_included": False,
        "reward_per_turn": 1.0,
        "failure_score": 0.0,
        "turn_cap": MAX_EPISODE_STEPS,
    },
    environment_parameters={
        "generator": GENERATOR_ID,
        "minimum_size": MIN_SIZE,
        "maximum_size": MAX_SIZE,
        "turn_cap": MAX_EPISODE_STEPS,
    },
    max_episode_steps=MAX_EPISODE_STEPS,
    primary_metric="capped_mean_turns",
    score_direction="maximize",
)


class TreantsForestBenchmark:
    """Capped mean delay over deterministic independent forest cases."""

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
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return TreantsForestEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        turns = tuple(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        score = statistics.fmean(turns)
        failures = sum(record.policy_failure is not None for record in records)
        reached = sum(_terminal_flag(record, "flower_reached") for record in records)
        capped = sum(_terminal_flag(record, "turn_cap_reached") for record in records)
        placements = tuple(_final_integer(record, "placed_treants") for record in records)
        trace, traced_episodes, traced_transitions, omitted_transitions = (
            _trace_artifact(records)
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Capped mean delay {score:.3f} turns; "
                    f"{reached}/{len(records)} adventurers reached the flower "
                    f"and {capped} reached the turn cap."
                ),
                "capped_mean_turns": score,
                "mean_placed_treants": statistics.fmean(placements),
                "flower_reached": reached,
                "turn_cap_reached": capped,
                "episodes": len(records),
                "policy_failures": failures,
                "failure_score": 0.0,
                "turn_cap": MAX_EPISODE_STEPS,
                "traced_episodes": traced_episodes,
                "trace_episodes_omitted": len(records) - traced_episodes,
                "traced_transitions": traced_transitions,
                "trace_transitions_omitted": omitted_transitions,
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


def _terminal_flag(record: EpisodeRecord, name: str) -> bool:
    if record.policy_failure is not None or not record.transitions:
        return False
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Treant's Forest terminal metrics are invalid")
    value = metrics.get(name)
    if type(value) is not bool:
        raise ValueError("Treant's Forest terminal metrics are invalid")
    return value


def _final_integer(record: EpisodeRecord, name: str) -> int:
    if not record.transitions:
        return 0
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Treant's Forest terminal metrics are invalid")
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError("Treant's Forest terminal metrics are invalid")
    return value


def _trace_artifact(
    records: Sequence[EpisodeRecord],
) -> tuple[Artifact, int, int, int]:
    lines: list[bytes] = []
    traced_episodes = 0
    traced_transitions = 0
    total_transitions = sum(record.steps for record in records)
    for episode_index, record in enumerate(records[:_MAX_TRACED_EPISODES]):
        traced_episodes += 1
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": _status(record),
                    "steps": record.steps,
                    "score": (
                        record.total_reward
                        if record.policy_failure is None
                        else 0.0
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if traced_transitions >= _MAX_TRACED_TRANSITIONS:
                break
            action = _trace_action(transition.action)
            next_observation = _trace_observation(
                transition.step.observation
            )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "reward": transition.step.reward,
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
            traced_transitions += 1
            observation = next_observation
    return (
        Artifact(
            name="trace.jsonl",
            media_type="application/x-ndjson",
            content=b"".join(lines),
        ),
        traced_episodes,
        traced_transitions,
        total_transitions - traced_transitions,
    )


def _status(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failed"
    if _terminal_flag(record, "turn_cap_reached"):
        return "turn_cap"
    return "flower_reached"


def _trace_action(action: PolicyValue) -> PolicyValue:
    if type(action) is not dict or set(action) != {"placements"}:
        raise ValueError("Treant's Forest trace Action is invalid")
    placements = action["placements"]
    if type(placements) is not list:
        raise ValueError("Treant's Forest trace Action is invalid")
    _validate_coordinates(placements)
    return copy_policy_value(action)


def _trace_observation(observation: PolicyValue) -> PolicyValue:
    if type(observation) is not dict or set(observation) != {
        "turn",
        "adventurer",
        "newly_revealed",
        "revealed_cells",
        "placed_treants",
        "initial",
    }:
        raise ValueError("Treant's Forest trace observation is invalid")
    for name in ("turn", "revealed_cells", "placed_treants"):
        if type(observation[name]) is not int:
            raise ValueError("Treant's Forest trace observation is invalid")
    adventurer = observation["adventurer"]
    newly_revealed = observation["newly_revealed"]
    if type(adventurer) is not list or type(newly_revealed) is not list:
        raise ValueError("Treant's Forest trace observation is invalid")
    _validate_coordinates([adventurer])
    _validate_coordinates(newly_revealed)
    initial = observation["initial"]
    if initial is not None:
        if type(initial) is not dict or set(initial) != {
            "size",
            "entrance",
            "flower",
            "trees",
        }:
            raise ValueError("Treant's Forest trace observation is invalid")
        if type(initial["size"]) is not int:
            raise ValueError("Treant's Forest trace observation is invalid")
        entrance = initial["entrance"]
        flower = initial["flower"]
        trees = initial["trees"]
        if (
            type(entrance) is not list
            or type(flower) is not list
            or type(trees) is not list
        ):
            raise ValueError("Treant's Forest trace observation is invalid")
        _validate_coordinates([entrance, flower, *trees])
    return copy_policy_value(observation)


def _validate_coordinates(values: list[PolicyValue]) -> None:
    for value in values:
        if (
            type(value) is not list
            or len(value) != 2
            or type(value[0]) is not int
            or type(value[1]) is not int
        ):
            raise ValueError("Treant's Forest coordinate is invalid")


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


__all__ = ["TreantsForestBenchmark"]
