"""The independent interactive AHC058 Apple Incremental Game Benchmark."""

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

from .environment import AppleIncrementalGameEnvironment
from .simulation import (
    GENERATOR_ID,
    INITIAL_APPLES,
    LEVELS,
    MACHINE_IDS,
    TURNS,
)

_EPISODE_SEED_DOMAIN = b"evopolicygym-apple-incremental/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 4
_MAX_TRACED_TRANSITIONS = 2_000

_SPEC = BenchmarkSpec(
    id="atcoder/AHC058/AppleIncrementalGame/mean-log2-score-v1",
    description=(
        "Allocate apples across a four-level machine production hierarchy for "
        f"{TURNS} turns. Maximize mean final log2 apple score."
    ),
    observation_space={
        "type": "object",
        "fields": {
            "turn": {"type": "integer", "minimum": 0, "maximum": TURNS},
            "turns_remaining": {
                "type": "integer",
                "minimum": 0,
                "maximum": TURNS,
            },
            "apples": {"type": "integer", "minimum": 0},
            "machines": {"type": "integer_matrix", "shape": [LEVELS, MACHINE_IDS]},
            "powers": {"type": "integer_matrix", "shape": [LEVELS, MACHINE_IDS]},
            "initial": {
                "one_of": ["machine_configuration", "null"],
                "notes": "Present only in the first observation.",
            },
        },
        "machine_configuration": {
            "fields": ["capacities", "costs"],
        },
        "notes": "The Episode seed and split identity are never exposed.",
    },
    action_space={
        "one_of": [
            {"type": "null", "meaning": "wait"},
            {
                "type": "object",
                "fields": {
                    "upgrade": {
                        "type": "array",
                        "items": ["level", "machine_id"],
                        "length": 2,
                    },
                },
                "required": ["upgrade"],
                "additional_fields": False,
            },
        ],
        "notes": "An unaffordable or out-of-range upgrade is invalid and is never repaired.",
    },
    metadata={
        "environment": "Apple Incremental Game",
        "provider": "AtCoder",
        "contest": "AtCoder Heuristic Contest 058",
        "task": "AHC058 A",
        "upstream_specification": (
            "https://atcoder.jp/contests/ahc058/tasks/ahc058_a"
        ),
        "upstream_specification_revision": "2025-12-14",
        "upstream_tool_archive_revision": "UpvAVdx6",
        "upstream_tool_license": "not declared; not redistributed",
        "implementation": "independent",
        "upstream_code_included": False,
        "upstream_inputs_included": False,
        "upstream_assets_included": False,
        "machine_ids": MACHINE_IDS,
        "levels": LEVELS,
        "turns": TURNS,
        "initial_apples": INITIAL_APPLES,
        "failure_score": 0.0,
    },
    environment_parameters={
        "generator": GENERATOR_ID,
        "machine_ids": MACHINE_IDS,
        "levels": LEVELS,
        "turns": TURNS,
        "initial_apples": INITIAL_APPLES,
    },
    max_episode_steps=TURNS,
    primary_metric="mean_log2_score",
    score_direction="maximize",
)


class AppleIncrementalGameBenchmark:
    """Mean final log2 apple score over independent generated cases."""

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
        return AppleIncrementalGameEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        scores = tuple(_episode_score(record) for record in records)
        score = statistics.fmean(scores)
        completed = sum(record.policy_failure is None for record in records)
        final_apples = tuple(
            _final_integer(record, "apples")
            for record in records
            if record.policy_failure is None
        )
        upgrades = tuple(
            _final_integer(record, "total_upgrades")
            for record in records
            if record.policy_failure is None
        )
        trace, traced_episodes, traced_transitions, omitted_transitions = (
            _trace_artifact(records)
        )
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean log2 score {score:.3f}; "
                    f"{completed}/{len(records)} Episodes completed."
                ),
                "mean_log2_score": score,
                "mean_final_apples": (
                    statistics.fmean(final_apples)
                    if final_apples
                    else None
                ),
                "mean_total_upgrades": (
                    statistics.fmean(upgrades) if upgrades else None
                ),
                "episodes": len(records),
                "completed": completed,
                "policy_failures": len(records) - completed,
                "failure_score": 0.0,
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


def _episode_score(record: EpisodeRecord) -> float:
    if record.policy_failure is not None:
        return 0.0
    if (
        len(record.transitions) != TURNS
        or not record.transitions[-1].step.terminated
    ):
        raise ValueError("completed Apple Incremental Episode evidence is invalid")
    return float(_final_integer(record, "final_score"))


def _final_integer(record: EpisodeRecord, name: str) -> int:
    if not record.transitions:
        raise ValueError("Apple Incremental terminal evidence is missing")
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Apple Incremental terminal metrics are invalid")
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError("Apple Incremental terminal metrics are invalid")
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
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "score": _episode_score(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            if traced_transitions >= _MAX_TRACED_TRANSITIONS:
                break
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
                        "action": _trace_action(transition.action),
                        "reward": transition.step.reward,
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
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


def _trace_action(action: PolicyValue) -> PolicyValue:
    if action is None:
        return None
    if type(action) is not dict or set(action) != {"upgrade"}:
        raise ValueError("Apple Incremental trace Action is invalid")
    upgrade = action["upgrade"]
    if (
        type(upgrade) is not list
        or len(upgrade) != 2
        or any(type(value) is not int for value in upgrade)
    ):
        raise ValueError("Apple Incremental trace Action is invalid")
    return copy_policy_value(action)


def _trace_observation(observation: PolicyValue) -> PolicyValue:
    if type(observation) is not dict or set(observation) != {
        "turn",
        "turns_remaining",
        "apples",
        "machines",
        "powers",
        "initial",
    }:
        raise ValueError("Apple Incremental trace observation is invalid")
    for name in ("turn", "turns_remaining", "apples"):
        if type(observation[name]) is not int:
            raise ValueError("Apple Incremental trace observation is invalid")
    for name in ("machines", "powers"):
        matrix = observation[name]
        if (
            type(matrix) is not list
            or len(matrix) != LEVELS
            or any(type(row) is not list or len(row) != MACHINE_IDS for row in matrix)
        ):
            raise ValueError("Apple Incremental trace observation is invalid")
        for row in matrix:
            if type(row) is not list or any(
                type(value) is not int for value in row
            ):
                raise ValueError(
                    "Apple Incremental trace observation is invalid"
                )
    initial = observation["initial"]
    if initial is not None and (
        type(initial) is not dict
        or set(initial) != {"capacities", "costs"}
    ):
        raise ValueError("Apple Incremental trace observation is invalid")
    return copy_policy_value(observation)


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


__all__ = ["AppleIncrementalGameBenchmark"]
