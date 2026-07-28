"""The independent CodeChef WAREHOUS Warehouseman Benchmark."""

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

from .environment import WarehousemanEnvironment
from .simulation import (
    GENERATOR_ID,
    MAX_COLUMNS,
    MAX_INSTRUCTION_CHARACTERS,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
)

FAILURE_COST = 1_000_000.0

_EPISODE_SEED_DOMAIN = b"evopolicygym-warehouseman/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_DIAGNOSTIC_EPISODES = 64

_SPEC = BenchmarkSpec(
    id="codechef/WAREHOUS/Warehouseman/mean-normalized-cost-v1",
    description=(
        "Store every arriving shipment, retrieve shipments in numeric order, "
        "and return the forklift to the entrance. Minimize mean normalized "
        "instruction cost."
    ),
    observation_space={
        "type": "object",
        "fields": {
            "rows": {"type": "integer", "minimum": MIN_ROWS, "maximum": MAX_ROWS},
            "columns": {
                "type": "integer",
                "minimum": MIN_COLUMNS,
                "maximum": MAX_COLUMNS,
            },
            "arrivals": {
                "type": "array",
                "items": {"type": "integer"},
                "notes": "A permutation of shipment numbers; shipment 1 is not last.",
            },
            "instruction_limit": {
                "type": "integer",
                "constant": MAX_INSTRUCTION_CHARACTERS,
            },
        },
        "notes": "The Environment seed and split identity are never exposed.",
    },
    action_space={
        "type": "string",
        "alphabet": ["N", "W", "S", "E", "P", "D", "L", "U"],
        "instruction_vocabulary": [
            "N",
            "W",
            "S",
            "E",
            "P",
            "D",
            "LN",
            "LW",
            "LS",
            "LE",
            "UN",
            "UW",
            "US",
            "UE",
        ],
        "maximum_characters": MAX_INSTRUCTION_CHARACTERS,
        "notes": "The complete instruction string is validated and executed atomically.",
    },
    metadata={
        "environment": "Warehouseman (Challenge)",
        "provider": "CodeChef",
        "problem": "WAREHOUS",
        "contest": "June Challenge 2018",
        "upstream_specification": "https://www.codechef.com/problems/WAREHOUS",
        "upstream_editorial": "https://discuss.codechef.com/t/18954",
        "upstream_specification_revision": "2018-05-08",
        "upstream_material_license": "CodeChef terms; not redistributed",
        "implementation": "independent",
        "upstream_code_included": False,
        "upstream_inputs_included": False,
        "upstream_assets_included": False,
        "instruction_limit": MAX_INSTRUCTION_CHARACTERS,
        "failure_cost": FAILURE_COST,
    },
    environment_parameters={
        "generator": GENERATOR_ID,
        "minimum_rows": MIN_ROWS,
        "maximum_rows": MAX_ROWS,
        "minimum_columns": MIN_COLUMNS,
        "maximum_columns": MAX_COLUMNS,
        "instruction_limit": MAX_INSTRUCTION_CHARACTERS,
        "failure_cost": FAILURE_COST,
    },
    max_episode_steps=1,
    primary_metric="mean_normalized_cost",
    score_direction="minimize",
)


class WarehousemanBenchmark:
    """Mean official normalized cost over independent generated cases."""

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
        return WarehousemanEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        costs = tuple(_episode_cost(record) for record in records)
        score = statistics.fmean(costs)
        completed = sum(record.policy_failure is None for record in records)
        instruction_counts = tuple(
            _metric_integer(record, "instruction_characters")
            for record in records
            if record.policy_failure is None
        )
        diagnostics = _diagnostic_artifact(records[:_MAX_DIAGNOSTIC_EPISODES])
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean normalized cost {score:.3f}; "
                    f"{completed}/{len(records)} solutions completed."
                ),
                "mean_normalized_cost": score,
                "mean_instruction_characters": (
                    statistics.fmean(instruction_counts)
                    if instruction_counts
                    else None
                ),
                "episodes": len(records),
                "completed": completed,
                "policy_failures": len(records) - completed,
                "failure_cost": FAILURE_COST,
                "diagnostic_episodes": min(
                    len(records),
                    _MAX_DIAGNOSTIC_EPISODES,
                ),
                "diagnostic_episodes_omitted": max(
                    0,
                    len(records) - _MAX_DIAGNOSTIC_EPISODES,
                ),
            },
            artifacts=(diagnostics,),
        )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _episode_cost(record: EpisodeRecord) -> float:
    if record.policy_failure is not None:
        return FAILURE_COST
    if len(record.transitions) != 1 or not record.transitions[0].step.terminated:
        raise ValueError("completed Warehouseman Episode evidence is invalid")
    return _metric_number(record, "normalized_cost")


def _metric_number(record: EpisodeRecord, name: str) -> float:
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Warehouseman terminal metrics are invalid")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Warehouseman terminal metrics are invalid")
    return float(value)


def _metric_integer(record: EpisodeRecord, name: str) -> int:
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Warehouseman terminal metrics are invalid")
    value = metrics.get(name)
    if type(value) is not int:
        raise ValueError("Warehouseman terminal metrics are invalid")
    return value


def _diagnostic_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    metric_names = (
        "instruction_characters",
        "moves",
        "picks",
        "drops",
        "loads",
        "unloads",
    )
    for episode_index, record in enumerate(records):
        document: dict[str, object] = {
            "type": "episode",
            "episode_index": episode_index,
            "status": (
                "completed"
                if record.policy_failure is None
                else "policy_failed"
            ),
            "score": (
                _episode_cost(record)
                if record.policy_failure is None
                else FAILURE_COST
            ),
            "failure": record.policy_failure,
        }
        if record.policy_failure is None:
            document["operations"] = {
                name: _metric_integer(record, name)
                for name in metric_names
            }
        lines.append(_json_line(document))
    return Artifact(
        name="diagnostics.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


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


__all__ = ["FAILURE_COST", "WarehousemanBenchmark"]
