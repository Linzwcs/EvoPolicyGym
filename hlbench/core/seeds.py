"""Seed pool generation and persistence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from hlbench.core.scenario import ScenarioSplit, seed_roots


DEFAULT_MAX_SEED = 2_147_483_647


@dataclass(frozen=True)
class SeedGenerationConfig:
    generator_seed: int
    train_count: int
    validation_count: int
    heldout_count: int
    min_seed: int = 0
    max_seed: int = DEFAULT_MAX_SEED

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            ScenarioSplit.TRAIN.value: self.train_count,
            ScenarioSplit.VALIDATION.value: self.validation_count,
            ScenarioSplit.HELDOUT.value: self.heldout_count,
        }


def random_seed_partition(config: SeedGenerationConfig) -> dict[str, list[int]]:
    total = sum(config.split_counts.values())
    population_size = config.max_seed - config.min_seed + 1
    if total > population_size:
        raise ValueError(f"requested {total} seeds from population of {population_size}")
    rng = random.Random(config.generator_seed)
    seeds = rng.sample(range(config.min_seed, config.max_seed + 1), total)
    partitions: dict[str, list[int]] = {}
    offset = 0
    for split, count in config.split_counts.items():
        partitions[split] = seeds[offset : offset + count]
        offset += count
    return partitions


def write_seed_files(*, pool_name: str, config: SeedGenerationConfig, root: Path | None = None) -> dict[str, Path]:
    seed_root = root or seed_roots()[-1]
    partitions = random_seed_partition(config)
    written: dict[str, Path] = {}
    for split, seeds in partitions.items():
        target = seed_root / pool_name / f"{split}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pool": pool_name,
            "split": split,
            "generator": {
                "method": "random_partition",
                "generator_seed": config.generator_seed,
                "min_seed": config.min_seed,
                "max_seed": config.max_seed,
                "split_counts": config.split_counts,
            },
            "seeds": seeds,
        }
        target.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        written[split] = target
    return written
