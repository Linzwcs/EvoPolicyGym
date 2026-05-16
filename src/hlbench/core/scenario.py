"""Scenario metadata contract."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ScenarioSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELDOUT = "heldout"


class ObservationType(StrEnum):
    STATE = "state"
    SYMBOLIC = "symbolic"
    IMAGE = "image"


@dataclass(frozen=True)
class TaskSpec:
    goal: str
    success_condition: str
    reward_description: str
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            goal=str(data["goal"]),
            success_condition=str(data["success_condition"]),
            reward_description=str(data["reward_description"]),
            notes=tuple(str(item) for item in data.get("notes", ())),
        )


@dataclass(frozen=True)
class MinimumScore:
    value: float
    reason: str = "Policy errors, invalid actions, contract violations, or evaluator failures receive the minimum score."

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | float | int) -> "MinimumScore":
        if isinstance(data, dict):
            return cls(value=float(data["value"]), reason=str(data.get("reason", cls.reason)))
        return cls(value=float(data))


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seed_pool: str
    seeds: tuple[int, ...]
    public_feedback: bool
    env_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any], scenario_dir: Path) -> "SplitSpec":
        del scenario_dir
        seed_pool = str(data["seed_pool"])
        seeds = _load_seed_pool(seed_pool, expected_split=name)
        return cls(
            name=name,
            seed_pool=seed_pool,
            seeds=seeds,
            public_feedback=bool(data.get("public_feedback", name == ScenarioSplit.TRAIN.value)),
            env_overrides=dict(data.get("env_overrides", {})),
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "episodes": len(self.seeds),
            "public_feedback": self.public_feedback,
            "has_env_overrides": bool(self.env_overrides),
        }


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    scenario_id: str
    scenario_level: str
    env_backend: str
    env_id: str
    env_kwargs: dict[str, Any]
    observation_mode: str
    observation_type: str
    task: TaskSpec
    observation_meanings: tuple[dict[str, Any], ...]
    action_meanings: tuple[dict[str, Any], ...]
    max_steps: int
    splits: dict[str, SplitSpec]
    minimum_score: MinimumScore
    metadata: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    @classmethod
    def from_json(cls, name: str, path: Path) -> "ScenarioSpec":
        data = json.loads(path.read_text())
        known = {
            "scenario_id",
            "scenario_level",
            "env_backend",
            "env_id",
            "env_kwargs",
            "observation_mode",
            "observation_type",
            "task",
            "observation_meanings",
            "action_meanings",
            "max_steps",
            "seed_generation",
            "splits",
            "minimum_score",
        }
        splits = data.get("splits")
        if not isinstance(splits, dict):
            raise ValueError(f"{path} must define splits.train, splits.validation, and splits.heldout")
        normalized_splits = {}
        for split in ScenarioSplit:
            raw_split = splits.get(split.value)
            if not isinstance(raw_split, dict):
                raise ValueError(f"{path} must define splits.{split.value}.seed_pool")
            normalized_splits[split.value] = SplitSpec.from_mapping(
                split.value,
                raw_split,
                path.parent,
            )
        missing = [name for name, spec in normalized_splits.items() if not spec.seeds]
        if missing:
            raise ValueError(f"{path} has empty required split(s): {', '.join(missing)}")
        return cls(
            name=name,
            scenario_id=str(data["scenario_id"]),
            scenario_level=str(data.get("scenario_level", "smoke")),
            env_backend=str(data.get("env_backend", "gymnasium")),
            env_id=str(data["env_id"]),
            env_kwargs=dict(data.get("env_kwargs", {})),
            observation_mode=str(data.get("observation_mode", "jsonable")),
            observation_type=ObservationType(data.get("observation_type", ObservationType.STATE.value)).value,
            task=TaskSpec.from_mapping(dict(data["task"])),
            observation_meanings=tuple(dict(item) for item in data.get("observation_meanings", ())),
            action_meanings=tuple(dict(item) for item in data.get("action_meanings", ())),
            max_steps=int(data["max_steps"]),
            splits=normalized_splits,
            minimum_score=MinimumScore.from_mapping(data.get("minimum_score", 0.0)),
            metadata={key: value for key, value in data.items() if key not in known},
            root=path.parent,
        )

    def seeds_for_split(
        self,
        split: str | ScenarioSplit,
        limit: int | None = None,
        *,
        sampler_seed: int | None = None,
    ) -> list[int]:
        split_value = ScenarioSplit(split)
        selected = list(self.splits[split_value.value].seeds)
        if limit is None or limit >= len(selected):
            return selected
        rng = random.Random(sampler_seed) if sampler_seed is not None else random.SystemRandom()
        return rng.sample(selected, limit)

    def to_record(self, *, include_private_splits: bool = False) -> dict[str, Any]:
        record = {
            "name": self.name,
            "scenario_id": self.scenario_id,
            "scenario_level": self.scenario_level,
            "env_backend": self.env_backend,
            "env_id": self.env_id,
            "env_kwargs": self.env_kwargs,
            "observation_mode": self.observation_mode,
            "observation_type": self.observation_type,
            "task": asdict(self.task),
            "observation_meanings": [dict(item) for item in self.observation_meanings],
            "action_meanings": [dict(item) for item in self.action_meanings],
            "max_steps": self.max_steps,
            "minimum_score": asdict(self.minimum_score),
            "metadata": self.metadata,
        }
        if not include_private_splits:
            record["splits"] = {
                name: split.public_record()
                for name, split in self.splits.items()
                if name != ScenarioSplit.HELDOUT.value
            }
        else:
            record["splits"] = {
                name: {
                    "seed_pool": split.seed_pool,
                    "seed_count": len(split.seeds),
                    "public_feedback": split.public_feedback,
                    "env_overrides": split.env_overrides,
                }
                for name, split in self.splits.items()
            }
        return record


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def scenario_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("HLBENCH_SCENARIO_ROOT")
    if env_root:
        roots.extend(Path(item) for item in env_root.split(os.pathsep) if item)
    root = repo_root()
    roots.append(root / "src" / "hlbench" / "scenarios")
    return roots


def seed_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("HLBENCH_SEED_ROOT")
    if env_root:
        roots.extend(Path(item) for item in env_root.split(os.pathsep) if item)
    roots.append(repo_root() / "src" / "hlbench" / "seeds")
    return roots


def find_scenario_dir(name: str) -> Path:
    for root in scenario_roots():
        candidate = root / name
        if (candidate / "scenario.json").exists():
            return candidate
    searched = ", ".join(str(path) for path in scenario_roots())
    raise FileNotFoundError(f"scenario {name!r} not found under: {searched}")


def load_scenario(name: str) -> ScenarioSpec:
    scenario_dir = find_scenario_dir(name)
    return ScenarioSpec.from_json(name=name, path=scenario_dir / "scenario.json")


def list_scenarios() -> list[str]:
    names: set[str] = set()
    for root in scenario_roots():
        if not root.exists():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and (candidate / "scenario.json").exists():
                names.add(candidate.name)
    return sorted(names)


Scenario = ScenarioSpec


def _load_seed_pool(seed_pool: str, *, expected_split: str) -> tuple[int, ...]:
    candidate = Path(seed_pool)
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    search_paths: list[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(root / candidate for root in seed_roots())
    for path in search_paths:
        if path.exists():
            return _load_seed_file(path, expected_split=expected_split)
    searched = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"seed pool {seed_pool!r} not found under: {searched}")


def _load_seed_file(path: Path, *, expected_split: str) -> tuple[int, ...]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        seeds = data
    elif isinstance(data, dict):
        split = data.get("split")
        if split is not None and str(split) != expected_split:
            raise ValueError(f"{path} declares split {split!r}, expected {expected_split!r}")
        seeds = data.get("seeds")
    else:
        raise ValueError(f"{path} must contain a seed list or an object with seeds")
    if not isinstance(seeds, list):
        raise ValueError(f"{path} must define a seeds list")
    return tuple(int(seed) for seed in seeds)
