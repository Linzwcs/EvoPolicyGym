"""External case split loading for EvoPolicyGym runs."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Case, Env, Pool, PoolKind

Json = Any

_FILES = {
    PoolKind.train: "train.json",
    PoolKind.valid: "valid.json",
    PoolKind.final: "heldout.json",
}

_LABELS = {
    PoolKind.train: "train",
    PoolKind.valid: "valid",
    PoolKind.final: "heldout",
}


@dataclass(frozen=True, slots=True)
class Split:
    """One loaded case split and its reproducibility metadata."""

    kind: PoolKind
    path: Path
    pool: Pool
    hash: str
    rows: tuple[Json, ...]


@dataclass(frozen=True, slots=True)
class Corpus:
    """Three external case splits for one benchmark run."""

    root: Path
    train: Split
    valid: Split
    final: Split

    def pool(self, kind: PoolKind) -> Pool:
        if kind == PoolKind.train:
            return self.train.pool
        if kind == PoolKind.valid:
            return self.valid.pool
        if kind == PoolKind.final:
            return self.final.pool
        raise ValueError(f"unknown pool kind: {kind}")

    def versions(self) -> dict[str, str]:
        return {
            "data_train_hash": self.train.hash,
            "data_valid_hash": self.valid.hash,
            "data_heldout_hash": self.final.hash,
        }


def make(
    root: str | Path,
    env: Env,
    *,
    seed: int = 0,
    train_size: int | None = None,
    valid_size: int | None = None,
    heldout_size: int | None = None,
    overwrite: bool = False,
) -> Corpus:
    """Write deterministic seed-backed split files for `env` and reload them."""

    sizes = {
        PoolKind.train: train_size if train_size is not None else env.task.cases,
        PoolKind.valid: valid_size if valid_size is not None else env.secret.valid_size,
        PoolKind.final: heldout_size if heldout_size is not None else env.secret.final_size,
    }
    for kind, size in sizes.items():
        if size <= 0:
            raise ValueError(f"{_LABELS[kind]} size must be positive")

    base = Path(root)
    paths = {kind: base / filename for kind, filename in _FILES.items()}
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(existing[0])

    seeds = _seeds(seed, sum(sizes.values()))
    offset = 0
    base.mkdir(parents=True, exist_ok=True)
    for kind in (PoolKind.train, PoolKind.valid, PoolKind.final):
        size = sizes[kind]
        rows = [{"seed": item} for item in seeds[offset : offset + size]]
        offset += size
        _write_split(
            paths[kind],
            env=env,
            kind=kind,
            seed=seed,
            cases=rows,
        )
    return load(base, env=env.task.name)


def load(root: str | Path, *, env: str | None = None) -> Corpus:
    """Load `train.json`, `valid.json`, and `heldout.json` from `root`."""

    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    if not base.is_dir():
        raise NotADirectoryError(base)

    splits = {kind: _split(base, kind, env=env) for kind in _FILES}
    _disjoint(splits)
    return Corpus(
        root=base,
        train=splits[PoolKind.train],
        valid=splits[PoolKind.valid],
        final=splits[PoolKind.final],
    )


def _split(root: Path, kind: PoolKind, *, env: str | None) -> Split:
    path = root / _FILES[kind]
    data = _json(path)
    meta, rows = _shape(data, path)
    _match(meta, "env", env, path)
    _match(meta, "split", _LABELS[kind], path)
    if not rows:
        raise ValueError(f"{path}: split must contain at least one case")

    ref = _ref(meta, kind)
    cases = tuple(_case(index, ref, row) for index, row in enumerate(rows))
    if len({_key(row) for row in rows}) != len(rows):
        raise ValueError(f"{path}: split contains duplicate cases")
    return Split(
        kind=kind,
        path=path,
        pool=Pool(kind=kind, size=len(cases), ref=ref, cases=cases),
        hash=_hash(path),
        rows=tuple(rows),
    )


def _json(path: Path) -> Json:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _shape(data: Json, path: Path) -> tuple[Mapping[str, Json], list[Json]]:
    if isinstance(data, Mapping):
        cases = data.get("cases")
        if not isinstance(cases, Sequence) or isinstance(cases, str | bytes | bytearray):
            raise ValueError(f"{path}: object split must contain a cases array")
        return data, list(cases)
    if isinstance(data, Sequence) and not isinstance(data, str | bytes | bytearray):
        return {}, list(data)
    raise ValueError(f"{path}: split must be a JSON array or object with cases")


def _match(meta: Mapping[str, Json], name: str, expected: str | None, path: Path) -> None:
    if expected is None or name not in meta:
        return
    value = meta[name]
    if value != expected:
        raise ValueError(f"{path}: {name} must be {expected!r}")


def _ref(meta: Mapping[str, Json], kind: PoolKind) -> str:
    value = meta.get("ref")
    if isinstance(value, str) and value:
        return value
    return _LABELS[kind]


def _case(index: int, pool_ref: str, row: Json) -> Case:
    if isinstance(row, Mapping):
        data = dict(row)
        ref = row.get("ref")
        case_ref = ref if isinstance(ref, str) and ref else f"{pool_ref}/{index:06d}"
        return Case(id=index, ref=case_ref, data=data)
    return Case(id=index, ref=f"{pool_ref}/{index:06d}", data={"value": row})


def _disjoint(splits: Mapping[PoolKind, Split]) -> None:
    seen: dict[str, PoolKind] = {}
    for kind, split in splits.items():
        for row in split.rows:
            key = _key(row)
            if key in seen:
                raise ValueError(
                    f"{split.path}: {kind.value} overlaps with {seen[key].value}"
                )
            seen[key] = kind


def _key(value: Json) -> str:
    return json.dumps(value, default=repr, sort_keys=True, separators=(",", ":"))


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _seeds(master: int, count: int) -> list[int]:
    rng = random.Random(master)
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        value = rng.randrange(0, 2**31)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _write_split(
    path: Path,
    *,
    env: Env,
    kind: PoolKind,
    seed: int,
    cases: list[dict[str, int]],
) -> None:
    payload = {
        "env": env.task.name,
        "split": _LABELS[kind],
        "ref": env.pool(kind).ref,
        "generator": {
            "kind": "seed",
            "seed": seed,
            "size": len(cases),
        },
        "cases": cases,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
