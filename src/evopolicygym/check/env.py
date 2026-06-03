"""Environment registration invariant checker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import Case, Env, Pool, PoolKind, Turn
from ..data import load as load_data
from .run import Issue, Report
from .task import check_task_text


@dataclass(slots=True)
class _EnvCheck:
    env: Env
    root: Path | None = None
    issues: list[Issue] = field(default_factory=list)

    def run(self) -> Report:
        self._task()
        self._secret()
        self._pools()
        self._sources()
        self._leaks()
        self._world()
        self._value()
        return Report(root=self.root or Path(self.env.task.name or "env"), issues=tuple(self.issues))

    def _task(self) -> None:
        task = self.env.task
        if not task.name:
            self._issue("task_name", "task.name must be non-empty")
        if not task.version:
            self._issue("task_version", "task.version must be non-empty")
        if task.cases <= 0:
            self._issue("task_cases", "task.cases must be positive")
        if task.steps <= 0:
            self._issue("task_steps", "task.steps must be positive")
        if not isinstance(task.obs, dict):
            self._issue("task_obs", "task.obs must be a dict")
        if not isinstance(task.act, dict):
            self._issue("task_act", "task.act must be a dict")
        if task.storage not in {"inline", "external"}:
            self._issue("task_storage", "task.storage must be inline or external")
        if task.storage == "external" and not self.env.caps.observations:
            self._issue(
                "caps_observations",
                "external observation storage requires Caps(observations=True)",
            )
        self.issues.extend(check_task_text(self.env.text, path=task.name or "env"))

    def _secret(self) -> None:
        secret = self.env.secret
        refs = {secret.train, secret.valid, secret.final}
        if any(not ref for ref in refs):
            self._issue("secret_ref", "secret pool refs must be non-empty")
        if len(refs) != 3:
            self._issue("secret_ref_unique", "train/valid/final refs must be distinct")
        if secret.valid_size <= 0:
            self._issue("valid_size", "valid_size must be positive")
        if secret.final_size <= 0:
            self._issue("final_size", "final_size must be positive")

    def _pools(self) -> None:
        expected = {
            PoolKind.train: (self.env.task.cases, self.env.secret.train),
            PoolKind.valid: (self.env.secret.valid_size, self.env.secret.valid),
            PoolKind.final: (self.env.secret.final_size, self.env.secret.final),
        }
        for kind, (size, ref) in expected.items():
            try:
                pool = self.env.pool(kind)
            except Exception as exc:
                self._issue("pool", f"Env.pool({kind.value}) raised {type(exc).__name__}: {exc}")
                continue
            if not isinstance(pool, Pool):
                self._issue("pool_type", f"Env.pool({kind.value}) must return Pool")
                continue
            if pool.kind != kind or pool.size != size or pool.ref != ref:
                self._issue("pool_shape", f"Env.pool({kind.value}) does not match task/secret")
            if pool.size > 0:
                case = pool.case(0)
                if not isinstance(case, Case):
                    self._issue("case_type", "Pool.case(0) must return Case")
                if not case.ref.startswith(f"{pool.ref}/"):
                    self._issue("case_ref", "Case.ref must be scoped under Pool.ref")

    def _sources(self) -> None:
        if self.root is None:
            return

        try:
            corpus = load_data(self.root, env=self.env.task.name)
        except FileNotFoundError as exc:
            self._issue("source_missing", f"case source is missing: {exc}", path=exc.filename)
            return
        except (NotADirectoryError, ValueError) as exc:
            code = "source_overlap" if "overlaps" in str(exc) else "source_shape"
            if "duplicate" in str(exc):
                code = "source_duplicate"
            if "JSON" in str(exc):
                code = "source_json"
            self._issue(code, str(exc), path=self.root)
            return

        expected = {
            PoolKind.train: self.env.pool(PoolKind.train).size,
            PoolKind.valid: self.env.pool(PoolKind.valid).size,
            PoolKind.final: self.env.pool(PoolKind.final).size,
        }
        for kind, size in expected.items():
            pool = corpus.pool(kind)
            if pool.size != size:
                self._issue(
                    "source_size",
                    f"{kind.value} source size {pool.size} != pool size {size}",
                    path=pool.ref,
                )
            if pool.size > 0:
                case = pool.case(0)
                if not isinstance(case.data, dict):
                    self._issue("case_source_data", "Case.data must be a dict", path=pool.ref)

    def _leaks(self) -> None:
        hidden = (self.env.secret.valid, self.env.secret.final)
        visible = _dump(
            {
                "task": self.env.task,
                "text": self.env.text,
                "caps": self.env.caps.body(),
            }
        )
        for ref in hidden:
            if ref and ref in visible:
                self._issue("hidden_leak", f"hidden pool ref {ref!r} appears in agent-visible metadata")

    def _world(self) -> None:
        if self.env.task.cases <= 0:
            return
        try:
            world = self.env.make()
        except Exception as exc:
            self._issue("world_make", f"make() raised {type(exc).__name__}: {exc}")
            return
        try:
            obs = world.reset(self.env.pool(PoolKind.train).case(0))
        except Exception as exc:
            self._issue("world_reset", f"World.reset(Case) raised {type(exc).__name__}: {exc}")
            return
        try:
            action = world.sample()
        except Exception as exc:
            self._issue("world_sample", f"World.sample() raised {type(exc).__name__}: {exc}")
            return
        try:
            turn = world.step(action)
        except Exception as exc:
            self._issue("world_step", f"World.step(sample()) raised {type(exc).__name__}: {exc}")
            return
        if not isinstance(turn, Turn):
            self._issue("world_turn", "World.step() must return Turn")
        elif not isinstance(turn.reward, int | float):
            self._issue("world_reward", "Turn.reward must be numeric")
        _ = obs

    def _value(self) -> None:
        if self.env.value is None:
            return
        returns = (0.0, 1.0)
        final: float | None = None
        final_checked = False
        for kind in (PoolKind.valid, PoolKind.final):
            try:
                value = self.env.value(self.env.pool(kind), returns)
            except Exception as exc:
                self._issue("value", f"Env.value({kind.value}) raised {type(exc).__name__}: {exc}")
                continue
            if value is not None and not isinstance(value, int | float):
                self._issue("value_type", f"Env.value({kind.value}) must return float or None")
            if kind == PoolKind.final:
                final = value
                final_checked = True
        if final_checked and final is None:
            self._issue("final_value", "Env.value must return a scalar for final pool")

    def _issue(self, code: str, message: str, *, path: str | Path | None = None) -> None:
        self.issues.append(Issue(code=code, path=self._path(path), message=message))

    def _path(self, path: str | Path | None) -> str:
        if path is None:
            return self.env.task.name or "env"
        if isinstance(path, str):
            return path
        if self.root is not None:
            try:
                return str(path.relative_to(self.root))
            except ValueError:
                pass
        return str(path)


def check_env(env: Env, root: str | Path | None = None) -> Report:
    """Check one environment registration, and optional file-backed case sources."""

    return _EnvCheck(env, root=Path(root) if root is not None else None).run()


def _dump(value: Any) -> str:
    return json.dumps(value, default=repr, sort_keys=True)


def _source_path(root: Path, ref: str) -> Path | None:
    base = root / ref
    candidates = (base.with_suffix(".json"), base / "cases.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolves(ref: str, pool: Pool, source: list[Any]) -> bool:
    prefix = f"{pool.ref}/"
    if not ref.startswith(prefix):
        return False
    value = ref[len(prefix) :]
    if not value.isdigit():
        return False
    index = int(value)
    return 0 <= index < len(source)


def _key(value: Any) -> str:
    return json.dumps(value, default=repr, sort_keys=True, separators=(",", ":"))
