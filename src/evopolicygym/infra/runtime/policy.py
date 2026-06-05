"""Policy runtime skeleton.

This module owns Python policy loading and lifecycle orchestration. It
delegates environment-specific episode execution to a Roller adapter.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, replace
from math import sqrt
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from ...core import (
    Budget,
    Exec,
    Pool,
    PoolKind,
    Run,
    Score,
    Snap,
    Submit,
    Task,
    Trace,
    Verdict,
)


class Runner(Protocol):
    """Episode executor used by PolicyRuntime."""

    def run(
        self,
        policy: object,
        task: Task,
        pool: Pool,
        cases: tuple[int, ...],
    ) -> tuple[Trace, ...]: ...


Value = Callable[[Pool, tuple[float, ...]], float | None]


@dataclass(slots=True)
class PolicyRuntime:
    """Load `policy.py` and run it through a Roller."""

    root: Path
    roller: Runner
    denied: frozenset[str] = frozenset()
    value: Value | None = None
    _classes: dict[int, type] = field(default_factory=dict, init=False, repr=False)
    _policies: dict[int, object] = field(default_factory=dict, init=False, repr=False)
    _streams: dict[int, tuple[str, str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = _absolute(self.root)

    def scan(self, snap: Snap, task: Task) -> Verdict | None:
        system = self._system(snap)
        policy = system / "policy.py"
        if not policy.exists():
            return Verdict.missing_policy

        has_policy = False
        for path in system.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                return Verdict.import_error
            if path == policy:
                has_policy = any(
                    isinstance(node, ast.ClassDef) and node.name == "Policy"
                    for node in tree.body
                )
            if self._denied(tree):
                return Verdict.denied_import

        if not has_policy:
            return Verdict.missing_policy
        return None

    def load(self, snap: Snap, task: Task) -> Verdict | None:
        try:
            self._classes[snap.index] = self._load(snap)
        except FileNotFoundError:
            return Verdict.missing_policy
        except Exception:
            return Verdict.import_error
        return None

    def start(
        self,
        run: Run,
        snap: Snap,
        submit: Submit,
        task: Task,
        pool: Pool,
    ) -> Verdict | None:
        cls = self._classes.get(snap.index)
        if cls is None:
            verdict = self.load(snap, task)
            if verdict is not None:
                return verdict
            cls = self._classes[snap.index]

        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with _project(self._system(snap)):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    self._policies[snap.index] = cls(
                        task.obs,
                        task.act,
                        _meta(run, submit, task, pool),
                    )
            self._streams[snap.index] = (stdout.getvalue(), stderr.getvalue())
        except Exception:
            return Verdict.init_error
        return None

    def execute(self, snap: Snap, submit: Submit, task: Task, pool: Pool) -> Exec:
        policy = self._policies.get(snap.index)
        if policy is None:
            return Exec(
                verdict=Verdict.init_error,
                score=Score(mean=None, std=None),
                errors=("policy was not initialized",),
            )

        try:
            with _project(self._system(snap)):
                traces = self.roller.run(policy, task, pool, submit.cases)
            traces = self._with_start_streams(snap, traces)
        except MemoryError as exc:
            return _failed(Verdict.oom, exc)
        except TimeoutError as exc:
            return _failed(Verdict.rollout, exc)
        errors = tuple(trace.error for trace in traces if trace.error)
        verdict = Verdict.rollout if errors else Verdict.ok
        return Exec(
            verdict=verdict,
            score=_score(traces, pool, self.value),
            errors=errors,
            traces=traces,
        )

    def eval(
        self,
        snap: Snap,
        pool: Pool,
        task: Task,
        *,
        ready: Callable[[], None] | None = None,
    ) -> Score:
        if snap.cost is None:
            raise ValueError("hidden eval requires snapshot submit cost")

        cls = self._classes.get(snap.index)
        if cls is None:
            cls = self._load(snap)
            self._classes[snap.index] = cls

        cases = tuple(range(pool.size))
        view = Submit(index=snap.submit, cases=tuple(range(snap.cost)))
        run = Run(
            key="eval",
            model="eval",
            env=task.name,
            exp="eval",
            protocol="protocol/v2.0-draft",
            budget=_ZERO,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with _project(self._system(snap)):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                policy = cls(task.obs, task.act, _meta(run, view, task, pool))
            if ready is not None:
                ready()
            traces = self.roller.run(policy, task, pool, cases)
        return _score(traces, pool, self.value)

    def _system(self, snap: Snap) -> Path:
        return self.root / snap.ref

    def _with_start_streams(
        self,
        snap: Snap,
        traces: tuple[Trace, ...],
    ) -> tuple[Trace, ...]:
        stdout, stderr = self._streams.pop(snap.index, ("", ""))
        if not traces or (not stdout and not stderr):
            return traces
        first = replace(
            traces[0],
            stdout=stdout + traces[0].stdout,
            stderr=stderr + traces[0].stderr,
        )
        return (first, *traces[1:])

    def _load(self, snap: Snap) -> type:
        system = self._system(snap)
        policy = system / "policy.py"
        if not policy.exists():
            raise FileNotFoundError(policy)

        name = f"_evopolicygym_policy_{snap.index}"
        spec = importlib.util.spec_from_file_location(name, policy)
        if spec is None or spec.loader is None:
            raise ImportError(policy)

        module = importlib.util.module_from_spec(spec)
        with _project(system):
            _purge_helpers(system)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return _policy(module)

    def _denied(self, tree: ast.AST) -> bool:
        if not self.denied:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = (node.module,)
            else:
                continue
            for name in names:
                if name.split(".", 1)[0] in self.denied:
                    return True
        return False


_ZERO = Budget(limit=0)


def _meta(run: Run, submit: Submit, task: Task, pool: Pool) -> dict[str, Any]:
    left = max(0, run.budget.left - submit.cost) if pool.kind == PoolKind.train else 0
    meta: dict[str, Any] = {
        "env": task.name,
        "submit_index": submit.index,
        "n_episodes_this_submit": len(submit.cases),
        "remaining_budget_after": left,
        "max_episode_steps": task.steps,
        "obs_space": task.obs,
        "action_space": task.act,
        "obs_storage": task.storage,
    }
    if task.rewards:
        meta["reward_components"] = task.rewards
    return meta


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path.cwd() / value


def _policy(module: ModuleType) -> type:
    policy = getattr(module, "Policy", None)
    if not isinstance(policy, type):
        raise AttributeError("policy.py must define class Policy")
    return policy


def _score(traces: tuple[Trace, ...], pool: Pool, value: Value | None) -> Score:
    returns = tuple(trace.reward for trace in traces)
    if not returns:
        return Score(mean=None, std=None)
    if any(trace.error for trace in traces):
        return Score(mean=None, std=None, value=None, returns=returns)
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / len(returns)
    scalar = value(pool, returns) if value is not None else None
    return Score(mean=mean, std=sqrt(variance), value=scalar, returns=returns)


def _purge_helpers(system: Path) -> None:
    # Helper modules live next to policy.py in the snapshot's system/ dir.
    # Python caches them in sys.modules across submits, so a later submit that
    # changes a helper's signature would otherwise still see the stale version.
    stems = {p.stem for p in system.glob("*.py") if p.stem != "policy"}
    if not stems:
        return
    for name in list(sys.modules):
        head = name.split(".", 1)[0]
        if head in stems:
            del sys.modules[name]


def _failed(verdict: Verdict, exc: BaseException) -> Exec:
    return Exec(
        verdict=verdict,
        score=Score(mean=None, std=None),
        errors=(f"{type(exc).__name__}: {exc}",),
    )


@contextmanager
def _project(path: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
