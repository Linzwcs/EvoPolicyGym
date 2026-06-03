"""Outer capabilities used by EvoPolicyGym flow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from .models import (
    Case,
    Env,
    Eval,
    Exec,
    Pool,
    Report,
    Run,
    Score,
    Snap,
    Submit,
    Task,
    Turn,
    Verdict,
)


class Runs(Protocol):
    """Run lifecycle persistence."""

    def open(self, run: Run) -> None: ...
    def save(self, run: Run) -> None: ...
    def close(self, run: Run) -> None: ...


class Snaps(Protocol):
    """Immutable Work snapshot persistence."""

    def snap(self, run: Run, submit: Submit) -> Snap: ...


class Feeds(Protocol):
    """Agent-visible feedback persistence."""

    def feed(self, run: Run, submit: Submit, report: Report) -> dict[str, Any]: ...


class Evals(Protocol):
    """Hidden validation and final score persistence."""

    def eval(self, run: Run, eval: Eval) -> None: ...


class Works(Protocol):
    """Agent workspace materialization."""

    def mirror(self, run: Run, snap: Snap) -> None: ...


class Store(Runs, Snaps, Feeds, Evals, Works, Protocol):
    """Complete artifact store facade."""


class Runtime(Protocol):
    def scan(self, snap: Snap, task: Task) -> Verdict | None: ...
    def load(self, snap: Snap, task: Task) -> Verdict | None: ...
    def start(
        self,
        run: Run,
        snap: Snap,
        submit: Submit,
        task: Task,
        pool: Pool,
    ) -> Verdict | None: ...
    def execute(self, snap: Snap, submit: Submit, task: Task, pool: Pool) -> Exec: ...
    def eval(self, snap: Snap, pool: Pool, task: Task) -> Score: ...


class World(Protocol):
    """Minimal environment adapter used by runtime rollers."""

    def reset(self, case: Case) -> Any: ...
    def step(self, action: Any) -> Turn: ...
    def sample(self) -> Any: ...


class Catalog(Protocol):
    def get(self, name: str) -> Env: ...
    def list(self) -> tuple[str, ...]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class Log(Protocol):
    def emit(self, event: str, **data: Any) -> None: ...
