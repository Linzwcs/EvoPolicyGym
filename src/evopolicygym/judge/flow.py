"""Application flow for EvoPolicyGym.

This layer expresses protocol transitions. It still delegates all I/O,
sandbox execution, environment rollout, and schema writing to ports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..core import Budget, Eval, Feed, Pool, Run, Runs, Runtime, Snap, Store, Task
from ..core import Submit as SubmitRecord
from ..protocol import PROTOCOL
from .close import JudgeClose
from .submit import JudgeSubmit, Limits


@dataclass(frozen=True, slots=True)
class Open:
    store: Runs

    def __call__(
        self,
        *,
        key: str,
        model: str,
        env: str,
        exp: str,
        budget: int,
        protocol: str = PROTOCOL,
    ) -> Run:
        run = Run(
            key=key,
            model=model,
            env=env,
            exp=exp,
            protocol=protocol,
            budget=Budget(limit=budget),
        )
        self.store.open(run)
        return run


@dataclass(frozen=True, slots=True)
class Submit:
    store: Store
    runtime: Runtime
    limits: Limits

    def __call__(
        self,
        run: Run,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
    ) -> tuple[Run, Feed]:
        outcome = JudgeSubmit(self.store, self.runtime)(
            run, submit, task, pool, self.limits
        )
        return outcome.run, outcome.feed


@dataclass(frozen=True, slots=True)
class Close:
    store: Store
    runtime: Runtime

    def __call__(
        self,
        run: Run,
        snaps: Iterable[Snap],
        task: Task,
        valid: Pool,
        final: Pool,
    ) -> tuple[Run, Eval | None]:
        outcome = JudgeClose(self.store, self.runtime)(run, snaps, task, valid, final)
        return outcome.run, outcome.final
