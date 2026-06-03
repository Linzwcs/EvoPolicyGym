"""Run closing orchestration.

Closing is judge-owned: agents may submit freely while the run is open,
but the judge selects the best ok snapshot on the hidden validation pool
and evaluates that snapshot on the hidden final pool.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..core import (
    Eval,
    OutcomeStatus,
    Pick,
    Pool,
    PoolKind,
    Run,
    Runtime,
    Score,
    Snap,
    Store,
    Task,
)


@dataclass(frozen=True, slots=True)
class CloseOutcome:
    """Result of closing a Run."""

    run: Run
    pick: Pick
    vals: tuple[Eval, ...]
    final: Eval | None
    status: OutcomeStatus


@dataclass(frozen=True, slots=True)
class JudgeClose:
    """Close a run through validation selection and final evaluation."""

    store: Store
    runtime: Runtime

    def __call__(
        self,
        run: Run,
        snaps: Iterable[Snap],
        task: Task,
        valid: Pool,
        final: Pool,
    ) -> CloseOutcome:
        if valid.kind != PoolKind.valid or final.kind != PoolKind.final:
            raise ValueError("close requires valid and final pools")

        vals: list[Eval] = []
        scores: dict[int, Score] = {}
        snaps_by_index: dict[int, Snap] = {}
        for snap in snaps:
            if not snap.candidate:
                continue

            score = self.runtime.eval(snap, valid, task)
            record = Eval(
                kind=PoolKind.valid,
                snap=snap.index,
                pool=valid.ref,
                score=score,
            )
            vals.append(record)
            self.store.eval(run, record)

            if score.primary is not None:
                scores[snap.index] = score
                snaps_by_index[snap.index] = snap

        pick = Pick.from_vals(scores)
        if pick.best is None:
            closed = run.done(pick=pick)
            self.store.close(closed)
            return CloseOutcome(
                run=closed,
                pick=pick,
                vals=tuple(vals),
                final=None,
                status=_status(closed),
            )

        best = snaps_by_index[pick.best]
        score = self.runtime.eval(best, final, task)
        final_eval = Eval(
            kind=PoolKind.final,
            snap=best.index,
            pool=final.ref,
            score=score,
        )
        closed = run.done(pick=pick)
        self.store.eval(closed, final_eval)
        self.store.mirror(closed, best)
        self.store.close(closed)
        return CloseOutcome(
            run=closed,
            pick=pick,
            vals=tuple(vals),
            final=final_eval,
            status=_status(closed),
        )


def _status(run: Run) -> OutcomeStatus:
    if run.outcome is None:
        raise RuntimeError("closed run has no outcome status")
    return run.outcome
