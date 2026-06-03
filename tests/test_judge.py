from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from evopolicygym import (
    Budget,
    Eval,
    Exec,
    Feed,
    JudgeClose,
    JudgeSubmit,
    Limits,
    OutcomeStatus,
    Phase,
    Pool,
    PoolKind,
    Report,
    Run,
    RunState,
    Score,
    Snap,
    SubmitRecord,
    Task,
    Verdict,
)


@dataclass(slots=True)
class MemoryStore:
    opened: list[Run] = field(default_factory=list)
    saved: list[Run] = field(default_factory=list)
    closed: list[Run] = field(default_factory=list)
    snaps: list[Snap] = field(default_factory=list)
    feeds: list[tuple[SubmitRecord, Feed]] = field(default_factory=list)
    evals: list[Eval] = field(default_factory=list)
    mirrors: list[tuple[Run, Snap]] = field(default_factory=list)

    def open(self, run: Run) -> None:
        self.opened.append(run)

    def save(self, run: Run) -> None:
        self.saved.append(run)

    def close(self, run: Run) -> None:
        self.closed.append(run)

    def snap(self, run: Run, submit: SubmitRecord) -> Snap:
        snap = Snap(
            index=submit.index,
            submit=submit.index,
            ref=f"submit_{submit.index:03d}",
            cost=submit.cost,
        )
        self.snaps.append(snap)
        return snap

    def feed(self, run: Run, submit: SubmitRecord, report: Report) -> dict:
        self.feeds.append((submit, report.feed))
        return {"status": report.feed.verdict.value}

    def eval(self, run: Run, record: Eval) -> None:
        self.evals.append(record)

    def mirror(self, run: Run, snap: Snap) -> None:
        self.mirrors.append((run, snap))


@dataclass(slots=True)
class StubRuntime:
    execute_verdict: Verdict = Verdict.ok
    execute_score: Score = field(default_factory=lambda: Score(mean=1.0, std=0.0))
    scan_verdict: Verdict | None = None
    load_verdict: Verdict | None = None
    start_verdict: Verdict | None = None
    scores: dict[tuple[int, PoolKind], Score] = field(default_factory=dict)

    def scan(self, snap: Snap, task: Task) -> Verdict | None:
        return self.scan_verdict

    def load(self, snap: Snap, task: Task) -> Verdict | None:
        return self.load_verdict

    def start(
        self,
        run: Run,
        snap: Snap,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
    ) -> Verdict | None:
        return self.start_verdict

    def execute(
        self,
        snap: Snap,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
    ) -> Exec:
        return Exec(verdict=self.execute_verdict, score=self.execute_score)

    def eval(self, snap: Snap, pool: Pool, task: Task) -> Score:
        return self.scores[(snap.index, pool.kind)]


def make_run(*, budget: int = 4) -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0",
        budget=Budget(limit=budget),
    )


def make_task() -> Task:
    return Task(name="toy", version="0.1", obs={}, act={}, steps=10, cases=8)


class JudgeSubmitTest(unittest.TestCase):
    def test_request_reject_does_not_spend_or_snapshot(self) -> None:
        store = MemoryStore()
        runtime = StubRuntime()
        judge = JudgeSubmit(store, runtime)

        outcome = judge(
            make_run(budget=2),
            SubmitRecord(index=1, cases=(0, 1, 2)),
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            Limits(minimum=1, maximum=4),
        )

        self.assertEqual(outcome.verdict, Verdict.budget_invalid)
        self.assertEqual(outcome.run.budget.used, 0)
        self.assertIsNone(outcome.snap)
        self.assertEqual(store.snaps, [])
        self.assertEqual(store.feeds[0][0].cases, (0, 1, 2))
        self.assertEqual(store.feeds[0][1].verdict, Verdict.budget_invalid)
        self.assertEqual(outcome.steps[0].phase, Phase.request)
        self.assertEqual(outcome.steps[0].verdict, Verdict.budget_invalid)

    def test_successful_submit_spends_budget_and_records_feed(self) -> None:
        store = MemoryStore()
        runtime = StubRuntime(execute_score=Score(mean=7.0, std=0.5))
        judge = JudgeSubmit(store, runtime)

        outcome = judge(
            make_run(budget=4),
            SubmitRecord(index=2, cases=(0, 1)),
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            Limits(minimum=1, maximum=4),
        )

        self.assertEqual(outcome.verdict, Verdict.ok)
        self.assertEqual(outcome.run.budget.used, 2)
        self.assertEqual(outcome.submit.snap, 2)
        self.assertEqual(outcome.snap, Snap(2, 2, "submit_002", Verdict.ok, 2))
        self.assertEqual(store.feeds[0][1].score.mean, 7.0)
        self.assertEqual(store.saved[-1], outcome.run)

    def test_late_failure_spends_budget_and_returns_judged_snap(self) -> None:
        store = MemoryStore()
        runtime = StubRuntime(load_verdict=Verdict.import_error)
        judge = JudgeSubmit(store, runtime)

        outcome = judge(
            make_run(budget=4),
            SubmitRecord(index=3, cases=(0, 1)),
            make_task(),
            Pool(kind=PoolKind.train, size=8, ref="train"),
            Limits(minimum=1, maximum=4),
        )

        self.assertEqual(outcome.verdict, Verdict.import_error)
        self.assertEqual(outcome.run.budget.used, 2)
        self.assertEqual(
            outcome.snap,
            Snap(3, 3, "submit_003", Verdict.import_error, 2),
        )
        self.assertEqual(outcome.steps[-1].phase, Phase.commit)


class JudgeCloseTest(unittest.TestCase):
    def test_close_selects_best_validation_score_and_mirrors_workspace(self) -> None:
        store = MemoryStore()
        runtime = StubRuntime(
            scores={
                (1, PoolKind.valid): Score(mean=3.0, std=0.1),
                (2, PoolKind.valid): Score(mean=3.0, std=0.2),
                (2, PoolKind.final): Score(mean=11.0, std=0.3, value=95.0),
            }
        )
        judge = JudgeClose(store, runtime)
        snaps = (
            Snap(index=1, submit=1, ref="submit_001", verdict=Verdict.ok),
            Snap(index=2, submit=2, ref="submit_002", verdict=Verdict.ok),
        )

        outcome = judge(
            make_run(),
            snaps,
            make_task(),
            Pool(kind=PoolKind.valid, size=64, ref="validation"),
            Pool(kind=PoolKind.final, size=256, ref="heldout"),
        )

        self.assertEqual(outcome.status, OutcomeStatus.completed)
        self.assertEqual(outcome.run.state, RunState.closed)
        self.assertEqual(outcome.pick.best, 2)
        self.assertEqual(outcome.final.snap, 2)
        self.assertEqual(
            [record.kind for record in store.evals],
            [PoolKind.valid, PoolKind.valid, PoolKind.final],
        )
        self.assertEqual(store.mirrors, [(outcome.run, snaps[1])])
        self.assertEqual(store.closed, [outcome.run])

    def test_close_without_ok_candidate_does_not_eval_or_mirror(self) -> None:
        store = MemoryStore()
        runtime = StubRuntime()
        judge = JudgeClose(store, runtime)

        outcome = judge(
            make_run(),
            (Snap(index=1, submit=1, ref="submit_001", verdict=Verdict.import_error),),
            make_task(),
            Pool(kind=PoolKind.valid, size=64, ref="validation"),
            Pool(kind=PoolKind.final, size=256, ref="heldout"),
        )

        self.assertEqual(outcome.status, OutcomeStatus.no_ok_submit)
        self.assertIsNone(outcome.final)
        self.assertEqual(store.evals, [])
        self.assertEqual(store.mirrors, [])
        self.assertEqual(store.closed, [outcome.run])


if __name__ == "__main__":
    unittest.main()
