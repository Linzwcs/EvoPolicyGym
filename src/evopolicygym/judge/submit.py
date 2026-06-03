"""Submit lifecycle orchestration.

The objects here mirror protocol §5 without binding to filesystems or a
specific sandbox. Runtime and Store ports provide the external effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from ..core import (
    Feed,
    Phase,
    Pool,
    PoolKind,
    Report,
    Run,
    Runtime,
    Score,
    Snap,
    Store,
    Task,
    Trace,
    Verdict,
)
from ..core import (
    Submit as SubmitRecord,
)

Clock = Callable[[], datetime]
Timer = Callable[[], float]


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Limits:
    """Submit request bounds exposed through /info."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0:
            raise ValueError("minimum must be non-negative")
        if self.maximum < self.minimum:
            raise ValueError("maximum must be >= minimum")


@dataclass(frozen=True, slots=True)
class Step:
    phase: Phase
    verdict: Verdict | None = None

    @property
    def ok(self) -> bool:
        return self.verdict is None


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of judging one Submit."""

    run: Run
    submit: SubmitRecord
    feed: Feed
    report: Report
    summary: dict[str, Any]
    snap: Snap | None
    steps: tuple[Step, ...]

    @property
    def verdict(self) -> Verdict:
        return self.feed.verdict


@dataclass(frozen=True, slots=True)
class JudgeSubmit:
    """Judge one agent Submit through the v2 seven-phase lifecycle."""

    store: Store
    runtime: Runtime
    clock: Clock = _now
    timer: Timer = perf_counter

    def __call__(
        self,
        run: Run,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
        limits: Limits,
    ) -> Outcome:
        if pool.kind != PoolKind.train:
            raise ValueError("submit requires a train pool")
        if not run.alive():
            raise ValueError("run is not open")

        started = self.clock()
        begin = self.timer()

        reject = submit.reject(
            task,
            run.budget,
            minimum=limits.minimum,
            maximum=limits.maximum,
        )
        if reject is not None:
            feed = _empty_feed(submit, reject)
            next_run = run.charge(submit.cost, reject)
            report = self._report(run, feed, started, begin)
            body = self.store.feed(next_run, submit, report)
            self.store.save(next_run)
            return Outcome(
                run=next_run,
                submit=submit.with_verdict(reject),
                feed=feed,
                report=report,
                summary=body,
                snap=None,
                steps=(Step(Phase.request, reject),),
            )

        steps: list[Step] = [Step(Phase.request)]

        snap = self.store.snap(run, submit)
        steps.append(Step(Phase.snapshot))

        verdict = self.runtime.scan(snap, task)
        steps.append(Step(Phase.validate, verdict))
        if verdict is not None:
            return self._fail(run, submit, snap, verdict, steps, started, begin)

        verdict = self.runtime.load(snap, task)
        steps.append(Step(Phase.compile, verdict))
        if verdict is not None:
            return self._fail(run, submit, snap, verdict, steps, started, begin)

        verdict = self.runtime.start(run, snap, submit, task, pool)
        steps.append(Step(Phase.init, verdict))
        if verdict is not None:
            return self._fail(run, submit, snap, verdict, steps, started, begin)

        result = self.runtime.execute(snap, submit, task, pool)
        feed = Feed.from_exec(submit, result)
        steps.append(
            Step(Phase.execute, None if feed.verdict == Verdict.ok else feed.verdict)
        )
        next_run = run.charge(submit.cost, feed.verdict)
        report = self._report(run, feed, started, begin, result.traces)
        body = self.store.feed(next_run, submit, report)
        self.store.save(next_run)
        steps.append(Step(Phase.commit))

        judged = snap.with_verdict(feed.verdict)
        return Outcome(
            run=next_run,
            submit=submit.with_snap(snap.index).with_verdict(feed.verdict),
            feed=feed,
            report=report,
            summary=body,
            snap=judged,
            steps=tuple(steps),
        )

    def _fail(
        self,
        run: Run,
        submit: SubmitRecord,
        snap: Snap,
        verdict: Verdict,
        steps: list[Step],
        started: datetime,
        begin: float,
    ) -> Outcome:
        feed = _empty_feed(submit, verdict)
        next_run = run.charge(submit.cost, verdict)
        report = self._report(run, feed, started, begin)
        body = self.store.feed(next_run, submit, report)
        self.store.save(next_run)
        steps.append(Step(Phase.commit))
        judged = snap.with_verdict(verdict)
        return Outcome(
            run=next_run,
            submit=submit.with_snap(snap.index).with_verdict(verdict),
            feed=feed,
            report=report,
            summary=body,
            snap=judged,
            steps=tuple(steps),
        )

    def _report(
        self,
        before: Run,
        feed: Feed,
        started: datetime,
        begin: float,
        traces: tuple[Trace, ...] = (),
    ) -> Report:
        return Report(
            feed=feed,
            started=started,
            completed=self.clock(),
            wall=max(0.0, self.timer() - begin),
            first=None if feed.verdict.rejected else before.budget.used,
            traces=traces,
        )


def _empty_feed(submit: SubmitRecord, verdict: Verdict) -> Feed:
    return Feed(
        submit=submit.index,
        verdict=verdict,
        cost=submit.cost,
        score=Score(mean=None, std=None),
    )
