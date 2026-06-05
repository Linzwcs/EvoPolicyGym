from __future__ import annotations

import json
import time
import tempfile
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from evopolicygym import (
    Budget,
    Caps,
    Eval,
    Exec,
    Feed,
    Pool,
    PoolKind,
    Report,
    Run,
    Score,
    Snap,
    SubmitRecord,
    Task,
    Trace,
    Verdict,
)
from evopolicygym.infra.fs import FileStore
from evopolicygym.infra.http import (
    ErrorResponse,
    Service,
    SubmitRequest,
    parse_cases,
)
from evopolicygym.judge import Limits
from evopolicygym.protocol import feedback as build_feedback


@dataclass(slots=True)
class MemoryStore:
    snaps: list[Snap] = field(default_factory=list)
    feeds: list[tuple[SubmitRecord, Feed]] = field(default_factory=list)
    saved: list[Run] = field(default_factory=list)
    closed: list[Run] = field(default_factory=list)
    evals: list[Eval] = field(default_factory=list)
    mirrors: list[tuple[Run, Snap]] = field(default_factory=list)

    def open(self, run: Run) -> None:
        return None

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
        return build_feedback(run, submit, report)

    def eval(self, run: Run, record: Eval) -> None:
        self.evals.append(record)

    def mirror(self, run: Run, snap: Snap) -> None:
        self.mirrors.append((run, snap))


@dataclass(slots=True)
class StubRuntime:
    def scan(self, snap: Snap, task: Task) -> Verdict | None:
        return None

    def load(self, snap: Snap, task: Task) -> Verdict | None:
        return None

    def start(
        self,
        run: Run,
        snap: Snap,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
    ) -> Verdict | None:
        return None

    def execute(
        self,
        snap: Snap,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
    ) -> Exec:
        return Exec(
            verdict=Verdict.ok,
            score=Score(mean=2.0, std=1.0, returns=(1.0, 3.0)),
            traces=(
                Trace(episode=submit.cases[0], reward=1.0, steps=({"t": 0},)),
                Trace(episode=submit.cases[1], reward=3.0, steps=({"t": 0},)),
            ),
        )

    def eval(self, snap: Snap, pool: Pool, task: Task) -> Score:
        return Score(mean=0.0, std=0.0)


@dataclass(slots=True)
class SlowRuntime(StubRuntime):
    delay: float = 0.2
    active: int = 0
    max_active: int = 0
    entered: Event = field(default_factory=Event)
    lock: Any = field(default_factory=Lock)

    def execute(
        self,
        snap: Snap,
        submit: SubmitRecord,
        task: Task,
        pool: Pool,
    ) -> Exec:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered.set()
        try:
            time.sleep(self.delay)
            return StubRuntime.execute(self, snap, submit, task, pool)
        finally:
            with self.lock:
                self.active -= 1


class HttpServiceTest(unittest.TestCase):
    def test_parse_cases_accepts_lists_and_specs(self) -> None:
        self.assertEqual(parse_cases([1, 2]), (1, 2))
        self.assertEqual(parse_cases("1,3-5"), (1, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "invalid range"):
            parse_cases("5-3")

    def test_info_and_task_do_not_expose_hidden_pools(self) -> None:
        service = _service(caps=Caps(observations=True))

        info = service.info().body()
        task = service.task_doc()

        self.assertEqual(info["state"]["remaining_budget"], 4)
        self.assertEqual(info["state"]["n_submits"], 0)
        self.assertEqual(info["env_meta"]["n_env_instances"], 8)
        self.assertNotIn("validation", info["env_meta"])
        self.assertEqual(info["env_meta"]["artifacts"], {"observations": True, "video": False})
        self.assertEqual(task.media, "text/markdown")
        self.assertIn("# Toy", task.text)
        self.assertIn("## Policy Contract", task.text)
        self.assertIn("def act(self, obs)", task.text)
        self.assertIn('"env_instances": [0, 1, 2, 3]', task.text)
        self.assertIn("## Policy Input And Output", task.text)
        self.assertIn("Input observation space", task.text)
        self.assertIn("Output action space", task.text)

    def test_submit_returns_summary_and_updates_state(self) -> None:
        service = _service()

        response = service.submit(SubmitRequest([0, 1]))

        self.assertEqual(response.code, 200)
        self.assertEqual(response.submit_id, 0)
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.summary["returns"], [1.0, 3.0])
        self.assertEqual(response.summary["remaining_budget"], 2)
        self.assertEqual(response.summary["episode_lengths"], [1, 1])
        self.assertEqual(service.info().state["remaining_budget"], 2)
        self.assertEqual(service.info().state["n_submits"], 1)

    def test_submit_response_matches_written_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root)
            run = _run()
            store.open(run)
            service = _service(run=run, store=store)

            response = service.submit(SubmitRequest([0, 1]))

            written = json.loads(
                (root / "workspace" / "feedback" / "submit_000" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(response.summary, written)

    def test_submit_parse_error_returns_400_and_increments(self) -> None:
        service = _service()

        response = service.submit(SubmitRequest(""))

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertEqual(response.status, "budget_invalid")
        self.assertEqual(service.submits, 1)

    def test_submit_skips_existing_artifact_indices(self) -> None:
        class SkippingStore(MemoryStore):
            def next_submit_index(self, start: int = 0) -> int:
                return 2 if start == 0 else start

        store = SkippingStore()
        service = _service(store=store)

        response = service.submit(SubmitRequest([0, 1]))

        self.assertEqual(response.code, 200)
        self.assertEqual(response.submit_id, 2)
        self.assertEqual(store.snaps[0].index, 2)
        self.assertEqual(service.submits, 3)

    def test_concurrent_submits_are_serialized(self) -> None:
        runtime = SlowRuntime()
        service = _service(runtime=runtime)
        responses: list[Any] = []
        errors: list[BaseException] = []

        def submit() -> None:
            try:
                responses.append(service.submit(SubmitRequest([0, 1])))
            except BaseException as exc:  # noqa: BLE001 - test captures thread failures.
                errors.append(exc)

        first = Thread(target=submit)
        second = Thread(target=submit)

        first.start()
        self.assertTrue(runtime.entered.wait(timeout=1.0))
        second.start()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(errors)
        self.assertEqual([response.submit_id for response in responses], [0, 1])
        self.assertEqual(service.submits, 2)
        self.assertEqual(runtime.max_active, 1)

    def test_phase_one_reject_returns_400_summary(self) -> None:
        service = _service()

        response = service.submit(SubmitRequest([0, 1, 2, 3, 4]))

        self.assertEqual(response.code, 400)
        self.assertEqual(response.status, "budget_invalid")
        self.assertEqual(response.summary["n_episodes"], 0)
        self.assertEqual(response.summary["remaining_budget"], 4)
        self.assertEqual(service.submits, 1)

    def test_submit_auto_closes_when_budget_is_exhausted(self) -> None:
        service = _service(
            budget=2,
            valid=Pool(kind=PoolKind.valid, size=2, ref="validation"),
            final=Pool(kind=PoolKind.final, size=2, ref="heldout"),
        )

        response = service.submit(SubmitRequest([0, 1]))

        self.assertEqual(response.code, 200)
        self.assertEqual(response.summary["remaining_budget"], 0)
        self.assertFalse(service.run.alive())
        self.assertTrue(service.info().state["is_finalized"])
        self.assertEqual(service.run.pick.best, 0)


def _service(
    *,
    budget: int = 4,
    run: Run | None = None,
    store: Any | None = None,
    runtime: Any | None = None,
    valid: Pool | None = None,
    final: Pool | None = None,
    caps: Caps | None = None,
) -> Service:
    return Service(
        run=run or _run(budget=budget),
        task=Task(name="toy", version="0.1", obs={}, act={}, steps=10, cases=8),
        train=Pool(kind=PoolKind.train, size=8, ref="train"),
        store=store or MemoryStore(),
        runtime=runtime or StubRuntime(),
        limits=Limits(minimum=1, maximum=4),
        valid=valid,
        final=final,
        caps=caps or Caps(),
        task_text="# Toy\n",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def _run(*, budget: int = 4) -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=budget),
    )


if __name__ == "__main__":
    unittest.main()
