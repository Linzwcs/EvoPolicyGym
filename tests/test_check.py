from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from evopolicygym import (
    Budget,
    Eval,
    Feed,
    Pick,
    PoolKind,
    Report,
    Run,
    Score,
    SubmitRecord,
    Trace,
    Verdict,
)
from evopolicygym.check import check
from evopolicygym.infra.fs import FileStore


def make_run() -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=2),
    )


class CheckTest(unittest.TestCase):
    def test_check_accepts_valid_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))

            report = check(root)

            self.assertTrue(report.ok, report.issues)

    def test_check_reports_budget_conservation_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            summary = root / "workspace" / "feedback" / "submit_000" / "summary.json"
            data = _json(summary)
            data["remaining_budget"] = 2
            summary.write_text(json.dumps(data), encoding="utf-8")

            report = check(root)

            self.assertIn("budget_conservation", _codes(report))

    def test_check_reports_feedback_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            (root / "feedback").mkdir()

            report = check(root)

            self.assertIn("feedback_location", _codes(report))

    def test_check_reports_workspace_mirror_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            (root / "workspace" / "system" / "policy.py").write_text(
                "VERSION = 2\n",
                encoding="utf-8",
            )

            report = check(root)

            self.assertIn("workspace_mirror", _codes(report))

    def test_check_reports_missing_agents_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            (root / "workspace" / "AGENTS.md").unlink()

            report = check(root)

            self.assertIn("missing_agents", _codes(report))

    def test_check_reports_agents_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            (root / "workspace" / "AGENTS.md").write_text("changed\n", encoding="utf-8")

            report = check(root)

            self.assertIn("agents_hash", _codes(report))

    def test_check_reports_trajectory_length_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            trajectory = (
                root
                / "workspace"
                / "feedback"
                / "submit_000"
                / "episodes"
                / "ep_000"
                / "trajectory.jsonl"
            )
            trajectory.write_text("", encoding="utf-8")

            report = check(root)

            self.assertIn("trajectory_length", _codes(report))

    def test_check_reports_missing_external_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            trajectory = (
                root
                / "workspace"
                / "feedback"
                / "submit_000"
                / "episodes"
                / "ep_000"
                / "trajectory.jsonl"
            )
            trajectory.write_text(
                json.dumps({"t": 0, "obs": None, "action": 0, "reward": 1.0}) + "\n",
                encoding="utf-8",
            )

            report = check(root)

            self.assertIn("missing_observations", _codes(report))

    def test_check_parses_json_string_val_score_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            data = _json(root / "run.json")
            self.assertEqual(data["outcome"]["val_scores"], {"0": 5.0})

            report = check(root)

            self.assertNotIn("val_scores", _codes(report))

    def test_check_reports_missing_checkpoint_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            (root / "checkpoints" / "submit_000" / "metrics.json").unlink()

            report = check(root)

            self.assertIn("metrics_missing", _codes(report))

    def test_check_reports_stale_checkpoint_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            path = root / "checkpoints" / "submit_000" / "metrics.json"
            data = _json(path)
            data["source_lines"] = 999
            _write_json(path, data)

            report = check(root)

            self.assertIn("metrics_mismatch", _codes(report))

    def test_check_reports_missing_metrics_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            path = root / "run.json"
            data = _json(path)
            del data["outcome"]["auxiliary"]["code_metrics_by_submit"]
            _write_json(path, data)

            report = check(root)

            self.assertIn("metrics_auxiliary", _codes(report))

    def test_check_reports_best_metrics_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            path = root / "run.json"
            data = _json(path)
            data["outcome"]["auxiliary"]["code_metrics_best"]["source_lines"] = 999
            _write_json(path, data)

            report = check(root)

            self.assertIn("metrics_best", _codes(report))

    def test_check_reports_metrics_trend_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _completed(Path(tmp))
            path = root / "run.json"
            data = _json(path)
            data["outcome"]["auxiliary"]["code_metrics_trend"]["submits"] = [1]
            _write_json(path, data)

            report = check(root)

            self.assertIn("metrics_auxiliary", _codes(report))


def _completed(base: Path) -> Path:
    root = base / "run"
    store = FileStore(root)
    run = make_run()
    submit = SubmitRecord(index=0, cases=(0, 1))
    store.open(run)
    (store.workspace / "policy.py").write_text("VERSION = 1\n", encoding="utf-8")

    snap = store.snap(run, submit).with_verdict(Verdict.ok)
    feed = Feed(
        submit=0,
        verdict=Verdict.ok,
        cost=2,
        score=Score(mean=2.0, std=1.0, returns=(1.0, 3.0)),
        lengths=(1, 1),
    )
    charged = run.charge(submit.cost, Verdict.ok)
    store.feed(
        charged,
        submit,
        _report(
            feed,
            traces=(
                Trace(episode=0, reward=1.0, steps=({"t": 0},)),
                Trace(episode=1, reward=3.0, steps=({"t": 0},)),
            ),
        ),
    )

    pick = Pick.from_vals({0: Score(mean=5.0, std=0.0)})
    closed = charged.done(pick)
    final = Eval(
        kind=PoolKind.final,
        snap=0,
        pool="heldout",
        score=Score(mean=4.0, std=0.0, value=80.0, returns=(4.0, 4.0)),
    )
    store.eval(closed, final)
    store.mirror(closed, snap)
    store.close(closed)
    return root


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(feed: Feed, *, traces: tuple[Trace, ...] = ()) -> Report:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Report(
        feed=feed,
        started=stamp,
        completed=stamp,
        wall=0.0,
        first=0,
        traces=traces,
    )


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


if __name__ == "__main__":
    unittest.main()
