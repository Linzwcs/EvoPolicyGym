from __future__ import annotations

import hashlib
import importlib.util
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
from evopolicygym.infra.fs import FileStore

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def make_run() -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=4),
    )


class FileStoreTest(unittest.TestCase):
    def test_snap_feed_mirror_and_close_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root, versions={"harness": "0.1.0"})
            run = make_run()
            submit = SubmitRecord(index=1, cases=(0, 1))

            store.open(run)
            self.assertTrue(store.agents.exists())
            rules = store.agents.read_text(encoding="utf-8")
            self.assertIn("EvoPolicyGym Agent Rules", rules)
            self.assertIn("system/policy.py", rules)
            self.assertNotIn("workspace/system", rules)
            policy = store.workspace / "policy.py"
            policy.write_text("VERSION = 1\n", encoding="utf-8")

            snap = store.snap(run, submit)
            metrics = _read_json(root / "checkpoints" / "submit_001" / "metrics.json")
            self.assertEqual(metrics["source_lines"], 1)
            self.assertTrue(str(metrics["tree_hash"]).startswith("sha256:"))
            policy.write_text("VERSION = 2\n", encoding="utf-8")

            feed = Feed(
                submit=1,
                verdict=Verdict.ok,
                cost=2,
                score=Score(mean=2.0, std=1.0, returns=(1.0, 3.0)),
                lengths=(5, 7),
            )
            traces = (
                Trace(episode=0, reward=1.0, steps=_steps(5), stdout="hello\n"),
                Trace(episode=1, reward=3.0, steps=_steps(7), stderr="warn\n"),
            )
            charged = run.charge(submit.cost, Verdict.ok)
            store.feed(charged, submit, _report(feed, traces=traces))

            feedback = root / "workspace" / "feedback"
            summary = _read_json(feedback / "submit_001" / "summary.json")
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["env_instances"], [0, 1])
            self.assertEqual(summary["remaining_budget"], 2)
            self.assertEqual(summary["episode_lengths"], [5, 7])
            first = feedback / "submit_001" / "episodes" / "ep_000"
            second = feedback / "submit_001" / "episodes" / "ep_001"
            self.assertEqual(_line_count(first / "trajectory.jsonl"), 5)
            self.assertEqual(_line_count(second / "trajectory.jsonl"), 7)
            self.assertEqual((first / "stdout.txt").read_text(encoding="utf-8"), "hello\n")
            self.assertEqual((second / "stderr.txt").read_text(encoding="utf-8"), "warn\n")

            pick = Pick.from_vals({1: Score(mean=10.0, std=0.0)})
            closed = charged.done(pick)
            final = Eval(
                kind=PoolKind.final,
                snap=1,
                pool="heldout",
                score=Score(mean=8.0, std=0.5, value=90.0, returns=(7.0, 9.0)),
            )
            store.eval(closed, final)
            store.mirror(closed, snap)
            store.close(closed)

            self.assertEqual(policy.read_text(encoding="utf-8"), "VERSION = 1\n")
            self.assertFalse((store.workspace / "metrics.json").exists())
            run_json = _read_json(root / "run.json")
            self.assertEqual(run_json["outcome"]["status"], "completed")
            self.assertEqual(run_json["outcome"]["best_submit_index"], 1)
            self.assertEqual(run_json["outcome"]["final_score"], 90.0)
            self.assertEqual(run_json["versions"]["harness"], "0.1.0")
            self.assertEqual(
                run_json["versions"]["agents_md_hash"],
                _sha(store.agents),
            )
            self.assertEqual(run_json["artifacts"]["workspace"], "workspace/")
            self.assertEqual(run_json["artifacts"]["feedback"], "workspace/feedback/")
            self.assertEqual(run_json["artifacts"]["logs_harness"], "logs/harness.log")
            auxiliary = run_json["outcome"]["auxiliary"]
            self.assertEqual(auxiliary["code_metrics_best"]["source_lines"], 1)
            self.assertEqual(auxiliary["code_metrics_by_submit"]["1"]["source_lines"], 1)
            self.assertEqual(auxiliary["code_metrics_trend"]["submits"], [1])
            events = [row["event"] for row in _read_jsonl(root / "logs" / "harness.log")]
            self.assertIn("run.open", events)
            self.assertIn("submit.snapshot", events)
            self.assertIn("submit.feedback", events)
            self.assertIn("eval.record", events)
            self.assertIn("workspace.mirror", events)
            self.assertIn("run.close", events)

    def test_feed_writes_episode_error_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root)
            run = make_run()
            submit = SubmitRecord(index=0, cases=(0,))
            feed = Feed(
                submit=0,
                verdict=Verdict.ok,
                cost=1,
                score=Score(mean=0.0, std=0.0, returns=(0.0,)),
                lengths=(0,),
            )
            trace = Trace(
                episode=0,
                reward=0.0,
                steps=(),
                error="reset_error: ValueError: bad reset",
            )

            store.open(run)
            store.feed(run.charge(1, Verdict.ok), submit, _report(feed, traces=(trace,)))

            feedback = root / "workspace" / "feedback"
            summary = _read_json(feedback / "submit_000" / "summary.json")
            error = _read_jsonl(
                feedback / "submit_000" / "episodes" / "ep_000" / "error.txt"
            )[0]

            self.assertEqual(summary["errors"], [0])
            self.assertFalse((feedback / "submit_000" / "errors.txt").exists())
            self.assertEqual(error["category"], "reset_error")

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required for observation artifacts")
    def test_feed_externalizes_large_observations(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root)
            run = make_run()
            submit = SubmitRecord(index=0, cases=(0,))
            feed = Feed(
                submit=0,
                verdict=Verdict.ok,
                cost=1,
                score=Score(mean=2.0, std=0.0, returns=(2.0,)),
                lengths=(2,),
            )
            obs = np.zeros((64, 64, 3), dtype=np.uint8).tolist()
            trace = Trace(
                episode=0,
                reward=2.0,
                steps=(
                    {"t": 0, "obs": obs, "action": 0, "reward": 1.0},
                    {"t": 1, "obs": obs, "action": 0, "reward": 1.0},
                ),
            )

            store.open(run)
            store.feed(run.charge(1, Verdict.ok), submit, _report(feed, traces=(trace,)))

            episode = root / "workspace" / "feedback" / "submit_000" / "episodes" / "ep_000"
            rows = _read_jsonl(episode / "trajectory.jsonl")
            observations = np.load(episode / "observations.npy", allow_pickle=False)

            self.assertEqual([row["obs"] for row in rows], [None, None])
            self.assertEqual(observations.shape, (2, 64, 64, 3))
            self.assertEqual(observations.dtype, np.dtype("uint8"))

    def test_close_no_ok_submit_writes_zero_score_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root)
            run = make_run()

            store.open(run)
            closed = run.done(Pick.from_vals({}))
            store.close(closed)

            run_json = _read_json(root / "run.json")
            self.assertEqual(run_json["outcome"]["status"], "no_ok_submit")
            self.assertEqual(run_json["outcome"]["final_score"], 0.0)
            self.assertIsNone(run_json["outcome"]["best_submit_index"])

    def test_close_failed_run_writes_error_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root)
            run = make_run()

            store.open(run)
            store.close(run.fail())

            run_json = _read_json(root / "run.json")
            self.assertEqual(run_json["outcome"]["status"], "error")
            self.assertEqual(run_json["outcome"]["error"]["type"], "error")
            self.assertIsNone(run_json["outcome"]["final_score"])

    def test_open_rejects_active_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            first = FileStore(root)
            second = FileStore(root)

            first.open(make_run())
            try:
                self.assertTrue(first.lock.exists())
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    second.open(make_run())
            finally:
                first.release_lock()

    def test_open_rejects_existing_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root)
            run = make_run()

            store.open(run)
            store.close(run.done(Pick.from_vals({})))

            with self.assertRaisesRegex(FileExistsError, "run already exists"):
                FileStore(root).open(run)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


def _steps(count: int) -> tuple[dict, ...]:
    return tuple({"t": index, "obs": index, "action": 1, "reward": 1.0} for index in range(count))


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    unittest.main()
