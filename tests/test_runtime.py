from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evopolicygym import (
    Budget,
    Feed,
    Pool,
    PoolKind,
    Run,
    Score,
    Snap,
    SubmitRecord,
    Task,
    Trace,
    Verdict,
)
from evopolicygym.infra.runtime import PolicyRuntime


@dataclass(slots=True)
class RecordingRoller:
    calls: list[tuple[dict[str, Any], PoolKind, tuple[int, ...]]] = field(
        default_factory=list
    )

    def run(
        self,
        policy: object,
        task: Task,
        pool: Pool,
        cases: tuple[int, ...],
    ) -> tuple[Trace, ...]:
        if hasattr(policy, "cwd"):
            self.assert_cwd(policy.cwd)
        self.calls.append((policy.env_meta, pool.kind, cases))
        return tuple(
            Trace(
                episode=case,
                reward=float(case),
                steps=({"t": 0}, {"t": 1}),
            )
            for case in cases
        )

    def assert_cwd(self, cwd: str) -> None:
        if cwd != str(Path.cwd()):
            raise AssertionError("policy lifecycle must run inside checkpoint dir")


def make_run() -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=5),
    )


def make_task() -> Task:
    return Task(
        name="toy",
        version="0.1",
        obs={"type": "Box"},
        act={"type": "Discrete", "n": 2},
        steps=20,
        cases=8,
    )


class PolicyRuntimeTest(unittest.TestCase):
    def test_scan_reports_missing_policy_and_denied_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, "import socket\nclass Policy: pass\n")
            runtime = PolicyRuntime(root, RecordingRoller(), denied=frozenset({"socket"}))

            self.assertEqual(
                runtime.scan(Snap(2, 2, "missing"), make_task()),
                Verdict.missing_policy,
            )
            self.assertEqual(runtime.scan(snap, make_task()), Verdict.denied_import)

    def test_execute_uses_started_policy_and_scores_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _policy_source())
            roller = RecordingRoller()
            runtime = PolicyRuntime(root, roller)
            run = make_run()
            task = make_task()
            submit = SubmitRecord(index=1, cases=(2, 4))
            pool = Pool(kind=PoolKind.train, size=8, ref="train")

            self.assertIsNone(runtime.scan(snap, task))
            self.assertIsNone(runtime.load(snap, task))
            self.assertIsNone(runtime.start(run, snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertEqual(result.verdict, Verdict.ok)
            self.assertEqual(result.score.returns, (2.0, 4.0))
            self.assertEqual(result.score.mean, 3.0)
            self.assertEqual(Feed.from_exec(submit, result).lengths, (2, 2))

            meta, kind, cases = roller.calls[0]
            self.assertEqual(kind, PoolKind.train)
            self.assertEqual(cases, (2, 4))
            self.assertEqual(meta["submit_index"], 1)
            self.assertEqual(meta["n_episodes_this_submit"], 2)
            self.assertEqual(meta["remaining_budget_after"], 3)
            self.assertEqual(meta["max_episode_steps"], 20)

    def test_start_maps_policy_init_error_to_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(
                root,
                "class Policy:\n"
                "    def __init__(self, obs_space, action_space, env_meta):\n"
                "        raise RuntimeError('bad init')\n",
            )
            runtime = PolicyRuntime(root, RecordingRoller())

            self.assertIsNone(runtime.load(snap, make_task()))
            verdict = runtime.start(
                make_run(),
                snap,
                SubmitRecord(index=1, cases=(0,)),
                make_task(),
                Pool(kind=PoolKind.train, size=8, ref="train"),
            )

            self.assertEqual(verdict, Verdict.init_error)

    def test_execute_attaches_policy_init_streams_to_first_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _stream_policy_source())
            runtime = PolicyRuntime(root, RecordingRoller())
            run = make_run()
            task = make_task()
            submit = SubmitRecord(index=1, cases=(2, 4))
            pool = Pool(kind=PoolKind.train, size=8, ref="train")

            self.assertIsNone(runtime.load(snap, task))
            self.assertIsNone(runtime.start(run, snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertIn("init out", result.traces[0].stdout)
            self.assertIn("init err", result.traces[0].stderr)
            self.assertEqual(result.traces[1].stdout, "")

    def test_eval_loads_policy_and_applies_value_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _policy_source())
            roller = RecordingRoller()
            runtime = PolicyRuntime(
                root,
                roller,
                value=lambda pool, returns: 42.0 if pool.kind == PoolKind.final else None,
            )

            score = runtime.eval(
                snap,
                Pool(kind=PoolKind.final, size=3, ref="heldout"),
                make_task(),
            )

            self.assertEqual(
                score,
                Score(
                    mean=1.0,
                    std=(2 / 3) ** 0.5,
                    value=42.0,
                    returns=(0.0, 1.0, 2.0),
                ),
            )
            meta, kind, cases = roller.calls[0]
            self.assertEqual(kind, PoolKind.final)
            self.assertEqual(cases, (0, 1, 2))
            self.assertEqual(meta["submit_index"], 1)
            self.assertEqual(meta["n_episodes_this_submit"], 2)
            self.assertEqual(meta["remaining_budget_after"], 0)
            self.assertNotIn("env_instances", meta)
            self.assertNotIn("pool", meta)

    def test_eval_requires_snapshot_cost_for_non_leaking_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _snap(root, _policy_source())
            snap = Snap(
                index=1,
                submit=1,
                ref="checkpoints/submit_001",
            )
            runtime = PolicyRuntime(root, RecordingRoller())

            with self.assertRaisesRegex(ValueError, "submit cost"):
                runtime.eval(
                    snap,
                    Pool(kind=PoolKind.valid, size=3, ref="validation"),
                    make_task(),
                )


def _snap(root: Path, policy: str) -> Snap:
    system = root / "checkpoints" / "submit_001"
    system.mkdir(parents=True, exist_ok=True)
    (system / "policy.py").write_text(policy, encoding="utf-8")
    return Snap(index=1, submit=1, ref="checkpoints/submit_001", cost=2)


def _policy_source() -> str:
    return (
        "import os\n"
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        self.obs_space = obs_space\n"
        "        self.action_space = action_space\n"
        "        self.env_meta = env_meta\n"
        "        self.cwd = os.getcwd()\n"
    )


def _stream_policy_source() -> str:
    return (
        "import os\n"
        "import sys\n"
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        print('init out')\n"
        "        print('init err', file=sys.stderr)\n"
        "        self.env_meta = env_meta\n"
        "        self.cwd = os.getcwd()\n"
    )


if __name__ == "__main__":
    unittest.main()
