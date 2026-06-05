from __future__ import annotations

import sys
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
from evopolicygym.infra.runtime.policy import _purge_helpers, _score


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

    def test_load_purges_stale_helper_modules_between_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_a = root / "checkpoints" / "submit_001"
            snap_a.mkdir(parents=True)
            (snap_a / "helper.py").write_text(
                "def choose(obs):\n    return obs + 1\n",
                encoding="utf-8",
            )
            (snap_a / "policy.py").write_text(
                "import helper\n"
                "class Policy:\n"
                "    def __init__(self, *a, **k): pass\n"
                "    def reset(self, episode_index): pass\n"
                "    def act(self, obs):\n"
                "        return helper.choose(obs)\n",
                encoding="utf-8",
            )
            snap_b = root / "checkpoints" / "submit_002"
            snap_b.mkdir(parents=True)
            (snap_b / "helper.py").write_text(
                "def choose(obs, scale):\n    return obs * scale\n",
                encoding="utf-8",
            )
            (snap_b / "policy.py").write_text(
                "import helper\n"
                "class Policy:\n"
                "    def __init__(self, *a, **k): pass\n"
                "    def reset(self, episode_index): pass\n"
                "    def act(self, obs):\n"
                "        return helper.choose(obs, 3)\n",
                encoding="utf-8",
            )
            runtime = PolicyRuntime(root, RecordingRoller())

            cls_a = runtime._load(
                Snap(index=1, submit=1, ref="checkpoints/submit_001", cost=1)
            )
            inst_a = cls_a({}, {}, {})
            self.assertEqual(inst_a.act(2), 3)

            cls_b = runtime._load(
                Snap(index=2, submit=2, ref="checkpoints/submit_002", cost=1)
            )
            inst_b = cls_b({}, {}, {})
            self.assertEqual(inst_b.act(4), 12)

    def test_execute_marks_trace_errors_as_rollout_failure(self) -> None:
        class ErrorRoller:
            def run(self, policy, task, pool, cases):
                return (
                    Trace(
                        episode=cases[0],
                        reward=-1.0,
                        steps=(),
                        error="act_error: TypeError: bad action",
                    ),
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _policy_source())
            runtime = PolicyRuntime(root, ErrorRoller())
            submit = SubmitRecord(index=1, cases=(0,))
            task = make_task()
            pool = Pool(kind=PoolKind.train, size=2, ref="train")

            self.assertIsNone(runtime.start(make_run(), snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertEqual(result.verdict, Verdict.rollout)
            self.assertEqual(result.errors, ("act_error: TypeError: bad action",))
            self.assertIsNone(result.score.mean)
            self.assertEqual(result.score.rank(0), float("-inf"))


class ScoreTest(unittest.TestCase):
    def test_score_returns_none_when_any_trace_errored(self) -> None:
        traces = (
            Trace(episode=0, reward=-1.0, steps=(), error="act_error: TypeError: x"),
            Trace(episode=1, reward=-500.0, steps=()),
        )
        score = _score(
            traces,
            Pool(kind=PoolKind.train, size=2, ref="train"),
            value=None,
        )
        self.assertIsNone(score.mean)
        self.assertIsNone(score.std)
        self.assertIsNone(score.value)
        self.assertIsNone(score.primary)
        self.assertEqual(score.returns, (-1.0, -500.0))

    def test_score_computes_mean_for_clean_traces(self) -> None:
        traces = (
            Trace(episode=0, reward=-2.0, steps=()),
            Trace(episode=1, reward=-4.0, steps=()),
        )
        score = _score(
            traces,
            Pool(kind=PoolKind.train, size=2, ref="train"),
            value=lambda pool, returns: sum(returns),
        )
        self.assertEqual(score.mean, -3.0)
        self.assertEqual(score.value, -6.0)
        self.assertEqual(score.returns, (-2.0, -4.0))

    def test_score_handles_empty_traces(self) -> None:
        score = _score((), Pool(kind=PoolKind.train, size=0, ref="train"), value=None)
        self.assertIsNone(score.mean)
        self.assertIsNone(score.std)


class PurgeHelpersTest(unittest.TestCase):
    def test_purge_removes_sibling_helpers_but_not_unrelated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            system = Path(tmp)
            (system / "policy.py").write_text("", encoding="utf-8")
            (system / "controller.py").write_text("", encoding="utf-8")
            sys.modules["controller"] = object()  # type: ignore[assignment]
            sys.modules["controller.helpers"] = object()  # type: ignore[assignment]
            sys.modules["unrelated"] = object()  # type: ignore[assignment]
            try:
                _purge_helpers(system)
                self.assertNotIn("controller", sys.modules)
                self.assertNotIn("controller.helpers", sys.modules)
                self.assertIn("unrelated", sys.modules)
            finally:
                sys.modules.pop("unrelated", None)

    def test_purge_skips_policy_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            system = Path(tmp)
            (system / "policy.py").write_text("", encoding="utf-8")
            sentinel = object()
            sys.modules["policy"] = sentinel  # type: ignore[assignment]
            try:
                _purge_helpers(system)
                self.assertIs(sys.modules.get("policy"), sentinel)
            finally:
                sys.modules.pop("policy", None)


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
