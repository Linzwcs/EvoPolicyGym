from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from evopolicygym import Budget, Case, Pool, PoolKind, Run, Snap, SubmitRecord, Task, Verdict
from evopolicygym.infra.runtime import Roller, Sandbox, SandboxRuntime, Turn


def make_run() -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=4),
    )


def make_task() -> Task:
    return Task(name="toy", version="0.1", obs={}, act={}, steps=2, cases=8)


@dataclass(slots=True)
class ToyWorld:
    state: int = 0

    def reset(self, case: Case) -> int:
        self.state = case.id
        return self.state

    def step(self, action: int) -> Turn:
        self.state += int(action)
        return Turn(obs=self.state, reward=float(self.state), terminated=True)

    def sample(self) -> int:
        return 0


class SandboxRuntimeTest(unittest.TestCase):
    def test_sandbox_rejects_invalid_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout"):
            Sandbox(rollout=0.0)
        with self.assertRaisesRegex(ValueError, "memory"):
            Sandbox(memory=0)
        with self.assertRaisesRegex(ValueError, "context"):
            Sandbox(context="missing-context")

    def test_execute_runs_policy_in_child_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _policy())
            runtime = SandboxRuntime(root, Roller(ToyWorld), sandbox=Sandbox(rollout=5.0))
            submit = SubmitRecord(index=0, cases=(0, 1))
            task = make_task()
            pool = Pool(kind=PoolKind.train, size=8, ref="train")

            self.assertIsNone(runtime.scan(snap, task))
            self.assertIsNone(runtime.load(snap, task))
            self.assertIsNone(runtime.start(make_run(), snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertEqual(result.verdict, Verdict.ok)
            self.assertEqual(result.score.returns, (1.0, 2.0))
            self.assertIn("init", result.traces[0].stdout)

    def test_start_is_not_limited_by_rollout_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _sleep_policy("__init__", seconds=0.2))
            runtime = SandboxRuntime(root, Roller(ToyWorld), sandbox=Sandbox(rollout=0.01))

            verdict = runtime.start(
                make_run(),
                snap,
                SubmitRecord(index=0, cases=(0,)),
                make_task(),
                Pool(kind=PoolKind.train, size=8, ref="train"),
            )

            self.assertIsNone(verdict)

    def test_execute_rollout_timeout_starts_after_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _sleep_policy("__init__", seconds=0.2))
            runtime = SandboxRuntime(root, Roller(ToyWorld), sandbox=Sandbox(rollout=0.1))
            submit = SubmitRecord(index=0, cases=(0,))
            task = make_task()
            pool = Pool(kind=PoolKind.train, size=8, ref="train")

            self.assertIsNone(runtime.start(make_run(), snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertEqual(result.verdict, Verdict.ok)

    def test_eval_rollout_timeout_starts_after_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _sleep_policy("__init__", seconds=0.2))
            runtime = SandboxRuntime(root, Roller(ToyWorld), sandbox=Sandbox(rollout=0.1))

            score = runtime.eval(
                snap,
                Pool(kind=PoolKind.valid, size=1, ref="valid"),
                make_task(),
            )

            self.assertEqual(score.mean, 1.0)

    def test_execute_timeout_maps_to_rollout_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _sleep_policy("act"))
            runtime = SandboxRuntime(root, Roller(ToyWorld), sandbox=Sandbox(rollout=0.1))
            submit = SubmitRecord(index=0, cases=(0,))
            task = make_task()
            pool = Pool(kind=PoolKind.train, size=8, ref="train")

            self.assertIsNone(runtime.start(make_run(), snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertEqual(result.verdict, Verdict.rollout)

    def test_execute_child_crash_maps_to_oom_with_exitcode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _snap(root, _crash_policy())
            runtime = SandboxRuntime(root, Roller(ToyWorld), sandbox=Sandbox(rollout=5.0))
            submit = SubmitRecord(index=0, cases=(0,))
            task = make_task()
            pool = Pool(kind=PoolKind.train, size=8, ref="train")

            self.assertIsNone(runtime.start(make_run(), snap, submit, task, pool))
            result = runtime.execute(snap, submit, task, pool)

            self.assertEqual(result.verdict, Verdict.oom)
            self.assertIn("exitcode=7", result.errors[0])


def _snap(root: Path, policy: str) -> Snap:
    system = root / "checkpoints" / "submit_000"
    system.mkdir(parents=True, exist_ok=True)
    (system / "policy.py").write_text(policy, encoding="utf-8")
    return Snap(index=0, submit=0, ref="checkpoints/submit_000", cost=1)


def _policy() -> str:
    return (
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        print('init')\n"
        "    def reset(self, episode_index):\n"
        "        self.episode_index = episode_index\n"
        "    def act(self, obs):\n"
        "        return 1\n"
    )


def _sleep_policy(where: str, *, seconds: float = 1.0) -> str:
    init = "        pass\n"
    act = "        return 1\n"
    if where == "__init__":
        init = f"        time.sleep({seconds!r})\n"
    if where == "act":
        act = f"        time.sleep({seconds!r})\n        return 1\n"
    return (
        "import time\n"
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        f"{init}"
        "    def reset(self, episode_index):\n"
        "        pass\n"
        "    def act(self, obs):\n"
        f"{act}"
    )


def _crash_policy() -> str:
    return (
        "import os\n"
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        pass\n"
        "    def reset(self, episode_index):\n"
        "        pass\n"
        "    def act(self, obs):\n"
        "        os._exit(7)\n"
    )


if __name__ == "__main__":
    unittest.main()
