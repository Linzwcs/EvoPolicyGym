from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym import (
    Budget,
    JudgeClose,
    JudgeSubmit,
    Limits,
    Pool,
    PoolKind,
    Run,
    SubmitRecord,
    Verdict,
)
from evopolicygym.check import check
from evopolicygym.envs import toy
from evopolicygym.infra.fs import FileStore
from evopolicygym.infra.runtime import PolicyRuntime, Roller


class FlowTest(unittest.TestCase):
    def test_submit_close_writes_summary_and_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            store = FileStore(root, versions={"harness": "test"})
            env = toy()
            runtime = PolicyRuntime(
                root,
                Roller(env.make),
                value=lambda pool, returns: 88.0 if pool.kind == PoolKind.final else None,
            )
            run = _run()
            task = env.task
            train = Pool(kind=PoolKind.train, size=4, ref="train")

            store.open(run)
            (store.workspace / "policy.py").write_text(_policy(), encoding="utf-8")

            submit = SubmitRecord(index=0, cases=(0, 1))
            judged = JudgeSubmit(store, runtime)(
                run,
                submit,
                task,
                train,
                Limits(minimum=1, maximum=2),
            )

            self.assertEqual(judged.verdict, Verdict.ok)
            self.assertEqual(judged.run.budget.used, 2)
            self.assertIsNotNone(judged.snap)

            feedback = root / "workspace" / "feedback"
            summary = _json(feedback / "submit_000" / "summary.json")
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["returns"], [1.0, 2.0])
            self.assertEqual(summary["episode_lengths"], [1, 1])
            stdout = (
                root
                / "workspace"
                / "feedback"
                / "submit_000"
                / "episodes"
                / "ep_000"
                / "stdout.txt"
            ).read_text(encoding="utf-8")
            stderr = (
                root
                / "workspace"
                / "feedback"
                / "submit_000"
                / "episodes"
                / "ep_000"
                / "stderr.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("init ready", stdout)
            self.assertIn("act 0", stdout)
            self.assertIn("reset 0", stderr)

            closed = JudgeClose(store, runtime)(
                judged.run,
                (judged.snap,),
                task,
                Pool(kind=PoolKind.valid, size=2, ref="validation"),
                Pool(kind=PoolKind.final, size=2, ref="heldout"),
            )

            self.assertEqual(closed.pick.best, 0)
            self.assertEqual(closed.final.score.value, 88.0)

            run_json = _json(root / "run.json")
            self.assertEqual(run_json["outcome"]["status"], "completed")
            self.assertEqual(run_json["outcome"]["best_submit_index"], 0)
            self.assertEqual(run_json["outcome"]["final_score"], 88.0)
            self.assertEqual(run_json["outcome"]["val_scores"], {"0": 1.5})
            self.assertTrue(check(root).ok)


def _run() -> Run:
    return Run(
        key="run-001",
        model="agent",
        env="toy",
        exp="smoke",
        protocol="protocol/v2.0-draft",
        budget=Budget(limit=2),
    )


def _policy() -> str:
    return (
        "import sys\n"
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        print('init ready')\n"
        "        self.env_meta = env_meta\n"
        "    def reset(self, episode_index):\n"
        "        print(f'reset {episode_index}', file=sys.stderr)\n"
        "        self.episode_index = episode_index\n"
        "    def act(self, obs):\n"
        "        print(f'act {self.episode_index}')\n"
        "        return 1\n"
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
