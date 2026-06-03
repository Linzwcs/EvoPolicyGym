from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym import PoolKind, Sandbox, local
from evopolicygym.check import check
from evopolicygym.envs import toy
from evopolicygym.infra.http import SubmitRequest


class HostTest(unittest.TestCase):
    def test_local_host_uses_env_contract_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            host = local(
                Path(tmp) / "run",
                toy(),
                key="run-001",
                model="agent",
                exp="smoke",
                budget=1,
                maximum=1,
            )

            self.assertEqual(host.train.size, 8)
            self.assertEqual(host.valid.size, 64)
            self.assertEqual(host.final.size, 256)
            self.assertIn("additive", host.service.task_doc().text)

    def test_local_host_uses_external_case_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            _write_data(data, "train", [{"start": 10}, {"start": 20}])
            _write_data(data, "valid", [{"start": 30}, {"start": 40}])
            _write_data(data, "heldout", [{"start": 50}, {"start": 60}])
            root = Path(tmp) / "run"
            host = local(
                root,
                toy(),
                key="run-001",
                model="agent",
                exp="smoke",
                budget=1,
                maximum=1,
                valid_size=1,
                final_size=1,
                data=data,
            )

            self.assertEqual(host.env.task.cases, 2)
            self.assertEqual(host.train.case(1).data["start"], 20)
            self.assertEqual(host.valid.size, 1)
            self.assertEqual(host.final.case(0).data["start"], 50)
            self.assertEqual(host.store.versions["data_train_hash"], _sha(data / "train.json"))

    def test_local_host_assembles_submit_and_close_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            host = local(
                root,
                toy(),
                key="run-001",
                model="agent",
                exp="smoke",
                budget=2,
                maximum=2,
                valid_size=2,
                final_size=2,
                value=lambda pool, returns: 88.0 if pool.kind == PoolKind.final else None,
                versions={"harness": "test"},
            )

            (host.store.workspace / "policy.py").write_text(
                _policy(),
                encoding="utf-8",
            )
            response = host.service.submit(SubmitRequest([0, 1]))

            self.assertEqual(response.code, 200)
            self.assertEqual(response.summary["remaining_budget"], 0)
            self.assertFalse(host.run.alive())
            self.assertEqual(host.train.ref, "toy/train")
            self.assertEqual(host.valid.ref, "toy/validation")
            self.assertEqual(host.final.ref, "toy/heldout")
            self.assertTrue(check(root).ok)

            run_json = _json(root / "run.json")
            self.assertEqual(run_json["outcome"]["status"], "completed")
            self.assertEqual(run_json["outcome"]["final_score"], 88.0)
            events = [row["event"] for row in _jsonl(root / "logs" / "harness.log")]
            self.assertIn("submit.start", events)
            self.assertIn("submit.finish", events)
            self.assertIn("run.auto_close.start", events)
            self.assertIn("run.auto_close.finish", events)

    def test_local_host_can_use_sandbox_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            host = local(
                root,
                toy(),
                key="run-001",
                model="agent",
                exp="smoke",
                budget=1,
                maximum=1,
                valid_size=1,
                final_size=1,
                value=_final_value,
                sandbox=Sandbox(rollout=5.0),
            )

            (host.store.workspace / "policy.py").write_text(
                _policy(),
                encoding="utf-8",
            )
            response = host.service.submit(SubmitRequest([0]))

            self.assertEqual(response.status, "ok")
            self.assertEqual(host.service.info().resource_limits["rollout_wall_s"], 5.0)
            self.assertFalse(host.run.alive())


def _policy() -> str:
    return (
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        self.env_meta = env_meta\n"
        "    def reset(self, episode_index):\n"
        "        self.episode_index = episode_index\n"
        "    def act(self, obs):\n"
        "        return 1\n"
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_data(root: Path, split: str, cases: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"env": "toy", "split": split, "cases": cases}
    (root / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _final_value(pool, returns) -> float | None:
    return 1.0 if pool.kind == PoolKind.final else None


if __name__ == "__main__":
    unittest.main()
