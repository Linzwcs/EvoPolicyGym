from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym import Result, Suite


class SuiteTest(unittest.TestCase):
    def test_expands_run_agent_matrix_with_repeats(self) -> None:
        suite = Suite.from_mapping(
            {
                "suite": {"root": "runs/smoke", "repeats": 2, "jobs": 3},
                "run": [
                    {
                        "env": "toy",
                        "budget": 2,
                        "maximum": 1,
                        "valid_size": 1,
                        "final_size": 1,
                    }
                ],
                "agent": [
                    {"kind": "command", "argv": ["python", "agent.py"], "name": "script"},
                    {"kind": "claude", "model": "sonnet"},
                ],
            }
        )

        self.assertEqual(suite.repeats, 2)
        self.assertEqual(suite.concurrency, 3)
        self.assertEqual(len(suite.jobs), 4)
        self.assertEqual([job.index for job in suite.jobs], [0, 1, 2, 3])
        self.assertEqual([job.repeat for job in suite.jobs], [0, 0, 1, 1])
        self.assertEqual(suite.jobs[0].agent, "script")
        self.assertEqual(suite.jobs[1].agent, "claude_sonnet")
        self.assertEqual(suite.jobs[0].spec.agent.kind, "command")
        self.assertEqual(suite.jobs[1].spec.agent.kind, "claude")
        self.assertEqual(
            suite.jobs[0].spec.root,
            Path("runs/smoke/script/toy/000_toy_script_r00"),
        )
        self.assertEqual(
            suite.jobs[1].spec.root,
            Path("runs/smoke/claude_sonnet/toy/001_toy_claude_sonnet_r00"),
        )
        self.assertEqual(suite.jobs[1].spec.model, "claude_sonnet")
        self.assertEqual(suite.jobs[1].spec.exp, "001_toy_claude_sonnet_r00")

    def test_writes_suite_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Suite.from_mapping(
                {
                    "suite": {"root": str(Path(tmp) / "suite")},
                    "run": {"env": "toy", "budget": 1},
                    "agent": {"kind": "command", "argv": ["python", "agent.py"]},
                }
            )
            job = suite.jobs[0]
            result = Result(
                job=job,
                done=True,
                reason="done",
                root=job.spec.root,
                run=job.spec.run_key,
                session="session-001",
                submits=1,
                check_ok=True,
            )

            path = suite.write([result])
            data = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(path, Path(tmp) / "suite" / "suite.json")
            self.assertTrue(data["done"])
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["passed"], 1)
            self.assertEqual(data["failed"], 0)
            self.assertEqual(data["jobs_configured"], 1)
            self.assertEqual(data["by_category"], {"completed": 1})
            self.assertEqual(data["jobs"][0]["session"], "session-001")
            self.assertTrue(data["jobs"][0]["check"]["ok"])

    def test_rejects_missing_suite_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "suite.root is required"):
            Suite.from_mapping({"run": {"budget": 1}, "agent": {"kind": "claude"}})


if __name__ == "__main__":
    unittest.main()
