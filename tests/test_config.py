from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym.config import Spec, load


class ConfigTest(unittest.TestCase):
    def test_loads_toml_run_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                "\n".join(
                    (
                        "[run]",
                        "env = 'toy'",
                        "root = 'experiment/run-001'",
                        "data = 'data/cartpole'",
                        "budget = 8",
                        "maximum = 2",
                        "valid_size = 3",
                        "final_size = 4",
                        "[agent]",
                        "kind = 'command'",
                        "argv = ['python', 'agent.py']",
                        "name = 'runner'",
                        "limit = 9",
                        "retries = 2",
                        "retry_backoff = 0.25",
                        "[server]",
                        "bind = '127.0.0.1'",
                        "port = 8080",
                    )
                ),
                encoding="utf-8",
            )

            spec = load(path)

            self.assertEqual(spec.env, "toy")
            self.assertEqual(spec.root, Path("experiment/run-001"))
            self.assertEqual(spec.data, Path("data/cartpole"))
            self.assertEqual(spec.budget, 8)
            self.assertEqual(spec.maximum, 2)
            self.assertEqual(spec.valid_size, 3)
            self.assertEqual(spec.final_size, 4)
            self.assertEqual(spec.agent.argv, ("python", "agent.py"))
            self.assertEqual(spec.agent.name, "runner")
            self.assertEqual(spec.agent.limit, 9)
            self.assertEqual(spec.agent.retries, 2)
            self.assertEqual(spec.agent.retry_backoff, 0.25)
            self.assertEqual(spec.server.port, 8080)

    def test_loads_json_with_flat_run_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "root": "runs/flat",
                        "budget": 2,
                        "agent": {"argv": ["python", "agent.py"]},
                    }
                ),
                encoding="utf-8",
            )

            spec = load(path)

            self.assertEqual(spec.env, "toy")
            self.assertEqual(spec.root, Path("runs/flat"))
            self.assertEqual(spec.budget, 2)
            self.assertEqual(spec.agent.limit, 2)
            self.assertEqual(spec.run_key, "flat")

    def test_defaults_agent_limit_to_budget(self) -> None:
        spec = Spec.from_mapping(
            {
                "run": {
                    "root": "runs/default-limit",
                    "budget": 16,
                },
                "agent": {"kind": "codex"},
            }
        )

        self.assertEqual(spec.agent.limit, 16)

    def test_explicit_agent_limit_overrides_budget(self) -> None:
        spec = Spec.from_mapping(
            {
                "run": {
                    "root": "runs/explicit-limit",
                    "budget": 16,
                },
                "agent": {"kind": "codex", "limit": 4},
            }
        )

        self.assertEqual(spec.agent.limit, 4)

    def test_builds_root_from_runs_layout(self) -> None:
        spec = Spec.from_mapping(
            {
                "run": {
                    "runs": "runs",
                    "env": "cartpole",
                    "model": "codex",
                    "exp_id": "smoke-001",
                    "budget": 16,
                }
            }
        )

        self.assertEqual(spec.root, Path("runs/codex/cartpole/smoke-001"))
        self.assertEqual(spec.runs, Path("runs"))
        self.assertEqual(spec.exp, "smoke-001")
        self.assertEqual(spec.run_key, "smoke-001")

    def test_loads_codex_agent_spec(self) -> None:
        spec = Spec.from_mapping(
            {
                "run": {"root": "runs/codex", "budget": 2},
                "agent": {
                    "kind": "codex",
                    "binary": "codex",
                    "model": "gpt-test",
                    "sandbox": "workspace-write",
                    "approval": "never",
                    "bypass": True,
                    "args": ["--skip-git-repo-check"],
                },
            }
        )

        self.assertEqual(spec.agent.kind, "codex")
        self.assertEqual(spec.agent.binary, "codex")
        self.assertEqual(spec.agent.model, "gpt-test")
        self.assertTrue(spec.agent.bypass)
        self.assertEqual(spec.agent.args, ("--skip-git-repo-check",))

    def test_loads_claude_agent_spec(self) -> None:
        spec = Spec.from_mapping(
            {
                "run": {"root": "runs/claude", "budget": 2},
                "agent": {
                    "kind": "claude",
                    "binary": "claude",
                    "model": "sonnet",
                    "permission": "bypassPermissions",
                    "tools": ["Bash", "Read"],
                    "args": ["--max-turns", "8"],
                },
            }
        )

        self.assertEqual(spec.agent.kind, "claude")
        self.assertEqual(spec.agent.binary, "claude")
        self.assertEqual(spec.agent.model, "sonnet")
        self.assertEqual(spec.agent.permission, "bypassPermissions")
        self.assertEqual(spec.agent.tools, ("Bash", "Read"))
        self.assertEqual(spec.agent.args, ("--max-turns", "8"))

    def test_loads_kimi_agent_spec(self) -> None:
        spec = Spec.from_mapping(
            {
                "run": {"root": "runs/kimi", "budget": 2},
                "agent": {
                    "kind": "kimi",
                    "binary": "kimi",
                    "model": "kimi-k2",
                    "args": ["--debug"],
                },
            }
        )

        self.assertEqual(spec.agent.kind, "kimi")
        self.assertEqual(spec.agent.binary, "kimi")
        self.assertEqual(spec.agent.model, "kimi-k2")
        self.assertEqual(spec.agent.args, ("--debug",))

    def test_rejects_invalid_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "run.budget is required"):
            Spec.from_mapping({"run": {"root": "runs/missing"}})
        with self.assertRaisesRegex(ValueError, "unsupported agent kind"):
            Spec.from_mapping(
                {
                    "run": {"root": "runs/bad", "budget": 1},
                    "agent": {"kind": "unknown"},
                }
            )
        with self.assertRaisesRegex(ValueError, "agent.bypass must be a boolean"):
            Spec.from_mapping(
                {
                    "run": {"root": "runs/bad", "budget": 1},
                    "agent": {"kind": "codex", "bypass": "yes"},
                }
            )
        with self.assertRaisesRegex(ValueError, "agent.retries must be non-negative"):
            Spec.from_mapping(
                {
                    "run": {"root": "runs/bad", "budget": 1},
                    "agent": {"retries": -1},
                }
            )
        with self.assertRaisesRegex(ValueError, "agent.limit must be positive"):
            Spec.from_mapping(
                {
                    "run": {"root": "runs/bad", "budget": 1},
                    "agent": {"limit": 0},
                }
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Spec.from_mapping(
                {
                    "run": {
                        "root": "runs/manual",
                        "runs": "runs",
                        "budget": 1,
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
