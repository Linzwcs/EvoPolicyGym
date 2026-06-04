from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evopolicygym.check import check


class CliTest(unittest.TestCase):
    def test_cli_run_drives_command_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            script = Path(tmp) / "agent.py"
            script.write_text(_agent_script(), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--env",
                    "toy",
                    "--root",
                    str(root),
                    "--budget",
                    "2",
                    "--maximum",
                    "1",
                    "--valid-size",
                    "1",
                    "--final-size",
                    "1",
                    "--limit",
                    "4",
                    "--",
                    sys.executable,
                    str(script),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)

            self.assertTrue(summary["done"])
            self.assertEqual(summary["reason"], "done")
            self.assertEqual(summary["submits"], 2)
            self.assertTrue((root / "run.json").exists())
            self.assertTrue(
                (root / "workspace" / "feedback" / "submit_000" / "summary.json").exists()
            )
            self.assertTrue((root / "logs" / "agent.jsonl").exists())

    def test_cli_check_envs_reads_discovery_source_for_bulk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "discovered.json"
            source.write_text(
                json.dumps(
                    {
                        "families": [
                            {
                                "name": "Official Gymnasium: Classic Control",
                                "source": "gymnasium.registry",
                                "ids": ["CartPole-v1"],
                                "count": 1,
                            }
                        ],
                        "packages": [],
                        "blocked": [],
                        "total": 1,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "check-envs",
                    "--bulk",
                    "--isolate",
                    "--source",
                    str(source),
                    "--family",
                    "Official Gymnasium: Classic Control",
                    "--min-level",
                    "L1",
                    "--timeout",
                    "10",
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=20.0,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            body = json.loads(result.stdout)
            names = {row["name"] for row in body["envs"]}
            self.assertIn("gymnasium/CartPole-v1", names)
            self.assertEqual(body["failed"], 0)

    def test_cli_check_envs_can_isolate_bulk_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "discovered.json"
            source.write_text(
                json.dumps(
                    {
                        "families": [
                            {
                                "name": "Official Gymnasium: Classic Control",
                                "source": "gymnasium.registry",
                                "ids": ["CartPole-v1", "MountainCar-v0"],
                                "count": 2,
                            }
                        ],
                        "packages": [],
                        "blocked": [],
                        "total": 2,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "check-envs",
                    "--bulk",
                    "--isolate",
                    "--source",
                    str(source),
                    "--family",
                    "Official Gymnasium: Classic Control",
                    "--min-level",
                    "L1",
                    "--timeout",
                    "10",
                    "--jobs",
                    "2",
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=20.0,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            body = json.loads(result.stdout)
            self.assertEqual(body["failed"], 0)
            self.assertGreaterEqual(body["checked"], 2)
            rows = {row["name"]: row for row in body["envs"]}
            self.assertIn("gymnasium/CartPole-v1", rows)
            self.assertIn("gymnasium/MountainCar-v0", rows)
            self.assertTrue(rows["gymnasium/CartPole-v1"]["registered"])
            self.assertTrue(rows["gymnasium/MountainCar-v0"]["registered"])

    def test_cli_run_builds_canonical_root_from_runs_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            root = runs / "script" / "toy" / "smoke-001"
            script = Path(tmp) / "agent.py"
            script.write_text(_agent_script(), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--env",
                    "toy",
                    "--runs",
                    str(runs),
                    "--model",
                    "script",
                    "--exp-id",
                    "smoke-001",
                    "--budget",
                    "1",
                    "--maximum",
                    "1",
                    "--valid-size",
                    "1",
                    "--final-size",
                    "1",
                    "--limit",
                    "2",
                    "--",
                    sys.executable,
                    str(script),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            run_json = json.loads((root / "run.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["root"], str(root))
            self.assertEqual(run_json["model"], "script")
            self.assertEqual(run_json["exp_id"], "smoke-001")

    def test_cli_run_accepts_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "configured"
            script = Path(tmp) / "agent.py"
            config = Path(tmp) / "run.json"
            script.write_text(_agent_script(), encoding="utf-8")
            config.write_text(
                json.dumps(
                    {
                        "run": {
                            "env": "toy",
                            "root": str(root),
                            "budget": 2,
                            "maximum": 1,
                            "valid_size": 1,
                            "final_size": 1,
                        },
                        "agent": {
                            "kind": "command",
                            "argv": [sys.executable, str(script)],
                            "limit": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--config",
                    str(config),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)

            self.assertTrue(summary["done"])
            self.assertEqual(summary["root"], str(root))
            self.assertEqual(summary["submits"], 2)
            self.assertTrue((root / "run.json").exists())

    def test_cli_data_make_writes_configurable_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "data",
                    "make",
                    "--env",
                    "toy",
                    "--root",
                    str(root),
                    "--seed",
                    "3",
                    "--train-size",
                    "4",
                    "--valid-size",
                    "2",
                    "--heldout-size",
                    "1",
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            train = json.loads((root / "train.json").read_text(encoding="utf-8"))
            valid = json.loads((root / "valid.json").read_text(encoding="utf-8"))
            heldout = json.loads((root / "heldout.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["env"], "toy")
            self.assertEqual(summary["train"], 4)
            self.assertEqual(summary["valid"], 2)
            self.assertEqual(summary["heldout"], 1)
            self.assertEqual(len(train["cases"]), 4)
            self.assertEqual(len(valid["cases"]), 2)
            self.assertEqual(len(heldout["cases"]), 1)
            self.assertEqual(train["generator"], {"kind": "seed", "seed": 3, "size": 4})

    def test_cli_check_envs_reports_manifest_status(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evopolicygym.cli",
                "check-envs",
                "--env",
                "toy",
            ],
            capture_output=True,
            env=_env(),
            text=True,
            timeout=10.0,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["registered"], 1)
        self.assertEqual(summary["checked"], 1)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["envs"][0]["name"], "toy")
        self.assertTrue(summary["envs"][0]["registered"])
        self.assertTrue(summary["envs"][0]["ok"])

    def test_cli_check_envs_reports_manifest_status_in_isolation(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evopolicygym.cli",
                "check-envs",
                "--env",
                "toy",
                "--isolate",
            ],
            capture_output=True,
            env=_env(),
            text=True,
            timeout=10.0,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["registered"], 1)
        self.assertEqual(summary["checked"], 1)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["envs"][0]["name"], "toy")

    def test_cli_run_accepts_codex_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "codex-run"
            fake = Path(tmp) / "codex"
            config = Path(tmp) / "codex.json"
            fake.write_text(_fake_codex(), encoding="utf-8")
            fake.chmod(0o755)
            config.write_text(
                json.dumps(
                    {
                        "run": {
                            "env": "toy",
                            "root": str(root),
                            "budget": 2,
                            "maximum": 1,
                            "valid_size": 1,
                            "final_size": 1,
                        },
                        "agent": {
                            "kind": "codex",
                            "binary": str(fake),
                            "bypass": True,
                            "limit": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--config",
                    str(config),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)

            self.assertTrue(summary["done"])
            self.assertEqual(summary["submits"], 2)
            self.assertIn("codex:", summary["session"])
            command = json.loads(
                (root / "logs" / "codex_turns" / "turn_000.command.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
            self.assertNotIn("--sandbox", command)
            self.assertTrue((root / "logs" / "codex_turns" / "turn_001.command.json").exists())
            self.assertTrue((root / "run.json").exists())

    def test_cli_run_cartpole_with_codex_budget_16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cartpole-codex"
            fake = Path(tmp) / "codex"
            config = Path(tmp) / "cartpole.json"
            fake.write_text(_fake_cartpole_codex(), encoding="utf-8")
            fake.chmod(0o755)
            config.write_text(
                json.dumps(
                    {
                        "run": {
                            "env": "cartpole",
                            "root": str(root),
                            "budget": 16,
                            "maximum": 4,
                            "valid_size": 2,
                            "final_size": 2,
                        },
                        "agent": {
                            "kind": "codex",
                            "binary": str(fake),
                            "limit": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--config",
                    str(config),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=20.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            run_json = json.loads((root / "run.json").read_text(encoding="utf-8"))

            self.assertTrue(summary["done"])
            self.assertEqual(summary["reason"], "done")
            self.assertEqual(summary["submits"], 4)
            self.assertIn("codex:", summary["session"])
            self.assertEqual(run_json["env"], "cartpole")
            self.assertEqual(run_json["outcome"]["status"], "completed")
            self.assertGreater(run_json["outcome"]["final_score"], 0.0)
            self.assertTrue(
                (root / "workspace" / "feedback" / "submit_003" / "summary.json").exists()
            )
            self.assertTrue((root / "logs" / "codex_turns" / "turn_003.command.json").exists())
            self.assertTrue(check(root).ok)

    def test_cli_run_accepts_claude_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "claude-run"
            fake = Path(tmp) / "claude"
            config = Path(tmp) / "claude.json"
            fake.write_text(_fake_claude(), encoding="utf-8")
            fake.chmod(0o755)
            config.write_text(
                json.dumps(
                    {
                        "run": {
                            "env": "toy",
                            "root": str(root),
                            "budget": 2,
                            "maximum": 1,
                            "valid_size": 1,
                            "final_size": 1,
                        },
                        "agent": {
                            "kind": "claude",
                            "binary": str(fake),
                            "limit": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--config",
                    str(config),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)

            self.assertTrue(summary["done"])
            self.assertEqual(summary["submits"], 2)
            self.assertIn("claude:", summary["session"])
            self.assertTrue((root / "logs" / "claude_turns" / "turn_001.command.json").exists())
            self.assertTrue((root / "run.json").exists())

    def test_cli_run_accepts_kimi_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kimi-run"
            fake = Path(tmp) / "kimi"
            config = Path(tmp) / "kimi.json"
            fake.write_text(_fake_kimi(), encoding="utf-8")
            fake.chmod(0o755)
            config.write_text(
                json.dumps(
                    {
                        "run": {
                            "env": "toy",
                            "root": str(root),
                            "budget": 2,
                            "maximum": 1,
                            "valid_size": 1,
                            "final_size": 1,
                        },
                        "agent": {
                            "kind": "kimi",
                            "binary": str(fake),
                            "limit": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "run",
                    "--config",
                    str(config),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)

            self.assertTrue(summary["done"])
            self.assertEqual(summary["submits"], 2)
            self.assertIn("kimi:", summary["session"])
            self.assertTrue((root / "logs" / "kimi_turns" / "turn_001.command.json").exists())
            command = (root / "logs" / "kimi_turns" / "turn_000.command.json").read_text(
                encoding="utf-8"
            )
            self.assertIn("kimi-k2", command)
            self.assertTrue((root / "run.json").exists())

    def test_cli_suite_runs_serial_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            script = Path(tmp) / "agent.py"
            config = Path(tmp) / "suite.json"
            script.write_text(_agent_script(), encoding="utf-8")
            config.write_text(
                json.dumps(
                    {
                        "suite": {"root": str(root), "repeats": 2, "jobs": 2},
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
                            {
                                "kind": "command",
                                "argv": [sys.executable, str(script)],
                                "limit": 4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evopolicygym.cli",
                    "suite",
                    "--config",
                    str(config),
                ],
                capture_output=True,
                env=_env(),
                text=True,
                timeout=15.0,
            )
            if result.returncode != 0 and _bind_denied(result.stderr):
                self.skipTest(f"TCP bind is not permitted in this sandbox: {result.stderr}")

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            report = json.loads((root / "suite.json").read_text(encoding="utf-8"))

            self.assertTrue(summary["done"])
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["passed"], 2)
            self.assertEqual(summary["jobs_configured"], 2)
            self.assertEqual(summary["by_category"], {"completed": 2})
            self.assertEqual(len(summary["jobs"]), 2)
            self.assertEqual(report["total"], 2)
            for job in report["jobs"]:
                self.assertTrue(job["done"])
                self.assertEqual(job["submits"], 2)
                self.assertEqual(job["category"], "completed")
                self.assertTrue(job["check"]["ok"])
                self.assertTrue((Path(job["root"]) / "run.json").exists())


def _env() -> dict[str, str]:
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not current else os.pathsep.join((src, current))
    return env


def _bind_denied(stderr: str) -> bool:
    return "PermissionError" in stderr or "Operation not permitted" in stderr


def _agent_script() -> str:
    policy = repr(_policy())
    return (
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import urllib.request\n"
        "policy = " + policy + "\n"
        "count = 0\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    count += 1\n"
        "    pathlib.Path(os.environ['EVOPOLICYGYM_SYSTEM'], 'policy.py').write_text(policy)\n"
        "    body = json.dumps({'env_instances': [count - 1]}).encode('utf-8')\n"
        "    post = urllib.request.Request(\n"
        "        os.environ['EVOPOLICYGYM_SUBMIT_URL'],\n"
        "        data=body,\n"
        "        headers={'Content-Type': 'application/json'},\n"
        "        method='POST',\n"
        "    )\n"
        "    with urllib.request.urlopen(post, timeout=5.0) as response:\n"
        "        result = json.loads(response.read().decode('utf-8'))\n"
        "    print(json.dumps({\n"
        "        'turn': req['turn'],\n"
        "        'text': result['status'],\n"
        "        'data': {'remaining': result['summary']['remaining_budget']},\n"
        "    }), flush=True)\n"
    )


def _fake_codex() -> str:
    policy = repr(_policy())
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import urllib.request\n"
        "args = sys.argv[1:]\n"
        "resume = 'resume' in args\n"
        "prompt = args[-1]\n"
        "session = args[-2] if resume else 'codex-thread-001'\n"
        "with urllib.request.urlopen(os.environ['EVOPOLICYGYM_INFO_URL'], timeout=5.0) as response:\n"
        "    info = json.loads(response.read().decode('utf-8'))\n"
        "case = info['state']['n_submits']\n"
        "pathlib.Path(os.environ['EVOPOLICYGYM_SYSTEM'], 'policy.py').write_text(" + policy + ")\n"
        "body = json.dumps({'env_instances': [case]}).encode('utf-8')\n"
        "post = urllib.request.Request(\n"
        "    os.environ['EVOPOLICYGYM_SUBMIT_URL'],\n"
        "    data=body,\n"
        "    headers={'Content-Type': 'application/json'},\n"
        "    method='POST',\n"
        ")\n"
        "with urllib.request.urlopen(post, timeout=5.0) as response:\n"
        "    result = json.loads(response.read().decode('utf-8'))\n"
        "print(json.dumps({'type': 'thread.started', 'payload': {'id': session}}))\n"
        "print(json.dumps({'type': 'agent_message', 'payload': {\n"
        "    'message': ('resume:' if resume else 'start:') + result['status'] + ':' + prompt[:20]\n"
        "}}))\n"
    )


def _fake_cartpole_codex() -> str:
    policy = repr(_cartpole_policy())
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import urllib.request\n"
        "args = sys.argv[1:]\n"
        "resume = 'resume' in args\n"
        "prompt = args[-1]\n"
        "session = args[-2] if resume else 'codex-cartpole-001'\n"
        "with urllib.request.urlopen(os.environ['EVOPOLICYGYM_INFO_URL'], timeout=5.0) as response:\n"
        "    info = json.loads(response.read().decode('utf-8'))\n"
        "state = info['state']\n"
        "remaining = state['remaining_budget']\n"
        "limit = min(info['max_episodes_per_submit'], remaining)\n"
        "start = state['n_submits'] * info['max_episodes_per_submit']\n"
        "cases = list(range(start, start + limit))\n"
        "pathlib.Path(os.environ['EVOPOLICYGYM_SYSTEM'], 'policy.py').write_text(" + policy + ")\n"
        "body = json.dumps({'env_instances': cases}).encode('utf-8')\n"
        "post = urllib.request.Request(\n"
        "    os.environ['EVOPOLICYGYM_SUBMIT_URL'],\n"
        "    data=body,\n"
        "    headers={'Content-Type': 'application/json'},\n"
        "    method='POST',\n"
        ")\n"
        "with urllib.request.urlopen(post, timeout=5.0) as response:\n"
        "    result = json.loads(response.read().decode('utf-8'))\n"
        "print(json.dumps({'type': 'thread.started', 'payload': {'id': session}}))\n"
        "print(json.dumps({'type': 'agent_message', 'payload': {\n"
        "    'message': (('resume:' if resume else 'start:') + result['status'] +\n"
        "        ':cases=' + ','.join(map(str, cases)) + ':' + prompt[:20])\n"
        "}}))\n"
    )


def _fake_claude() -> str:
    policy = repr(_policy())
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import urllib.request\n"
        "args = sys.argv[1:]\n"
        "resume = '--resume' in args\n"
        "prompt = sys.stdin.read()\n"
        "with urllib.request.urlopen(os.environ['EVOPOLICYGYM_INFO_URL'], timeout=5.0) as response:\n"
        "    info = json.loads(response.read().decode('utf-8'))\n"
        "case = info['state']['n_submits']\n"
        "pathlib.Path(os.environ['EVOPOLICYGYM_SYSTEM'], 'policy.py').write_text(" + policy + ")\n"
        "body = json.dumps({'env_instances': [case]}).encode('utf-8')\n"
        "post = urllib.request.Request(\n"
        "    os.environ['EVOPOLICYGYM_SUBMIT_URL'],\n"
        "    data=body,\n"
        "    headers={'Content-Type': 'application/json'},\n"
        "    method='POST',\n"
        ")\n"
        "with urllib.request.urlopen(post, timeout=5.0) as response:\n"
        "    result = json.loads(response.read().decode('utf-8'))\n"
        "print(json.dumps({\n"
        "    'type': 'result',\n"
        "    'session_id': 'claude-session-001',\n"
        "    'result': ('resume:' if resume else 'start:') + result['status'] + ':' + prompt[:20],\n"
        "    'total_cost_usd': 0.01,\n"
        "    'num_turns': 1,\n"
        "}))\n"
    )


def _fake_kimi() -> str:
    policy = repr(_policy())
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import urllib.request\n"
        "args = sys.argv[1:]\n"
        "resume = '-S' in args\n"
        "prompt = args[-1]\n"
        "session = args[args.index('-S') + 1] if resume else 'session_kimi_001'\n"
        "with urllib.request.urlopen(os.environ['EVOPOLICYGYM_INFO_URL'], timeout=5.0) as response:\n"
        "    info = json.loads(response.read().decode('utf-8'))\n"
        "case = info['state']['n_submits']\n"
        "pathlib.Path(os.environ['EVOPOLICYGYM_SYSTEM'], 'policy.py').write_text(" + policy + ")\n"
        "body = json.dumps({'env_instances': [case]}).encode('utf-8')\n"
        "post = urllib.request.Request(\n"
        "    os.environ['EVOPOLICYGYM_SUBMIT_URL'],\n"
        "    data=body,\n"
        "    headers={'Content-Type': 'application/json'},\n"
        "    method='POST',\n"
        ")\n"
        "with urllib.request.urlopen(post, timeout=5.0) as response:\n"
        "    result = json.loads(response.read().decode('utf-8'))\n"
        "print(json.dumps({\n"
        "    'type': 'assistant',\n"
        "    'sessionId': session,\n"
        "    'message': ('resume:' if resume else 'start:') + result['status'] + ':' + prompt[:20],\n"
        "}))\n"
    )


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


def _cartpole_policy() -> str:
    return (
        "class Policy:\n"
        "    def __init__(self, obs_space, action_space, env_meta):\n"
        "        self.env_meta = env_meta\n"
        "    def reset(self, episode_index):\n"
        "        self.episode_index = episode_index\n"
        "    def act(self, obs):\n"
        "        x, x_dot, theta, theta_dot = obs\n"
        "        return 1 if theta + 0.08 * theta_dot + 0.01 * x_dot > 0 else 0\n"
    )


if __name__ == "__main__":
    unittest.main()
