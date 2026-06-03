from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from evopolicygym import Command, Drive, Loop, local
from evopolicygym.check import check
from evopolicygym.envs import toy


class DriveTest(unittest.TestCase):
    def test_drive_runs_command_agent_against_local_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            script = Path(tmp) / "agent.py"
            script.write_text(_agent_script(), encoding="utf-8")
            host = local(
                root,
                toy(),
                key="run-001",
                model="agent",
                exp="drive",
                budget=2,
                maximum=1,
                valid_size=1,
                final_size=1,
            )
            loop = Loop(Command((sys.executable, str(script))), limit=4)

            try:
                trial = Drive(loop).run(host)
            except PermissionError as exc:
                self.skipTest(f"TCP bind is not permitted in this sandbox: {exc}")

            self.assertTrue(trial.done)
            self.assertFalse(host.run.alive())
            self.assertEqual(trial.transcript.reason, "done")
            self.assertEqual(len(trial.transcript.replies), 2)
            self.assertEqual(
                [reply.data["remaining"] for reply in trial.transcript.replies],
                [1, 0],
            )
            self.assertTrue(trial.launch.endpoint.startswith("http://127.0.0.1:"))
            events = [row["event"] for row in _jsonl(root / "logs" / "harness.log")]
            self.assertIn("drive.start", events)
            self.assertIn("server.start", events)
            self.assertIn("loop.start", events)
            self.assertIn("loop.finish", events)
            self.assertIn("server.stop", events)
            self.assertIn("drive.finish", events)
            self.assertTrue(check(root).ok)


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


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
