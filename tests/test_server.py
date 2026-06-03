from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from evopolicygym import PoolKind, local
from evopolicygym.check import check
from evopolicygym.envs import toy
from evopolicygym.infra.http import serve


class ServerTest(unittest.TestCase):
    def test_stdlib_server_serves_agent_api(self) -> None:
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
            )
            (host.store.workspace / "policy.py").write_text(_policy(), encoding="utf-8")

            with self._serve(host.service) as server:
                info = _get_json(f"{server.url}/info")
                task = _get_text(f"{server.url}/task")
                result = _post_json(f"{server.url}/submit", {"env_instances": "0-1"})

                self.assertEqual(info["state"]["remaining_budget"], 2)
                self.assertIn("# toy", task.lower())
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["summary"]["remaining_budget"], 0)

            self.assertFalse(host.run.alive())
            self.assertTrue(check(root).ok)

    def test_finalize_endpoint_is_not_agent_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            host = local(
                Path(tmp) / "run",
                toy(),
                key="run-001",
                model="agent",
                exp="smoke",
                budget=1,
            )

            with self._serve(host.service) as server:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    _post_json(f"{server.url}/finalize", {})

            self.assertEqual(ctx.exception.code, 405)

    def _serve(self, service):
        try:
            return serve(service)
        except PermissionError as exc:
            self.skipTest(f"TCP bind is not permitted in this sandbox: {exc}")


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5.0) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


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


if __name__ == "__main__":
    unittest.main()
