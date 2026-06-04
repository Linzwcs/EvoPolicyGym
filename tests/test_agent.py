from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from evopolicygym import Claude, Codex, Command, Kimi, Launch, Loop, Reply, local
from evopolicygym.check import check
from evopolicygym.envs import toy
from evopolicygym.infra.http import SubmitRequest
from evopolicygym.protocol.agents import body as agents_body


class AgentLaunchTest(unittest.TestCase):
    def test_launch_exposes_agent_startup_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            launch = Launch(root=root, endpoint="http://127.0.0.1:9999", env={"X": "Y"})
            resolved = root.resolve()

            env = launch.environ()
            prompt = launch.prompt()

            self.assertEqual(launch.workspace, resolved / "workspace")
            self.assertEqual(launch.system, resolved / "workspace" / "system")
            self.assertEqual(launch.agents, resolved / "workspace" / "AGENTS.md")
            self.assertEqual(launch.feedback, resolved / "workspace" / "feedback")
            self.assertEqual(env["EVOPOLICYGYM_API"], "http://127.0.0.1:9999")
            self.assertEqual(env["EVOPOLICYGYM_SUBMIT_URL"], "http://127.0.0.1:9999/submit")
            self.assertEqual(env["EVOPOLICYGYM_AGENTS"], str(resolved / "workspace" / "AGENTS.md"))
            self.assertEqual(env["EVOPOLICYGYM_FEEDBACK"], str(resolved / "workspace" / "feedback"))
            self.assertNotIn("EVOPOLICYGYM_ROOT", env)
            self.assertNotIn("EVOPOLICYGYM_LOGS", env)
            self.assertEqual(env["X"], "Y")
            self.assertIn("First read `AGENTS.md`", prompt)
            self.assertIn("`system/`", prompt)
            self.assertIn("`feedback/`", prompt)
            self.assertIn("code structure", prompt)
            self.assertIn("Do not run local environment rollouts", prompt)
            self.assertIn("candidate-policy scores", prompt)
            self.assertNotIn(str(resolved), prompt)
            self.assertIn("Do not call /finalize", prompt)

    def test_packaged_agents_md_documents_submit_contract(self) -> None:
        text = agents_body()

        self.assertIn("## Submit Format", text)
        self.assertIn('"env_instances": [0, 1, 2, 3]', text)
        self.assertIn("EVOPOLICYGYM_SUBMIT_URL", text)
        self.assertIn("## Code Quality", text)
        self.assertIn("## Policy Interface", text)
        self.assertIn("## Feedback Artifacts", text)
        self.assertIn("`Image`", text)
        self.assertIn('"type": "External"', text)
        self.assertIn("feedback/submit_NNN/summary.json", text)
        self.assertNotIn("$EVOPOLICYGYM_FEEDBACK", text)
        self.assertNotIn("workspace root, which contains", text)

    def test_loop_reuses_one_session_for_full_benchmark_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            host = local(
                root,
                toy(),
                key="run-001",
                model="agent",
                exp="smoke",
                budget=2,
                maximum=1,
                valid_size=1,
                final_size=1,
            )
            harness = ScriptHarness(lambda turn: _submit_once(host, turn))
            launch = Launch.from_host(host, "http://127.0.0.1:9999")

            transcript = Loop(harness, limit=4).run(
                launch,
                done=lambda: not host.run.alive(),
            )

            self.assertEqual(harness.starts, 1)
            self.assertEqual(len(harness.sessions), 1)
            self.assertEqual(harness.sessions[0].steps, 2)
            self.assertIn("EvoPolicyGym benchmark session", harness.sessions[0].messages[0])
            self.assertEqual(
                harness.sessions[0].messages[1],
                "Continue optimizing policy behavior and code structure in the same EvoPolicyGym context.",
            )
            self.assertTrue(harness.sessions[0].closed)
            self.assertTrue(transcript.done)
            self.assertEqual(transcript.session, "session-001")
            self.assertEqual([reply.turn for reply in transcript.replies], [0, 1])
            self.assertFalse(host.run.alive())
            self.assertTrue(check(root).ok)

    def test_loop_stops_when_session_requests_stop(self) -> None:
        harness = ScriptHarness(lambda turn: Reply(turn=turn, text="stop", stop=True))
        launch = Launch(root=Path("run"), endpoint="http://server")

        transcript = Loop(harness, limit=4).run(launch, done=lambda: False)

        self.assertEqual(transcript.reason, "session_stop")
        self.assertEqual(len(transcript.replies), 1)
        self.assertTrue(harness.sessions[0].closed)

    def test_loop_retries_step_exceptions(self) -> None:
        calls = 0

        def action(turn: int) -> Reply:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("service timeout")
            return Reply(turn=turn, text="recovered", stop=True)

        with tempfile.TemporaryDirectory() as tmp:
            launch = Launch(root=Path(tmp) / "run", endpoint="http://server")
            harness = ScriptHarness(action)

            transcript = Loop(harness, limit=4, retries=1, backoff=0).run(
                launch,
                done=lambda: False,
            )

            self.assertEqual(transcript.reason, "session_stop")
            self.assertEqual(transcript.replies[0].text, "recovered")
            self.assertIn("GET /info", harness.sessions[0].messages[1])
            events = _events(launch.logs / "harness.log")
            self.assertIn("agent.retry", events)

    def test_loop_retries_retryable_replies(self) -> None:
        calls = 0

        def action(turn: int) -> Reply:
            nonlocal calls
            calls += 1
            if calls == 1:
                return Reply(turn=turn, stop=True, data={"timed_out": True})
            return Reply(turn=turn, text="ok", stop=True)

        with tempfile.TemporaryDirectory() as tmp:
            launch = Launch(root=Path(tmp) / "run", endpoint="http://server")
            harness = ScriptHarness(action)

            transcript = Loop(harness, limit=4, retries=1, backoff=0).run(
                launch,
                done=lambda: False,
            )

            self.assertEqual(transcript.reason, "session_stop")
            self.assertEqual(len(transcript.replies), 1)
            self.assertEqual(transcript.replies[0].text, "ok")
            self.assertEqual(calls, 2)

    def test_loop_reports_retry_exhaustion(self) -> None:
        def action(turn: int) -> Reply:
            raise RuntimeError("network down")

        with tempfile.TemporaryDirectory() as tmp:
            launch = Launch(root=Path(tmp) / "run", endpoint="http://server")
            harness = ScriptHarness(action)

            transcript = Loop(harness, limit=4, retries=1, backoff=0).run(
                launch,
                done=lambda: False,
            )

            self.assertEqual(transcript.reason, "retry_exhausted")
            self.assertTrue(transcript.replies[0].data["retry_exhausted"])
            events = _events(launch.logs / "harness.log")
            self.assertIn("agent.retry.exhausted", events)

    def test_command_harness_preserves_process_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            script = Path(tmp) / "agent.py"
            script.write_text(_agent_script(), encoding="utf-8")
            launch = Launch(
                root=root,
                endpoint="http://127.0.0.1:9999",
                env={"CASE": "launch"},
            )

            session = Command(
                (sys.executable, str(script)),
                env={"CASE": "adapter"},
            ).start(launch)
            try:
                first = session.step("first")
                second = session.step("second")
            finally:
                session.close()

            self.assertFalse(first.stop)
            self.assertTrue(second.stop)
            self.assertEqual(first.data["count"], 1)
            self.assertEqual(second.data["count"], 2)
            self.assertEqual(first.data["pid"], second.data["pid"])
            self.assertEqual(first.data["api"], "http://127.0.0.1:9999")
            self.assertEqual(first.data["case"], "launch")
            self.assertEqual(Path(first.data["cwd"]).resolve(), launch.workspace.resolve())
            self.assertTrue((launch.logs / "agent.jsonl").exists())
            self.assertTrue((launch.logs / "agent.stderr.txt").exists())

    def test_codex_harness_resumes_scraped_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fake = Path(tmp) / "codex"
            fake.write_text(_fake_codex(), encoding="utf-8")
            fake.chmod(0o755)
            launch = Launch(root=root, endpoint="http://127.0.0.1:9999")

            session = Codex(binary=str(fake), model="gpt-test", timeout=5.0).start(launch)
            try:
                first = session.step("first")
                second = session.step("second")
            finally:
                session.close()

            self.assertFalse(first.stop)
            self.assertFalse(second.stop)
            self.assertEqual(first.data["codex_session"], "codex-thread-001")
            self.assertEqual(second.data["codex_session"], "codex-thread-001")
            self.assertIn("start:first", first.text)
            self.assertIn("resume:second", second.text)
            self.assertIn("codex:codex-thread-001", session.key)
            self.assertTrue((launch.logs / "codex_turns" / "turn_000.stream.jsonl").exists())
            self.assertTrue((launch.logs / "codex_turns" / "turn_001.command.json").exists())

            command = (launch.logs / "codex_turns" / "turn_001.command.json").read_text(
                encoding="utf-8"
            )
            argv = json.loads(command)
            self.assertIn("resume", command)
            self.assertIn("codex-thread-001", command)
            self.assertIn("approval_policy", command)
            self.assertNotIn("ask-for-approval", command)
            self.assertLess(argv.index("--cd"), argv.index("resume"))

    def test_codex_bypass_uses_unsandboxed_cli_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fake = Path(tmp) / "codex"
            fake.write_text(_fake_codex(), encoding="utf-8")
            fake.chmod(0o755)
            launch = Launch(root=root, endpoint="http://127.0.0.1:9999")

            session = Codex(
                binary=str(fake),
                sandbox="workspace-write",
                approval="never",
                bypass=True,
                timeout=5.0,
            ).start(launch)
            try:
                reply = session.step("first")
            finally:
                session.close()

            self.assertFalse(reply.stop)
            command = json.loads(
                (launch.logs / "codex_turns" / "turn_000.command.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
            self.assertNotIn("--sandbox", command)
            self.assertNotIn("approval_policy", " ".join(command))

    def test_claude_harness_resumes_stream_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fake = Path(tmp) / "claude"
            fake.write_text(_fake_claude(), encoding="utf-8")
            fake.chmod(0o755)
            launch = Launch(root=root, endpoint="http://127.0.0.1:9999")

            session = Claude(binary=str(fake), model="sonnet", timeout=5.0).start(launch)
            try:
                first = session.step("first")
                second = session.step("second")
            finally:
                session.close()

            self.assertFalse(first.stop)
            self.assertFalse(second.stop)
            self.assertEqual(first.data["claude_session"], "claude-session-001")
            self.assertEqual(second.data["claude_session"], "claude-session-001")
            self.assertEqual(first.data["cost_usd"], 0.01)
            self.assertEqual(second.data["inner_turns"], 1)
            self.assertIn("start:first", first.text)
            self.assertIn("resume:second", second.text)
            self.assertEqual(session.key, "claude:claude-session-001")
            self.assertTrue((launch.logs / "claude_turns" / "turn_000.stream.jsonl").exists())
            self.assertTrue((launch.logs / "claude_turns" / "turn_001.command.json").exists())

            command = (launch.logs / "claude_turns" / "turn_001.command.json").read_text(
                encoding="utf-8"
            )
            self.assertIn("--resume", command)
            self.assertIn("claude-session-001", command)

    def test_kimi_harness_resumes_stream_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fake = Path(tmp) / "kimi"
            fake.write_text(_fake_kimi(), encoding="utf-8")
            fake.chmod(0o755)
            launch = Launch(root=root, endpoint="http://127.0.0.1:9999")

            session = Kimi(binary=str(fake), timeout=5.0).start(launch)
            try:
                first = session.step("first")
                second = session.step("second")
            finally:
                session.close()

            self.assertFalse(first.stop)
            self.assertFalse(second.stop)
            self.assertEqual(first.data["kimi_session"], "session_kimi_001")
            self.assertEqual(second.data["kimi_session"], "session_kimi_001")
            self.assertIn("start:first", first.text)
            self.assertIn("resume:second", second.text)
            self.assertEqual(session.key, "kimi:session_kimi_001")
            self.assertTrue((launch.logs / "kimi_turns" / "turn_000.stream.jsonl").exists())
            self.assertTrue((launch.logs / "kimi_turns" / "turn_001.command.json").exists())

            command = (launch.logs / "kimi_turns" / "turn_001.command.json").read_text(
                encoding="utf-8"
            )
            self.assertIn("-S", command)
            self.assertIn("session_kimi_001", command)
            self.assertNotIn('"-m"', command)

    def test_kimi_harness_passes_explicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fake = Path(tmp) / "kimi"
            fake.write_text(_fake_kimi(), encoding="utf-8")
            fake.chmod(0o755)
            launch = Launch(root=root, endpoint="http://127.0.0.1:9999")

            session = Kimi(
                binary=str(fake), model="kimi-code/kimi-for-coding", timeout=5.0
            ).start(launch)
            try:
                reply = session.step("first")
            finally:
                session.close()

            self.assertFalse(reply.stop)
            command = (launch.logs / "kimi_turns" / "turn_000.command.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"-m"', command)
            self.assertIn("kimi-code/kimi-for-coding", command)


@dataclass(slots=True)
class ScriptHarness:
    action: Callable[[int], Reply]
    starts: int = 0
    sessions: list[ScriptSession] = field(default_factory=list)

    def start(self, launch: Launch) -> ScriptSession:
        self.starts += 1
        session = ScriptSession(action=self.action)
        self.sessions.append(session)
        return session


@dataclass(slots=True)
class ScriptSession:
    action: Callable[[int], Reply]
    key: str = "session-001"
    steps: int = 0
    messages: list[str] = field(default_factory=list)
    closed: bool = False

    def step(self, message: str) -> Reply:
        self.messages.append(message)
        turn = self.steps
        self.steps += 1
        return self.action(turn)

    def close(self) -> None:
        self.closed = True


def _submit_once(host, turn: int) -> Reply:
    (host.store.workspace / "policy.py").write_text(_policy(), encoding="utf-8")
    response = host.service.submit(SubmitRequest([turn]))
    return Reply(turn=turn, text=response.status, data={"remaining": response.summary["remaining_budget"]})


def _events(path: Path) -> set[str]:
    return {json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()}


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


def _agent_script() -> str:
    return (
        "import json\n"
        "import os\n"
        "import sys\n"
        "count = 0\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    count += 1\n"
        "    print(json.dumps({\n"
        "        'turn': req['turn'],\n"
        "        'text': req['message'],\n"
        "        'stop': count == 2,\n"
        "        'data': {\n"
        "            'api': os.environ['EVOPOLICYGYM_API'],\n"
        "            'case': os.environ['CASE'],\n"
        "            'count': count,\n"
        "            'cwd': os.getcwd(),\n"
        "            'pid': os.getpid(),\n"
        "        },\n"
        "    }), flush=True)\n"
    )


def _fake_codex() -> str:
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "resume = 'resume' in args\n"
        "prompt = args[-1]\n"
        "session = args[-2] if resume else 'codex-thread-001'\n"
        "print(json.dumps({'type': 'thread.started', 'payload': {'id': session}}))\n"
        "print(json.dumps({'type': 'agent_message', 'payload': {\n"
        "    'message': ('resume:' if resume else 'start:') + prompt\n"
        "}}))\n"
    )


def _fake_claude() -> str:
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "resume = '--resume' in args\n"
        "prompt = sys.stdin.read()\n"
        "print(json.dumps({'type': 'assistant', 'message': {'content': 'seen'}}))\n"
        "print(json.dumps({\n"
        "    'type': 'result',\n"
        "    'session_id': 'claude-session-001',\n"
        "    'result': ('resume:' if resume else 'start:') + prompt,\n"
        "    'total_cost_usd': 0.01,\n"
        "    'num_turns': 1,\n"
        "}))\n"
    )


def _fake_kimi() -> str:
    return (
        "#!" + sys.executable + "\n"
        "import json\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "resume = '-S' in args\n"
        "prompt = args[-1]\n"
        "session = args[args.index('-S') + 1] if resume else 'session_kimi_001'\n"
        "print(json.dumps({\n"
        "    'type': 'assistant',\n"
        "    'sessionId': session,\n"
        "    'message': ('resume:' if resume else 'start:') + prompt,\n"
        "}))\n"
    )


if __name__ == "__main__":
    unittest.main()
