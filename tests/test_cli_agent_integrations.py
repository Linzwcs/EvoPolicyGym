from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evopolicygym.agents import (
    AgentTask,
    ClaudeCode,
    CodingAgent,
    KimiCode,
)
from evopolicygym.authoring import BenchmarkSpec
from evopolicygym.run import RunConfig
from evopolicygym.run._task import build_agent_task


def _task() -> AgentTask:
    return build_agent_task(
        BenchmarkSpec(
            id="example/cli-agent-v1",
            description="Command-line Agent integration fixture.",
            observation_space={"shape": [4]},
            action_space={"enum": [0, 1]},
            metadata={},
            max_episode_steps=10,
            primary_metric="reward",
            score_direction="maximize",
        ),
        RunConfig(episode_budget=7),
    )


def _executable(directory: str, name: str) -> Path:
    executable = Path(directory) / name
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    return executable


class ClaudeCodeIntegrationTests(unittest.TestCase):
    def test_selection_becomes_a_bare_non_persistent_invocation(self) -> None:
        task = _task()
        with tempfile.TemporaryDirectory() as temporary:
            executable = _executable(temporary, "claude")
            invocation = ClaudeCode(
                model="sonnet",
                effort="high",
                executable=str(executable),
            ).build_invocation(task)

        self.assertIsInstance(
            ClaudeCode(model="sonnet", effort="medium"),
            CodingAgent,
        )
        self.assertEqual(invocation.instructions, task.instructions)
        self.assertEqual(invocation.identity["provider"], "claude-code")
        self.assertEqual(invocation.identity["model"], "sonnet")
        self.assertEqual(invocation.identity["effort"], "high")
        self.assertEqual(invocation.stdout_media_type, "application/x-ndjson")
        self.assertIn("--print", invocation.command)
        self.assertIn("--verbose", invocation.command)
        self.assertIn("--bare", invocation.command)
        self.assertIn("--strict-mcp-config", invocation.command)
        self.assertIn("--no-session-persistence", invocation.command)
        self.assertEqual(
            invocation.command[
                invocation.command.index("--permission-mode") + 1
            ],
            "bypassPermissions",
        )
        self.assertEqual(
            invocation.recorded_command[-1],
            "@agent/instructions.md",
        )
        self.assertNotIn(task.instructions, invocation.recorded_command)
        self.assertIn("ANTHROPIC_API_KEY", invocation.inherited_environment)
        self.assertIn(
            "CLAUDE_CODE_OAUTH_TOKEN",
            invocation.inherited_environment,
        )
        self.assertIn("CLAUDE_CONFIG_DIR", invocation.inherited_environment)

    def test_rejects_invalid_selections_and_tasks(self) -> None:
        for invalid in ("", "extra high", "high\n", "x" * 129):
            with self.subTest(model=invalid):
                with self.assertRaisesRegex(ValueError, "model"):
                    ClaudeCode(model=invalid, effort="high")
        for invalid in ("", "extra high", "high\n", "x" * 65):
            with self.subTest(effort=invalid):
                with self.assertRaisesRegex(ValueError, "effort"):
                    ClaudeCode(model="sonnet", effort=invalid)
        with self.assertRaisesRegex(TypeError, "task must be AgentTask"):
            ClaudeCode(model="sonnet", effort="high").build_invocation(
                object()  # type: ignore[arg-type]
            )


class KimiCodeIntegrationTests(unittest.TestCase):
    def test_selection_becomes_a_non_interactive_stream_invocation(self) -> None:
        task = _task()
        with tempfile.TemporaryDirectory() as temporary:
            executable = _executable(temporary, "kimi")
            invocation = KimiCode(
                model="kimi-code/kimi-for-coding",
                executable=str(executable),
            ).build_invocation(task)

        self.assertIsInstance(
            KimiCode(model="kimi-code/kimi-for-coding"),
            CodingAgent,
        )
        self.assertEqual(invocation.instructions, task.instructions)
        self.assertEqual(invocation.identity["provider"], "kimi-code")
        self.assertEqual(
            invocation.identity["model"],
            "kimi-code/kimi-for-coding",
        )
        self.assertEqual(invocation.stdout_media_type, "application/x-ndjson")
        self.assertEqual(
            invocation.command[
                invocation.command.index("--output-format") + 1
            ],
            "stream-json",
        )
        self.assertEqual(invocation.command[-2], "--prompt")
        self.assertEqual(invocation.command[-1], task.instructions)
        self.assertEqual(
            invocation.recorded_command[-1],
            "@agent/instructions.md",
        )
        self.assertNotIn(task.instructions, invocation.recorded_command)
        self.assertIn("KIMI_CODE_HOME", invocation.inherited_environment)
        self.assertNotIn(
            "KIMI_MODEL_THINKING_EFFORT",
            invocation.inherited_environment,
        )
        self.assertNotIn("KIMI_API_KEY", invocation.inherited_environment)

    def test_rejects_invalid_selections_and_tasks(self) -> None:
        for invalid in ("", "two models", "model\n", "x" * 129):
            with self.subTest(model=invalid):
                with self.assertRaisesRegex(ValueError, "model"):
                    KimiCode(model=invalid)
        with self.assertRaisesRegex(TypeError, "task must be AgentTask"):
            KimiCode(model="kimi-code/kimi-for-coding").build_invocation(
                object()  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
