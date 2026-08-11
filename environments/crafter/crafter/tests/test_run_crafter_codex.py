"""Tests for the Crafter Codex launcher's caller-owned Agent condition."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

from evopolicygym.agents import AgentTask

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_crafter_codex.py"


class CrafterCodexLauncherTests(unittest.TestCase):
    namespace: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = runpy.run_path(str(_SCRIPT))

    def test_parser_retains_submission_limit_and_feedback_options(self) -> None:
        parser = cast(Any, self.namespace["_parser"])()
        default = parser.parse_args(
            ["--model", "gpt-test", "--record-to", "runs/test"]
        )
        mp4_enabled = parser.parse_args(
            [
                "--model",
                "gpt-test",
                "--record-to",
                "runs/test",
                "--include-mp4-feedback",
            ]
        )
        lhs = parser.parse_args(
            [
                "--model",
                "gpt-test",
                "--record-to",
                "runs/test",
                "--profile",
                "lhs",
            ]
        )
        symbolic = parser.parse_args(
            [
                "--model",
                "gpt-test",
                "--record-to",
                "runs/test",
                "--observation-profile",
                "local-symbolic-v1",
            ]
        )
        self.assertEqual(default.max_episodes_per_submission, 64)
        self.assertEqual(default.profile, "lhs")
        self.assertEqual(default.observation_profile, "rgb")
        self.assertIs(default.include_mp4_feedback, False)
        self.assertIs(mp4_enabled.include_mp4_feedback, True)
        self.assertEqual(lhs.profile, "lhs")
        self.assertEqual(symbolic.observation_profile, "local-symbolic-v1")

    def test_launcher_rejects_overlong_session_socket_path(self) -> None:
        main = cast(Any, self.namespace["main"])
        record_to = Path("/tmp") / ("crafter-run-" + "x" * 100)
        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "--model",
                    "gpt-test",
                    "--record-to",
                    str(record_to),
                    "--allow-unsafe-process",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_launcher_adds_player_guide_and_no_reuse_instructions(self) -> None:
        agent_type = cast(Any, self.namespace["_CrafterCodex"])
        agent = agent_type(
            model="gpt-test",
            reasoning_effort="high",
            executable="codex-test",
        )
        task = AgentTask(instructions="unaltered Host task")
        with patch(
            "evopolicygym.agents.codex.resolve_executable",
            return_value="/bin/codex-test",
        ):
            invocation = agent.build_invocation(task)

        self.assertEqual(invocation.instructions, task.instructions)
        self.assertEqual(invocation.command[-1], task.instructions)
        self.assertEqual(invocation.recorded_command[-1], "@agent/instructions.md")
        self.assertEqual(invocation.identity["max_train_index_uses"], "1")
        rendered = "\n".join(invocation.recorded_command)
        self.assertIn("developer_instructions=", rendered)
        self.assertIn(
            "Before the first submission, read program/PLAYER_GUIDE.md in full",
            rendered,
        )
        self.assertIn(
            "authoritative gameplay-mechanics reference", rendered
        )
        self.assertIn(
            "Benchmark public specification as the authoritative evaluation",
            rendered,
        )
        self.assertIn(
            "not a prescribed Policy or fixed Action plan", rendered
        )
        self.assertIn(
            "Use `python` directly for Crafter feedback analysis", rendered
        )
        self.assertIn("NumPy and Pillow", rendered)
        self.assertIn("Do not invoke", rendered)
        self.assertIn("`uv run`", rendered)
        self.assertIn("may be selected at most once", rendered)
        self.assertIn(
            "Never select an index that has already been evaluated", rendered
        )
        self.assertIn("Every new", rendered)
        self.assertIn("submission must use previously unseen indices", rendered)
        self.assertIn("This requirement does not", rendered)
        self.assertIn("prescribe a submission size", rendered)
        self.assertIn("or require full budget consumption", rendered)
        self.assertNotIn("Training batch evidence guidance", rendered)
        self.assertNotIn("about 32 Episodes", rendered)
        self.assertNotIn("at least 64 total", rendered)
        self.assertNotIn("matched-index", rendered)
        self.assertNotIn("Host statement that index reuse", rendered)

    def test_symbolic_launcher_selects_matching_program_skill_and_instruction(
        self,
    ) -> None:
        starting_program = cast(Any, self.namespace["_starting_program"])
        skill_directory = cast(Any, self.namespace["_benchmark_skill_directory"])
        symbolic_program = starting_program("local-symbolic-v1")
        self.assertIn("policy.py", symbolic_program.files)
        self.assertIn(
            b"local-symbolic Crafter starting Policy",
            symbolic_program.read_bytes("policy.py"),
        )
        self.assertEqual(
            skill_directory("local-symbolic-v1").name,
            "optimize-crafter-local-symbolic-policy",
        )

        agent_type = cast(Any, self.namespace["_CrafterCodex"])
        agent = agent_type(
            model="gpt-test",
            reasoning_effort="high",
            executable="codex-test",
            observation_profile="local-symbolic-v1",
        )
        task = AgentTask(instructions="unaltered Host task")
        with patch(
            "evopolicygym.agents.codex.resolve_executable",
            return_value="/bin/codex-test",
        ):
            invocation = agent.build_invocation(task)
        rendered = "\n".join(invocation.recorded_command)
        self.assertEqual(
            invocation.identity["crafter_observation_profile"],
            "local-symbolic-v1",
        )
        self.assertIn("lossless local-symbolic NPZ", rendered)
        self.assertNotIn("producing images", rendered)


if __name__ == "__main__":
    unittest.main()
