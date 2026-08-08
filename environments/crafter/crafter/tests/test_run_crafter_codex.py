"""Tests for the Crafter Codex launcher's caller-owned Agent condition."""

from __future__ import annotations

import argparse
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

    def test_optional_positive_integer_parsers(self) -> None:
        for name in (
            "_max_train_index_uses",
            "_recommended_episodes_per_submission",
            "_minimum_candidate_evidence",
        ):
            parse = cast(Any, self.namespace[name])
            with self.subTest(parser=name):
                self.assertEqual(parse("1"), 1)
                self.assertEqual(parse("3"), 3)
                self.assertIsNone(parse("none"))
                self.assertIsNone(parse("NONE"))
                for invalid in ("0", "-1", "invalid"):
                    with self.subTest(invalid=invalid):
                        with self.assertRaises(argparse.ArgumentTypeError):
                            parse(invalid)

    def test_default_limit_is_two_and_can_be_disabled(self) -> None:
        parser = cast(Any, self.namespace["_parser"])()
        default = parser.parse_args(
            ["--model", "gpt-test", "--record-to", "runs/test"]
        )
        disabled = parser.parse_args(
            [
                "--model",
                "gpt-test",
                "--record-to",
                "runs/test",
                "--max-train-index-uses",
                "none",
                "--recommended-episodes-per-submission",
                "none",
                "--minimum-candidate-evidence",
                "none",
            ]
        )
        changed = parser.parse_args(
            [
                "--model",
                "gpt-test",
                "--record-to",
                "runs/test",
                "--max-train-index-uses",
                "4",
                "--recommended-episodes-per-submission",
                "24",
                "--minimum-candidate-evidence",
                "48",
            ]
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
        self.assertEqual(default.max_train_index_uses, 2)
        self.assertEqual(default.max_episodes_per_submission, 64)
        self.assertEqual(default.recommended_episodes_per_submission, 32)
        self.assertEqual(default.minimum_candidate_evidence, 64)
        self.assertIs(default.include_mp4_feedback, False)
        self.assertIsNone(disabled.max_train_index_uses)
        self.assertIsNone(disabled.recommended_episodes_per_submission)
        self.assertIsNone(disabled.minimum_candidate_evidence)
        self.assertEqual(changed.max_train_index_uses, 4)
        self.assertEqual(changed.recommended_episodes_per_submission, 24)
        self.assertEqual(changed.minimum_candidate_evidence, 48)
        self.assertIs(mp4_enabled.include_mp4_feedback, True)

    def test_launcher_rejects_inconsistent_evidence_guidance(self) -> None:
        main = cast(Any, self.namespace["main"])
        cases = (
            (
                "--max-episodes-per-submission",
                "16",
                "--recommended-episodes-per-submission",
                "32",
            ),
            (
                "--episode-budget",
                "32",
                "--minimum-candidate-evidence",
                "64",
            ),
        )
        for extra in cases:
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--model",
                            "gpt-test",
                            "--record-to",
                            "runs/test",
                            *extra,
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)

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

    def test_limit_retains_host_task_and_adds_recorded_instruction(self) -> None:
        agent_type = cast(Any, self.namespace["_CrafterCodex"])
        agent = agent_type(
            model="gpt-test",
            reasoning_effort="high",
            executable="codex-test",
            recommended_episodes_per_submission=32,
            minimum_candidate_evidence=64,
            max_train_index_uses=2,
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
        self.assertEqual(
            invocation.identity["recommended_episodes_per_submission"], "32"
        )
        self.assertEqual(invocation.identity["minimum_candidate_evidence"], "64")
        self.assertEqual(invocation.identity["max_train_index_uses"], "2")
        rendered = "\n".join(invocation.recorded_command)
        self.assertIn("developer_instructions=", rendered)
        self.assertIn("Run analysis scripts with `python` directly", rendered)
        self.assertIn("Do not prefix", rendered)
        self.assertIn("bare `uv run`", rendered)
        self.assertIn("about 32 Episodes per submission", rendered)
        self.assertIn("at least 64 total training Episode results", rendered)
        self.assertIn("exact same submitted Program revision", rendered)
        self.assertIn("may finish early", rendered)
        self.assertIn("at most 2 times", rendered)
        self.assertIn("at most 1 retry", rendered)
        self.assertIn("does not alter the Host's finish budget", rendered)

    def test_none_disables_only_the_index_limit_instruction(self) -> None:
        agent_type = cast(Any, self.namespace["_CrafterCodex"])
        agent = agent_type(
            model="gpt-test",
            reasoning_effort="high",
            executable="codex-test",
            recommended_episodes_per_submission=32,
            minimum_candidate_evidence=64,
            max_train_index_uses=None,
        )
        with patch(
            "evopolicygym.agents.codex.resolve_executable",
            return_value="/bin/codex-test",
        ):
            invocation = agent.build_invocation(
                AgentTask(instructions="unaltered Host task")
            )

        self.assertNotIn("max_train_index_uses", invocation.identity)
        self.assertEqual(invocation.identity["agent_python_tools"], "numpy,pillow")
        rendered = "\n".join(invocation.recorded_command)
        self.assertIn("developer_instructions=", rendered)
        self.assertIn("Run analysis scripts with `python` directly", rendered)
        self.assertIn("Training batch evidence guidance", rendered)
        self.assertNotIn("Training Episode diversity", rendered)

    def test_none_disables_batch_evidence_instruction(self) -> None:
        agent_type = cast(Any, self.namespace["_CrafterCodex"])
        agent = agent_type(
            model="gpt-test",
            reasoning_effort="high",
            executable="codex-test",
            recommended_episodes_per_submission=None,
            minimum_candidate_evidence=None,
            max_train_index_uses=None,
        )
        with patch(
            "evopolicygym.agents.codex.resolve_executable",
            return_value="/bin/codex-test",
        ):
            invocation = agent.build_invocation(
                AgentTask(instructions="unaltered Host task")
            )

        self.assertNotIn("recommended_episodes_per_submission", invocation.identity)
        self.assertNotIn("minimum_candidate_evidence", invocation.identity)
        rendered = "\n".join(invocation.recorded_command)
        self.assertNotIn("Training batch evidence guidance", rendered)


if __name__ == "__main__":
    unittest.main()
