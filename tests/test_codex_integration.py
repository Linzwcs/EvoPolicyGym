from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evopolicygym.agents import Codex, CodingAgent, resolve_executable
from evopolicygym.authoring import BenchmarkSpec
from evopolicygym.run import AssessmentConfig, RunConfig, ValidationConfig
from evopolicygym.run._task import build_agent_task


class CodexIntegrationTests(unittest.TestCase):
    def test_executable_resolution_is_shared_by_agent_integrations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "example-agent"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)

            resolved = resolve_executable(str(executable))

        self.assertEqual(resolved, str(executable.resolve()))

    def test_codex_selection_becomes_a_retained_process_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "codex"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)

            task = build_agent_task(
                BenchmarkSpec(
                    id="example/codex-v1",
                    description="Codex integration fixture.",
                    observation_space={"shape": [4]},
                    action_space={"enum": [0, 1]},
                    metadata={},
                    max_episode_steps=10,
                    primary_metric="reward",
                    score_direction="maximize",
                    environment_parameters={
                        "map_name": "4x4",
                        "is_slippery": True,
                    },
                ),
                RunConfig(
                    episode_budget=7,
                ),
            )
            invocation = Codex(
                model="fixture-model",
                executable=str(executable),
            ).build_invocation(task)

        self.assertIsInstance(
            Codex(model="fixture-model"),
            CodingAgent,
        )
        self.assertEqual(invocation.instructions, task.instructions)
        self.assertEqual(invocation.identity["provider"], "codex")
        self.assertEqual(invocation.identity["model"], "fixture-model")
        self.assertEqual(invocation.stdout_media_type, "application/x-ndjson")
        self.assertIn("--ephemeral", invocation.command)
        self.assertIn("--ignore-user-config", invocation.command)
        self.assertEqual(
            invocation.command[
                invocation.command.index("--sandbox") + 1
            ],
            "danger-full-access",
        )
        assert invocation.instructions is not None
        self.assertIn("whole Run has 7 Episode units", invocation.instructions)
        self.assertIn(
            "You decide",
            invocation.instructions,
        )
        self.assertIn("remaining Episode budget", invocation.instructions)
        self.assertIn(
            "content field and all Artifact contents are defined by",
            invocation.instructions,
        )
        self.assertIn('"environment_parameters": {', invocation.instructions)
        self.assertIn('"map_name": "4x4"', invocation.instructions)
        self.assertIn('"environment_digest": "sha256:', invocation.instructions)
        self.assertEqual(
            invocation.recorded_command[-1],
            "@agent/instructions.md",
        )

    def test_benchmark_skill_is_explicitly_enabled_per_run(self) -> None:
        spec = BenchmarkSpec(
            id="example/skill-v1",
            description="Skill selection fixture.",
            observation_space=None,
            action_space=None,
            metadata={},
            max_episode_steps=1,
            primary_metric="reward",
            score_direction="maximize",
            agent_skill="# Improve\n",
        )

        without_skill = build_agent_task(spec, RunConfig())
        with_skill = build_agent_task(
            spec,
            RunConfig(use_benchmark_skill=True),
        )

        self.assertNotIn("skill/SKILL.md", without_skill.instructions)
        self.assertIn("skill/SKILL.md", with_skill.instructions)

    def test_validation_handoff_rules_are_owned_by_the_host_task(self) -> None:
        spec = BenchmarkSpec(
            id="example/validation-task-v1",
            description="Validation task fixture.",
            observation_space=None,
            action_space=None,
            metadata={},
            max_episode_steps=1,
            primary_metric="loss",
            score_direction="minimize",
        )

        task = build_agent_task(
            spec,
            RunConfig(
                validation=ValidationConfig(
                    episodes_per_candidate=5,
                    max_candidates=3,
                ),
                assessment=AssessmentConfig(episodes=7),
            ),
        )

        self.assertIn(
            "evopolicygym finish SUBMISSION_ID [SUBMISSION_ID ...]",
            task.instructions,
        )
        self.assertIn(
            "Host evaluates every candidate on identical private Validation",
            task.instructions,
        )
        self.assertIn("loss (minimize)", task.instructions)
        self.assertIn("argument order", task.instructions)
        self.assertIn(
            "Host evaluates only the selected Program on",
            task.instructions,
        )
        self.assertIn(
            "Assessment never changes the selected candidate",
            task.instructions,
        )


if __name__ == "__main__":
    unittest.main()
