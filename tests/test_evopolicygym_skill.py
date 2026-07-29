from __future__ import annotations

import posixpath
import re
import unittest
from pathlib import Path

from evopolicygym.skills import AgentSkill

SKILL_DIRECTORY = (
    Path(__file__).parents[1]
    / "skills"
    / "evopolicygym"
)


class EvoPolicyGymSkillTests(unittest.TestCase):
    def test_skill_routes_caller_side_workflows(self) -> None:
        skill = AgentSkill.from_directory(SKILL_DIRECTORY)

        self.assertEqual(skill.name, "evopolicygym")
        self.assertTrue(
            {
                "agents/openai.yaml",
                "references/setup.md",
                "references/evaluation.md",
                "references/run.md",
                "references/providers.md",
                "references/authoring.md",
                "references/diagnostics.md",
            }.issubset(skill.files)
        )
        self.assertNotIn("references/api.md", skill.files)

        instructions = skill.read_bytes("SKILL.md").decode()
        self.assertIn("caller-side EvoPolicyGym work", instructions)
        self.assertIn("Host-generated task is authoritative", instructions)
        self.assertIn("Benchmark-strategy Skill", instructions)
        self.assertNotIn("evopolicygym-session submit", instructions)

        metadata = skill.read_bytes("agents/openai.yaml").decode()
        self.assertIn('display_name: "EvoPolicyGym"', metadata)
        self.assertIn("$evopolicygym", metadata)

    def test_all_local_markdown_links_resolve_inside_the_snapshot(self) -> None:
        skill = AgentSkill.from_directory(SKILL_DIRECTORY)
        markdown_links = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

        for source in skill.files:
            if not source.endswith(".md"):
                continue
            content = skill.read_bytes(source).decode()
            for target in markdown_links.findall(content):
                if target.startswith(("https://", "http://", "#")):
                    continue
                relative_target = target.split("#", 1)[0]
                resolved = posixpath.normpath(
                    posixpath.join(
                        posixpath.dirname(source),
                        relative_target,
                    )
                )
                with self.subTest(source=source, target=target):
                    self.assertIn(resolved, skill.files)


if __name__ == "__main__":
    unittest.main()
