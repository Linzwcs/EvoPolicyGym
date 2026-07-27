from __future__ import annotations

import unittest
from pathlib import Path

from evopolicygym.skills import AgentSkill

SKILLS = Path(__file__).parents[1] / "skills"


class RepositorySkillTests(unittest.TestCase):
    def test_every_first_party_skill_is_a_valid_directory_snapshot(self) -> None:
        directories = tuple(
            path
            for path in sorted(SKILLS.iterdir())
            if path.is_dir()
        )

        self.assertGreaterEqual(len(directories), 2)
        for directory in directories:
            with self.subTest(skill=directory.name):
                skill = AgentSkill.from_directory(directory)
                self.assertEqual(skill.name, directory.name)
                self.assertIn("SKILL.md", skill.files)
                self.assertTrue(skill.digest.startswith("sha256:"))

    def test_balatro_skill_retains_its_agent_metadata(self) -> None:
        skill = AgentSkill.from_directory(
            SKILLS / "optimize-balatro-policy"
        )

        self.assertIn("agents/openai.yaml", skill.files)
        instructions = skill.read_bytes("SKILL.md").decode()
        self.assertIn("expected Episode score", instructions)
        self.assertIn("Pool repeated evidence by digest", instructions)


if __name__ == "__main__":
    unittest.main()
