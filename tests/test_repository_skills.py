from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

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
        self.assertIn("references/experiment-catalog.md", skill.files)
        self.assertIn("references/experiment-protocol.md", skill.files)
        self.assertIn("references/implementation-playbook.md", skill.files)
        self.assertIn("references/modeling-stack.md", skill.files)
        self.assertIn("references/modularity-guide.md", skill.files)
        self.assertIn("references/strategy-lessons.md", skill.files)
        self.assertIn(
            "scripts/compare_indexed_feedback.py",
            skill.files,
        )
        self.assertIn("scripts/summarize_evidence.py", skill.files)
        instructions = skill.read_bytes("SKILL.md").decode()
        self.assertIn("expected Episode score", instructions)
        self.assertIn("Pool repeated evidence by digest", instructions)
        self.assertIn(
            "partitioning a train-only Episode pool and budget",
            instructions,
        )
        self.assertIn(
            "can choose and reuse Run-local Episode indices",
            instructions,
        )
        self.assertIn("Host Validation and Assessment", instructions)
        protocol = skill.read_bytes(
            "references/experiment-protocol.md"
        ).decode()
        self.assertIn("receives only train submissions", protocol)
        self.assertIn("matched train A/B", protocol)
        self.assertIn("pre-reserved unseen train indices", protocol)
        self.assertIn("Confirmation is still train evidence", protocol)
        catalog = skill.read_bytes(
            "references/experiment-catalog.md"
        ).decode()
        self.assertIn("opportunity audit before changing code", catalog)
        self.assertIn("Threat-ratio spending", catalog)
        implementation = skill.read_bytes(
            "references/implementation-playbook.md"
        ).decode()
        self.assertIn("Add a capability in layers", implementation)
        self.assertIn("Implement multi-observation transactions", implementation)
        modeling = skill.read_bytes(
            "references/modeling-stack.md"
        ).decode()
        self.assertIn("Fix the lowest", modeling)
        self.assertIn("Do not let L6 decide strategy", modeling)
        self.assertIn("Migrate an existing heuristic Policy", modeling)
        modularity = skill.read_bytes(
            "references/modularity-guide.md"
        ).decode()
        self.assertIn("One responsibility, one owner", modularity)
        self.assertIn("Do not use line count alone", modularity)
        self.assertIn(
            "Refactor and strategy change are separate experiments",
            modularity,
        )

    def test_balatro_evidence_summarizer_separates_wins_from_progress(
        self,
    ) -> None:
        digest = f"sha256:{'a' * 64}"
        document: dict[str, Any] = {
            "baseline": {
                "program_digest": f"sha256:{'b' * 64}",
                "episodes": [
                    {
                        "episode_index": 0,
                        "reward": 5,
                        "failure": None,
                    },
                    {
                        "episode_index": 1,
                        "reward": 12,
                        "failure": None,
                    },
                    {
                        "episode_index": 2,
                        "reward": 7,
                        "failure": None,
                    },
                ],
            },
            "candidate": {
                "program_digest": digest,
                "episodes": [
                    {
                        "episode_index": 0,
                        "reward": 11,
                        "failure": None,
                    },
                    {
                        "episode_index": 1,
                        "reward": 1024,
                        "failure": None,
                    },
                    {
                        "episode_index": 2,
                        "reward": None,
                        "failure": "invalid_action",
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "comparison.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILLS
                        / "optimize-balatro-policy"
                        / "scripts"
                        / "summarize_evidence.py"
                    ),
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            document["baseline"]["episodes"][0]["episode_index"] = 9
            source.write_text(json.dumps(document), encoding="utf-8")
            mismatched = subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILLS
                        / "optimize-balatro-policy"
                        / "scripts"
                        / "summarize_evidence.py"
                    ),
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        summary = json.loads(completed.stdout)
        self.assertEqual(summary["program_digests"], [digest])
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["policy_failures"], 1)
        self.assertEqual(summary["mean_blinds"], 35 / 3)
        self.assertEqual(summary["paired"]["improved"], 2)
        self.assertEqual(summary["paired"]["regressed"], 1)
        self.assertNotEqual(mismatched.returncode, 0)
        self.assertIn(
            "paired Episode indices differ",
            mismatched.stderr,
        )

    def test_balatro_indexed_feedback_comparison_requires_matched_indices(
        self,
    ) -> None:
        baseline: dict[str, Any] = {
            "program_digest": f"sha256:{'b' * 64}",
            "episodes": [
                {
                    "episode_index": 2,
                    "reward": 5,
                    "failure": None,
                },
                {
                    "episode_index": 7,
                    "reward": 12,
                    "failure": None,
                },
            ],
        }
        candidate: dict[str, Any] = {
            "program_digest": f"sha256:{'c' * 64}",
            "episodes": [
                {
                    "episode_index": 2,
                    "reward": 11,
                    "failure": None,
                },
                {
                    "episode_index": 7,
                    "reward": 1024,
                    "failure": None,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            candidate_path = Path(directory) / "candidate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            command = [
                sys.executable,
                str(
                    SKILLS
                    / "optimize-balatro-policy"
                    / "scripts"
                    / "compare_indexed_feedback.py"
                ),
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            candidate["episodes"][0]["episode_index"] = 3
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            mismatched = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

        comparison = json.loads(completed.stdout)
        self.assertEqual(comparison["matched"]["unique_episode_indices"], 2)
        self.assertEqual(comparison["matched"]["improved"], 2)
        self.assertEqual(comparison["candidate"]["wins"], 1)
        self.assertNotEqual(mismatched.returncode, 0)
        self.assertIn(
            "Episode index sets differ",
            mismatched.stderr,
        )


if __name__ == "__main__":
    unittest.main()
