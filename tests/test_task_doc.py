from __future__ import annotations

import unittest

from evopolicygym.check.task import REQUIRED_SECTIONS, check_task_text


class TaskDocTest(unittest.TestCase):
    def test_accepts_required_sections(self) -> None:
        text = "\n".join(f"## {section}\nbody" for section in REQUIRED_SECTIONS)

        issues = check_task_text(text, path="env", required=REQUIRED_SECTIONS)

        self.assertEqual(issues, ())

    def test_reports_empty_and_missing_sections(self) -> None:
        empty = check_task_text("", path="env")
        partial = check_task_text("# Task\n\n## Objective\nbody", path="env", required=REQUIRED_SECTIONS)

        self.assertEqual(empty[0].code, "task_text")
        self.assertIn("task_text_section", {issue.code for issue in partial})
        self.assertTrue(any("## Reward" in issue.message for issue in partial))


if __name__ == "__main__":
    unittest.main()
