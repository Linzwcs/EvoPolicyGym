from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evopolicygym.program import Program
from evopolicygym.results import (
    EpisodeSummary,
    Feedback,
    SubmissionResult,
)
from evopolicygym.run._feedback import (
    FilesystemSubmissionPublisher,
)


def make_submission(root: Path) -> SubmissionResult:
    source = root / "source"
    source.mkdir()
    (source / "policy.py").write_text(
        "def make_policy(context):\n    return object()\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        submission_id="submission-000001",
        program=Program.from_directory(source),
        episodes_used=1,
        episodes_remaining=0,
        feedback=Feedback(score=1.0, content="fixture"),
        episodes=(
            EpisodeSummary(status="completed", reward=1.0, steps=1),
        ),
    )


class FilesystemSubmissionPublisherTests(unittest.TestCase):
    def test_failed_latest_update_rolls_back_both_submission_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = root / "submissions"
            feedback = root / "workspace" / "feedback"
            feedback.mkdir(parents=True)
            (feedback / "latest.json").mkdir()
            result = make_submission(root)
            publisher = FilesystemSubmissionPublisher(
                submissions_root=submissions,
                feedback_root=feedback,
            )

            with self.assertRaises(OSError):
                publisher.commit(result)

            self.assertFalse(
                (submissions / result.submission_id).exists()
            )
            self.assertFalse(
                (
                    feedback
                    / "submissions"
                    / result.submission_id
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
