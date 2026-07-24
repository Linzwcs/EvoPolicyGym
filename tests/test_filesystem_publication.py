from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evopolicygym.program import Program
from evopolicygym.results import (
    EpisodeSummary,
    Feedback,
    SubmissionResult,
)
from evopolicygym.run._feedback import (
    FilesystemSubmissionPublisher,
    record_submission,
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
    def test_submission_is_frozen_after_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = root / "submissions"
            result = make_submission(root)
            source_mode_at_replace: int | None = None
            replace = os.replace

            def observing_replace(source: Path, destination: Path) -> None:
                nonlocal source_mode_at_replace
                source_mode_at_replace = stat.S_IMODE(source.stat().st_mode)
                replace(source, destination)

            with patch(
                "evopolicygym.run._feedback.os.replace",
                side_effect=observing_replace,
            ):
                record_submission(submissions, result)

            destination = submissions / result.submission_id
            self.assertEqual(source_mode_at_replace, 0o700)
            self.assertEqual(
                stat.S_IMODE(destination.stat().st_mode),
                0o555,
            )
            self.assertEqual(
                stat.S_IMODE(
                    (destination / "program" / "policy.py").stat().st_mode
                ),
                0o444,
            )
            self.assertEqual(
                stat.S_IMODE(
                    (destination / "feedback.json").stat().st_mode
                ),
                0o444,
            )

    def test_freeze_failure_removes_published_and_temporary_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = root / "submissions"
            result = make_submission(root)

            with (
                patch(
                    "evopolicygym.run._feedback._make_tree_read_only",
                    side_effect=OSError("freeze failed"),
                ),
                self.assertRaisesRegex(OSError, "freeze failed"),
            ):
                record_submission(submissions, result)

            self.assertFalse(
                (submissions / result.submission_id).exists()
            )
            self.assertEqual(
                list(
                    submissions.glob(
                        f".{result.submission_id}.tmp-*"
                    )
                ),
                [],
            )

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
