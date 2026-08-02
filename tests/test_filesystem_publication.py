from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evopolicygym.artifacts import Artifact
from evopolicygym.program import Program
from evopolicygym.results import (
    EpisodeSummary,
    Feedback,
    SubmissionResult,
)
from evopolicygym.run import _feedback as feedback_module
from evopolicygym.run._feedback import (
    FilesystemSubmissionPublisher,
    record_submission,
)


def make_submission(
    root: Path,
    *,
    submission_id: str = "submission-000001",
    artifacts: tuple[Artifact, ...] = (),
) -> SubmissionResult:
    source = root / f"source-{submission_id}"
    source.mkdir()
    (source / "policy.py").write_text(
        "def make_policy(context):\n    return object()\n",
        encoding="utf-8",
    )
    return SubmissionResult(
        submission_id=submission_id,
        program=Program.from_directory(source),
        episodes_used=1,
        episodes_remaining=0,
        feedback=Feedback(
            score=1.0,
            content="fixture",
            artifacts=artifacts,
        ),
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

    def test_old_bulk_is_evicted_from_host_and_workspace_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = root / "submissions"
            workspace = root / "workspace"
            feedback = workspace / "feedback"
            analysis = workspace / "analysis"
            analysis.mkdir(parents=True)
            (analysis / "agent-notes.txt").write_text(
                "retain me",
                encoding="utf-8",
            )
            publisher = FilesystemSubmissionPublisher(
                submissions_root=submissions,
                feedback_root=feedback,
                bulk_retention_bytes=8,
            )
            old = make_submission(
                root,
                submission_id="submission-000001",
                artifacts=(
                    Artifact(
                        "bulk/observations-000000.npz",
                        "application/x-npz",
                        b"bulk",
                        retention="bulk",
                    ),
                    Artifact(
                        "artifact-manifest.json",
                        "application/json",
                        b"keep",
                    ),
                ),
            )
            newest = make_submission(
                root,
                submission_id="submission-000002",
                artifacts=(
                    Artifact(
                        "bulk/trajectory-000000.jsonl.gz",
                        "application/gzip",
                        b"bulk",
                        retention="bulk",
                    ),
                    Artifact(
                        "artifact-manifest.json",
                        "application/json",
                        b"keep",
                    ),
                ),
            )

            publisher.commit(old)
            agent_bulk_directory = (
                feedback
                / "submissions"
                / old.submission_id
                / "artifacts"
                / "bulk"
            )
            os.chmod(agent_bulk_directory, 0o700)
            agent_derived = agent_bulk_directory / "agent-selected-note.txt"
            agent_derived.write_text("derived", encoding="utf-8")
            os.chmod(agent_bulk_directory, 0o555)
            publisher.commit(newest)

            for root_view in (submissions, feedback / "submissions"):
                old_root = root_view / old.submission_id
                newest_root = root_view / newest.submission_id
                self.assertFalse(
                    (
                        old_root
                        / "artifacts"
                        / "bulk"
                        / "observations-000000.npz"
                    ).exists()
                )
                self.assertTrue(
                    (old_root / "artifacts" / "artifact-manifest.json").is_file()
                )
                self.assertTrue((old_root / "feedback.json").is_file())
                self.assertTrue(
                    (
                        newest_root
                        / "artifacts"
                        / "bulk"
                        / "trajectory-000000.jsonl.gz"
                    ).is_file()
                )
                availability = json.loads(
                    (old_root / "availability.json").read_text(encoding="utf-8")
                )
                self.assertEqual(availability["bulk_status"], "evicted")
                self.assertIs(
                    availability["bulk_artifacts"][0]["available"],
                    False,
                )

            self.assertEqual(
                (analysis / "agent-notes.txt").read_text(encoding="utf-8"),
                "retain me",
            )
            self.assertEqual(
                agent_derived.read_text(encoding="utf-8"),
                "derived",
            )
            retention = json.loads(
                (feedback / "retention.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                retention["protected_submission_id"],
                newest.submission_id,
            )
            self.assertEqual(
                retention["evicted_submission_ids"],
                [old.submission_id],
            )
            self.assertIs(retention["over_limit_to_preserve_latest"], False)

    def test_newest_bulk_is_preserved_even_when_it_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = root / "submissions"
            feedback = root / "workspace" / "feedback"
            result = make_submission(
                root,
                artifacts=(
                    Artifact(
                        "bulk/observations-000000.npz",
                        "application/x-npz",
                        b"larger-than-limit",
                        retention="bulk",
                    ),
                ),
            )
            publisher = FilesystemSubmissionPublisher(
                submissions_root=submissions,
                feedback_root=feedback,
                bulk_retention_bytes=1,
            )

            publisher.commit(result)

            for root_view in (submissions, feedback / "submissions"):
                self.assertTrue(
                    (
                        root_view
                        / result.submission_id
                        / "artifacts"
                        / "bulk"
                        / "observations-000000.npz"
                    ).is_file()
                )
            retention = json.loads(
                (feedback / "retention.json").read_text(encoding="utf-8")
            )
            self.assertIs(retention["over_limit_to_preserve_latest"], True)

    def test_failed_paired_eviction_restores_both_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submissions = root / "submissions"
            feedback = root / "workspace" / "feedback"
            publisher = FilesystemSubmissionPublisher(
                submissions_root=submissions,
                feedback_root=feedback,
                bulk_retention_bytes=8,
            )
            old = make_submission(
                root,
                submission_id="submission-000001",
                artifacts=(
                    Artifact(
                        "bulk/observations-000000.npz",
                        "application/x-npz",
                        b"bulk",
                        retention="bulk",
                    ),
                ),
            )
            newest = make_submission(
                root,
                submission_id="submission-000002",
                artifacts=(
                    Artifact(
                        "bulk/observations-000000.npz",
                        "application/x-npz",
                        b"bulk",
                        retention="bulk",
                    ),
                ),
            )
            publisher.commit(old)
            workspace_old = (
                feedback / "submissions" / old.submission_id
            )
            replace_json = feedback_module._replace_json

            def fail_workspace_availability(
                path: Path,
                document: dict[str, object],
            ) -> None:
                if (
                    path == workspace_old / "availability.json"
                    and document.get("bulk_status") == "evicted"
                ):
                    raise OSError("workspace availability failed")
                replace_json(path, document)

            with patch(
                "evopolicygym.run._feedback._replace_json",
                side_effect=fail_workspace_availability,
            ):
                publisher.commit(newest)

            for root_view in (submissions, feedback / "submissions"):
                old_root = root_view / old.submission_id
                self.assertTrue(
                    (
                        old_root
                        / "artifacts"
                        / "bulk"
                        / "observations-000000.npz"
                    ).is_file()
                )
                availability = json.loads(
                    (old_root / "availability.json").read_text(encoding="utf-8")
                )
                self.assertEqual(availability["bulk_status"], "available")
            retention = json.loads(
                (feedback / "retention.json").read_text(encoding="utf-8")
            )
            self.assertEqual(retention["status"], "failed")
            self.assertEqual(retention["error_type"], "OSError")


if __name__ == "__main__":
    unittest.main()
