"""Directory-backed publication of Benchmark-authorized Feedback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from ..artifacts import Artifact
from ..results import EpisodeSummary, SubmissionResult
from ._json import encode_public_json_value

_FEEDBACK_SCHEMA = "evopolicygym/feedback/v1"


class FilesystemSubmissionPublisher:
    """Commit matching Host and Agent-visible Submission bundles."""

    def __init__(
        self,
        *,
        submissions_root: Path,
        feedback_root: Path,
    ) -> None:
        self._submissions_root = submissions_root
        self._feedback_root = feedback_root

    def commit(self, result: SubmissionResult) -> None:
        try:
            record_submission(self._submissions_root, result)
            publish_feedback(self._feedback_root, result)
        except Exception:
            for root in (
                self._submissions_root,
                self._feedback_root / "submissions",
            ):
                try:
                    _discard_tree(root / result.submission_id)
                except Exception:
                    pass
            raise


def publish_feedback(feedback_root: Path, result: SubmissionResult) -> None:
    """Publish one complete Agent-facing Feedback bundle and advance ``latest``."""

    submissions_root = feedback_root / "submissions"
    submissions_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    destination = submissions_root / result.submission_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("submission publication already exists")

    temporary = submissions_root / f".{result.submission_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        _materialize_feedback(temporary, result)
        _commit_read_only_tree(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    latest: dict[str, object] = {
        "schema": _FEEDBACK_SCHEMA,
        "submission_id": result.submission_id,
        "program_digest": result.program_digest,
        "score": result.feedback.score,
        "feedback": f"submissions/{result.submission_id}/feedback.json",
    }
    _replace_json(feedback_root / "latest.json", latest)


def record_submission(submissions_root: Path, result: SubmissionResult) -> None:
    """Atomically retain one Host-owned Program and Feedback bundle."""

    submissions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = submissions_root / result.submission_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("submission record already exists")
    temporary = submissions_root / f".{result.submission_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        result.program.write_to(temporary / "program")
        _materialize_feedback(temporary, result)
        _commit_read_only_tree(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def discard_submission_record(
    submissions_root: Path,
    submission_id: str,
) -> None:
    """Best-effort rollback before a submission becomes Session-visible."""

    _discard_tree(submissions_root / submission_id)


def _discard_tree(destination: Path) -> None:
    if not destination.is_dir() or destination.is_symlink():
        return
    for directory, directories, files in os.walk(destination):
        path = Path(directory)
        os.chmod(path, 0o700)
        for name in directories:
            os.chmod(path / name, 0o700)
        for name in files:
            os.chmod(path / name, 0o600)
    shutil.rmtree(destination)


def _commit_read_only_tree(temporary: Path, destination: Path) -> None:
    # Keep the staging root writable through the rename. macOS may reject
    # replacing a directory after the source tree itself has been frozen.
    os.replace(temporary, destination)
    try:
        _make_tree_read_only(destination)
    except BaseException:
        try:
            _discard_tree(destination)
        except Exception:
            pass
        raise


def _materialize_feedback(
    submission_root: Path,
    result: SubmissionResult,
) -> None:
    artifacts = _materialize_artifacts(
        submission_root,
        result.feedback.artifacts,
    )
    feedback_document: dict[str, object] = {
        "schema": _FEEDBACK_SCHEMA,
        "submission_id": result.submission_id,
        "program_digest": result.program_digest,
        "episodes_used": result.episodes_used,
        "episodes_remaining": result.episodes_remaining,
        "score": result.feedback.score,
        "content": encode_public_json_value(result.feedback.content),
        "episodes": [_episode_document(item) for item in result.episodes],
        "artifacts": artifacts,
    }
    _write_json(submission_root / "feedback.json", feedback_document)


def _materialize_artifacts(
    submission_root: Path,
    artifacts: tuple[Artifact, ...],
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for artifact in artifacts:
        relative = f"artifacts/{artifact.name}"
        destination = submission_root.joinpath(*relative.split("/"))
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        content = artifact.read_bytes()
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        documents.append(
            {
                "name": artifact.name,
                "media_type": artifact.media_type,
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return documents


def _episode_document(episode: EpisodeSummary) -> dict[str, object]:
    return {
        "status": episode.status,
        "reward": episode.reward,
        "steps": episode.steps,
        "failure": episode.failure,
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json(path: Path, document: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_json(temporary, document)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _make_tree_read_only(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        path = Path(directory)
        for name in files:
            os.chmod(path / name, 0o444)
        os.chmod(path, 0o555)


__all__: list[str] = []
