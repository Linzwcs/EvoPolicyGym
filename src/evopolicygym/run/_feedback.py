"""Directory-backed publication of Benchmark-authorized Feedback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from ..artifacts import Artifact
from ..results import EpisodeSummary, SubmissionResult
from ._json import encode_public_json_value

_FEEDBACK_SCHEMA = "evopolicygym/feedback/v2"
_AVAILABILITY_SCHEMA = "evopolicygym/artifact-availability/v1"
_RETENTION_SCHEMA = "evopolicygym/bulk-feedback-retention/v1"
_SUBMISSION_ID = re.compile(r"submission-[0-9]{6}")


class FilesystemSubmissionPublisher:
    """Commit matching Host and Agent-visible Submission bundles."""

    def __init__(
        self,
        *,
        submissions_root: Path,
        feedback_root: Path,
        bulk_retention_bytes: int | None = None,
    ) -> None:
        if bulk_retention_bytes is not None and (
            type(bulk_retention_bytes) is not int
            or bulk_retention_bytes <= 0
        ):
            raise ValueError(
                "bulk_retention_bytes must be a positive integer or None"
            )
        self._submissions_root = submissions_root
        self._feedback_root = feedback_root
        self._bulk_retention_bytes = bulk_retention_bytes

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
        if self._bulk_retention_bytes is not None:
            try:
                _enforce_bulk_retention(
                    submissions_root=self._submissions_root,
                    feedback_root=self._feedback_root,
                    protected_submission_id=result.submission_id,
                    limit_bytes=self._bulk_retention_bytes,
                )
            except Exception as error:
                _record_retention_failure(
                    self._feedback_root,
                    protected_submission_id=result.submission_id,
                    limit_bytes=self._bulk_retention_bytes,
                    error=error,
                )


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
    _write_json(
        submission_root / "availability.json",
        _availability_document(
            result.submission_id,
            artifacts,
            reason=None,
        ),
    )


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
                "retention": artifact.retention,
            }
        )
    return documents


def _availability_document(
    submission_id: str,
    artifacts: list[dict[str, object]],
    *,
    reason: str | None,
    submission_root: Path | None = None,
) -> dict[str, object]:
    bulk: list[dict[str, object]] = []
    available_count = 0
    for artifact in artifacts:
        if artifact.get("retention") != "bulk":
            continue
        path = artifact.get("path")
        available = bool(
            type(path) is str
            and (
                submission_root is None
                or (submission_root / path).is_file()
            )
        )
        available_count += available
        bulk.append(
            {
                "name": artifact["name"],
                "path": artifact["path"],
                "size": artifact["size"],
                "sha256": artifact["sha256"],
                "available": available,
            }
        )
    status = "available"
    if bulk and available_count == 0:
        status = "evicted"
    elif available_count != len(bulk):
        status = "partial"
    document: dict[str, object] = {
        "schema": _AVAILABILITY_SCHEMA,
        "submission_id": submission_id,
        "bulk_status": status,
        "bulk_artifacts": bulk,
    }
    if reason is not None:
        document["reason"] = reason
    return document


def _enforce_bulk_retention(
    *,
    submissions_root: Path,
    feedback_root: Path,
    protected_submission_id: str,
    limit_bytes: int,
) -> None:
    workspace_submissions = feedback_root / "submissions"
    identifiers = sorted(
        path.name
        for path in submissions_root.iterdir()
        if path.is_dir() and _SUBMISSION_ID.fullmatch(path.name)
    )
    retained_bytes = _combined_bulk_bytes(
        submissions_root,
        workspace_submissions,
        identifiers,
    )
    evicted: list[str] = []
    for submission_id in identifiers:
        if retained_bytes <= limit_bytes:
            break
        if submission_id == protected_submission_id:
            continue
        host_submission = submissions_root / submission_id
        artifacts = _feedback_artifacts(host_submission)
        freed = _evict_bulk_artifacts(
            (
                host_submission,
                workspace_submissions / submission_id,
            ),
            submission_id,
            artifacts,
        )
        if freed:
            retained_bytes -= freed
            evicted.append(submission_id)

    _replace_json(
        feedback_root / "retention.json",
        {
            "schema": _RETENTION_SCHEMA,
            "limit_bytes": limit_bytes,
            "retained_bytes_across_host_and_workspace": retained_bytes,
            "protected_submission_id": protected_submission_id,
            "evicted_submission_ids": evicted,
            "over_limit_to_preserve_latest": retained_bytes > limit_bytes,
            "status": "complete",
        },
    )


def _combined_bulk_bytes(
    submissions_root: Path,
    workspace_submissions: Path,
    identifiers: list[str],
) -> int:
    total = 0
    for submission_id in identifiers:
        host_submission = submissions_root / submission_id
        artifacts = _feedback_artifacts(host_submission)
        for root in (submissions_root, workspace_submissions):
            submission_root = root / submission_id
            if not submission_root.is_dir():
                continue
            for artifact in artifacts:
                if artifact.get("retention") != "bulk":
                    continue
                path = artifact.get("path")
                if type(path) is not str:
                    raise ValueError("Feedback artifact path is invalid")
                candidate = _artifact_target(submission_root, path)
                if candidate.is_file() and not candidate.is_symlink():
                    total += candidate.stat().st_size
    return total


def _evict_bulk_artifacts(
    submission_roots: tuple[Path, Path],
    submission_id: str,
    artifacts: list[dict[str, object]],
) -> int:
    if any(not root.is_dir() or root.is_symlink() for root in submission_roots):
        raise ValueError("matching Host and workspace submissions are required")
    targets: list[tuple[Path, int]] = []
    freed = 0
    for submission_root in submission_roots:
        for artifact in artifacts:
            if artifact.get("retention") != "bulk":
                continue
            relative = artifact.get("path")
            if type(relative) is not str:
                raise ValueError("Feedback artifact path is invalid")
            target = _artifact_target(submission_root, relative)
            if target.is_symlink():
                raise ValueError("Feedback bulk artifact must not be a symlink")
            if target.is_file():
                size = target.stat().st_size
                targets.append((target, size))
                freed += size

    mutable_directories = set(submission_roots)
    for target, _ in targets:
        current = target.parent
        submission_root = next(
            root for root in submission_roots if target.is_relative_to(root)
        )
        while current != submission_root:
            mutable_directories.add(current)
            current = current.parent
    for directory in sorted(mutable_directories, key=lambda path: len(path.parts)):
        os.chmod(directory, 0o700)
    staged: list[tuple[Path, Path]] = []
    try:
        for target, _ in targets:
            temporary = target.with_name(
                f".{target.name}.evict-{uuid.uuid4().hex}"
            )
            os.replace(target, temporary)
            staged.append((target, temporary))
        for submission_root in submission_roots:
            _replace_json(
                submission_root / "availability.json",
                _availability_document(
                    submission_id,
                    artifacts,
                    reason="run_bulk_feedback_capacity",
                    submission_root=submission_root,
                ),
            )
    except BaseException:
        for target, temporary in reversed(staged):
            if temporary.is_file() and not target.exists():
                os.replace(temporary, target)
        for submission_root in submission_roots:
            try:
                _replace_json(
                    submission_root / "availability.json",
                    _availability_document(
                        submission_id,
                        artifacts,
                        reason=None,
                        submission_root=submission_root,
                    ),
                )
            except Exception:
                pass
        raise
    else:
        for _, temporary in staged:
            try:
                temporary.unlink()
            except OSError:
                pass
    finally:
        for directory in sorted(
            mutable_directories,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
    return freed


def _artifact_target(submission_root: Path, relative: str) -> Path:
    parts = relative.split("/")
    if (
        not relative.startswith("artifacts/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("Feedback artifact path is invalid")
    return submission_root.joinpath(*parts)


def _feedback_artifacts(submission_root: Path) -> list[dict[str, object]]:
    document = json.loads(
        (submission_root / "feedback.json").read_text(encoding="utf-8")
    )
    artifacts = document.get("artifacts")
    if type(artifacts) is not list or any(
        type(artifact) is not dict for artifact in artifacts
    ):
        raise ValueError("Feedback artifact manifest is invalid")
    return artifacts


def _record_retention_failure(
    feedback_root: Path,
    *,
    protected_submission_id: str,
    limit_bytes: int,
    error: Exception,
) -> None:
    try:
        _replace_json(
            feedback_root / "retention.json",
            {
                "schema": _RETENTION_SCHEMA,
                "limit_bytes": limit_bytes,
                "protected_submission_id": protected_submission_id,
                "status": "failed",
                "error_type": type(error).__name__,
            },
        )
    except Exception:
        pass


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
