"""Validation and Assessment report projections from detached results."""

from __future__ import annotations

import os
from pathlib import Path

from ...errors import AgentRunError
from ...results import AssessmentResult, ValidationResult
from .._json import encode_public_json_value
from .writer import write_json_atomic

_VALIDATION_REPORT_SCHEMA = "evopolicygym/validation-report/v2"
_ASSESSMENT_REPORT_SCHEMA = "evopolicygym/assessment-report/v2"


def write_validation_report(
    directory: Path,
    validation: ValidationResult,
) -> None:
    try:
        directory.mkdir(mode=0o700)
    except OSError as error:
        raise AgentRunError(
            "Validation record could not be committed"
        ) from error
    write_json_atomic(
        directory / "report.json",
        {
            "schema": _VALIDATION_REPORT_SCHEMA,
            "split": validation.split,
            "episodes_per_candidate": validation.episodes_per_candidate,
            "primary_metric": validation.primary_metric,
            "score_direction": validation.score_direction,
            "candidates": [
                {
                    "submission_id": candidate.submission_id,
                    "program_digest": candidate.program_digest,
                    "score": candidate.score,
                    "episodes": candidate.episodes,
                    "policy_failures": candidate.policy_failures,
                    "feedback_content": encode_public_json_value(
                        candidate.feedback_content
                    ),
                }
                for candidate in validation.candidates
            ],
            "selected_submission_id": validation.selected_submission_id,
        },
        error_message="Validation record could not be committed",
    )
    _freeze_directory(
        directory,
        error_message="Validation record could not be committed",
    )


def write_assessment_report(
    directory: Path,
    assessment: AssessmentResult,
) -> None:
    try:
        directory.mkdir(mode=0o700)
    except OSError as error:
        raise AgentRunError(
            "Assessment record could not be committed"
        ) from error
    write_json_atomic(
        directory / "report.json",
        {
            "schema": _ASSESSMENT_REPORT_SCHEMA,
            "submission_id": assessment.submission_id,
            "program_digest": assessment.program_digest,
            "split": assessment.split,
            "episodes": assessment.episodes,
            "primary_metric": assessment.primary_metric,
            "score_direction": assessment.score_direction,
            "score": assessment.score,
            "policy_failures": assessment.policy_failures,
            "feedback_content": encode_public_json_value(
                assessment.feedback_content
            ),
        },
        error_message="Assessment record could not be committed",
    )
    _freeze_directory(
        directory,
        error_message="Assessment record could not be committed",
    )


def _freeze_directory(directory: Path, *, error_message: str) -> None:
    try:
        os.chmod(directory, 0o500)
    except OSError as error:
        raise AgentRunError(error_message) from error


__all__: list[str] = []
