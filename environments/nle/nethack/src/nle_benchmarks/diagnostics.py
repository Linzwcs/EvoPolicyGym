"""Post-run diagnostics that do not affect candidate selection."""

from __future__ import annotations

import json

from evopolicygym.results import ValidationResult


def identical_validation_feedback_groups(
    validation: ValidationResult | None,
) -> list[list[str]]:
    """Group candidates with byte-equivalent JSON aggregate Validation data."""

    if validation is None:
        return []
    groups: dict[str, list[str]] = {}
    for candidate in validation.candidates:
        signature = json.dumps(
            {
                "score": candidate.score,
                "policy_failures": candidate.policy_failures,
                "feedback_content": candidate.feedback_content,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(signature, []).append(candidate.submission_id)
    return [identifiers for identifiers in groups.values() if len(identifiers) > 1]


__all__ = ["identical_validation_feedback_groups"]
