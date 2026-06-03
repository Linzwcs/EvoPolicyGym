"""Task document quality checks."""

from __future__ import annotations

from collections.abc import Iterable

from .run import Issue

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Objective",
    "Policy Interface",
    "Observation",
    "Action",
    "Reward",
)


def check_task_text(
    text: str,
    *,
    path: str,
    required: Iterable[str] = (),
) -> tuple[Issue, ...]:
    """Return issues for missing or incomplete agent-facing task text."""

    issues: list[Issue] = []
    body = text.strip()
    if not body:
        issues.append(Issue("task_text", path, "agent-facing task text must be non-empty"))
        return tuple(issues)

    for section in required:
        marker = f"## {section}"
        if marker not in text:
            issues.append(Issue("task_text_section", path, f"task text must include {marker}"))
    return tuple(issues)
