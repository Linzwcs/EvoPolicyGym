"""Detached Agent Session receipts and sanitized rejection values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionError:
    """Sanitized rejection returned to one Coding Agent request."""

    code: str
    message: str

    def __post_init__(self) -> None:
        for name in ("code", "message"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Agent-visible receipt for one committed Submission."""

    submission_id: str
    program_digest: str
    score: float
    episode_indices: tuple[int, ...]
    episodes_used: int
    episodes_remaining: int


@dataclass(frozen=True, slots=True)
class FinishReceipt:
    """Agent-visible receipt transferring candidate selection to the Host."""

    candidate_submission_ids: tuple[str, ...]


type SubmissionOutcome = SubmissionReceipt | SessionError
type FinishOutcome = FinishReceipt | SessionError


__all__: list[str] = []
