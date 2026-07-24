"""Immutable public feedback and result values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .artifacts import (
    FEEDBACK_MAX_ARTIFACT_BYTES,
    FEEDBACK_MAX_ARTIFACTS,
    Artifact,
)
from .policy import PolicyValue, copy_policy_value
from .program import Program

type PolicyFailureCode = Literal[
    "exception",
    "timeout",
    "invalid_action",
    "protocol_error",
]
type EpisodeStatus = Literal["completed", "policy_failed"]

type RunTerminalReason = Literal[
    "finished",
    "agent_exited",
    "budget_exhausted",
    "agent_failed",
    "evaluation_failed",
]


@dataclass(frozen=True, slots=True)
class Feedback:
    """One Benchmark-defined public evaluation projection."""

    score: float
    content: PolicyValue = None
    artifacts: tuple[Artifact, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("feedback score must be a finite number")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("feedback score must be finite")
        artifacts = tuple(self.artifacts)
        if any(type(artifact) is not Artifact for artifact in artifacts):
            raise TypeError("feedback artifacts must contain Artifact values")
        if len(artifacts) > FEEDBACK_MAX_ARTIFACTS:
            raise ValueError("feedback contains too many artifacts")
        if (
            sum(artifact.size for artifact in artifacts)
            > FEEDBACK_MAX_ARTIFACT_BYTES
        ):
            raise ValueError("feedback artifacts exceed the total byte limit")
        names = tuple(artifact.name for artifact in artifacts)
        if len(names) != len(set(names)):
            raise ValueError("feedback artifact names must be unique")

        object.__setattr__(self, "score", score)
        object.__setattr__(self, "content", copy_policy_value(self.content))
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """A sanitized public Episode outcome without scenario or seed identity."""

    status: EpisodeStatus
    reward: float | None
    steps: int
    failure: PolicyFailureCode | None = None

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("episode steps must be a non-negative integer")
        if self.status == "completed":
            if (
                isinstance(self.reward, bool)
                or not isinstance(self.reward, (int, float))
                or not math.isfinite(float(self.reward))
            ):
                raise ValueError("completed Episode requires a finite reward")
            if self.failure is not None:
                raise ValueError("completed Episode cannot contain a failure")
            object.__setattr__(self, "reward", float(self.reward))
            return
        if self.status != "policy_failed":
            raise ValueError("episode status is invalid")
        if self.reward is not None:
            raise ValueError("failed Episode cannot publish a reward")
        if self.failure not in {
            "exception",
            "timeout",
            "invalid_action",
            "protocol_error",
        }:
            raise ValueError("failed Episode requires a Policy failure code")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """The public result of evaluating one immutable Program."""

    benchmark_id: str
    program_digest: str
    feedback: Feedback
    episodes: tuple[EpisodeSummary, ...]

    def __post_init__(self) -> None:
        _non_empty_text(self.benchmark_id, "benchmark_id")
        _digest(self.program_digest, "program_digest")
        if type(self.feedback) is not Feedback:
            raise TypeError("feedback must be Feedback")
        episodes = tuple(self.episodes)
        if any(type(episode) is not EpisodeSummary for episode in episodes):
            raise TypeError("episodes must contain EpisodeSummary values")
        object.__setattr__(self, "episodes", episodes)


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """One Agent submission and its committed public Feedback."""

    submission_id: str
    program: Program
    episodes_used: int
    episodes_remaining: int
    feedback: Feedback
    episodes: tuple[EpisodeSummary, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_text(self.submission_id, "submission_id")
        if type(self.program) is not Program:
            raise TypeError("program must be Program")
        for name in ("episodes_used", "episodes_remaining"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.episodes_used == 0:
            raise ValueError("episodes_used must be positive")
        if type(self.feedback) is not Feedback:
            raise TypeError("feedback must be Feedback")
        episodes = tuple(self.episodes)
        if len(episodes) != self.episodes_used:
            raise ValueError("episodes must match episodes_used")
        if any(type(episode) is not EpisodeSummary for episode in episodes):
            raise TypeError("episodes must contain EpisodeSummary values")
        object.__setattr__(self, "episodes", episodes)

    @property
    def program_digest(self) -> str:
        return self.program.digest


@dataclass(frozen=True, slots=True)
class RunResult:
    """The detached public outcome of one Coding Agent development run."""

    final_program: Program | None
    final_submission_id: str | None
    submissions: tuple[SubmissionResult, ...]
    terminal_reason: RunTerminalReason

    def __post_init__(self) -> None:
        if self.final_program is not None and type(self.final_program) is not Program:
            raise TypeError("final_program must be Program or None")
        if (self.final_program is None) != (self.final_submission_id is None):
            raise ValueError("final Program and submission ID must appear together")
        if self.final_submission_id is not None:
            _non_empty_text(self.final_submission_id, "final_submission_id")
        submissions = tuple(self.submissions)
        if any(type(item) is not SubmissionResult for item in submissions):
            raise TypeError("submissions must contain SubmissionResult values")
        submission_ids = tuple(item.submission_id for item in submissions)
        if len(submission_ids) != len(set(submission_ids)):
            raise ValueError("submission IDs must be unique")
        if self.final_submission_id is not None:
            selected = tuple(
                item
                for item in submissions
                if item.submission_id == self.final_submission_id
            )
            if len(selected) != 1:
                raise ValueError("final_submission_id must select one submission")
            assert self.final_program is not None
            if selected[0].program != self.final_program:
                raise ValueError(
                    "final Program must match the selected submission"
                )
        if self.terminal_reason not in {
            "finished",
            "agent_exited",
            "budget_exhausted",
            "agent_failed",
            "evaluation_failed",
        }:
            raise ValueError("terminal_reason is invalid")
        object.__setattr__(self, "submissions", submissions)


def _non_empty_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _digest(value: object, name: str) -> str:
    digest = _non_empty_text(value, name)
    prefix = "sha256:"
    suffix = digest.removeprefix(prefix)
    if not digest.startswith(prefix) or len(suffix) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(suffix, 16)
    except ValueError:
        raise ValueError(f"{name} must be a SHA-256 digest") from None
    if suffix != suffix.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


__all__ = [
    "EpisodeStatus",
    "EpisodeSummary",
    "EvaluationResult",
    "Feedback",
    "PolicyFailureCode",
    "RunResult",
    "RunTerminalReason",
    "SubmissionResult",
]
