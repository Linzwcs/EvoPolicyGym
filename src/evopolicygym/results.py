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
    "validation_failed",
    "assessment_failed",
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
    environment_digest: str
    program_digest: str
    feedback: Feedback
    episodes: tuple[EpisodeSummary, ...]

    def __post_init__(self) -> None:
        _non_empty_text(self.benchmark_id, "benchmark_id")
        _digest(self.environment_digest, "environment_digest")
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
    episode_indices: tuple[int, ...]
    episodes_used: int
    episodes_remaining: int
    feedback: Feedback
    episodes: tuple[EpisodeSummary, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_text(self.submission_id, "submission_id")
        if type(self.program) is not Program:
            raise TypeError("program must be Program")
        episode_indices = tuple(self.episode_indices)
        if any(
            type(index) is not int or not 0 <= index <= 2**64 - 1
            for index in episode_indices
        ):
            raise ValueError(
                "episode_indices must contain unsigned 64-bit integers"
            )
        if any(
            previous >= current
            for previous, current in zip(
                episode_indices,
                episode_indices[1:],
                strict=False,
            )
        ):
            raise ValueError("episode_indices must be strictly increasing")
        for name in ("episodes_used", "episodes_remaining"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.episodes_used == 0:
            raise ValueError("episodes_used must be positive")
        if len(episode_indices) != self.episodes_used:
            raise ValueError(
                "episode_indices must match episodes_used"
            )
        if type(self.feedback) is not Feedback:
            raise TypeError("feedback must be Feedback")
        episodes = tuple(self.episodes)
        if len(episodes) != self.episodes_used:
            raise ValueError("episodes must match episodes_used")
        if any(type(episode) is not EpisodeSummary for episode in episodes):
            raise TypeError("episodes must contain EpisodeSummary values")
        object.__setattr__(self, "episode_indices", episode_indices)
        object.__setattr__(self, "episodes", episodes)

    @property
    def program_digest(self) -> str:
        return self.program.digest


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    """Aggregate held-out evidence for the selected final Program."""

    submission_id: str
    program_digest: str
    split: str
    episodes: int
    primary_metric: str
    score_direction: Literal["maximize", "minimize"]
    score: float
    policy_failures: int

    def __post_init__(self) -> None:
        _non_empty_text(self.submission_id, "submission_id")
        _digest(self.program_digest, "program_digest")
        _non_empty_text(self.split, "split")
        _non_empty_text(self.primary_metric, "primary_metric")
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if self.score_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "score_direction must be 'maximize' or 'minimize'"
            )
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("score must be a finite number")
        if (
            type(self.policy_failures) is not int
            or not 0 <= self.policy_failures <= self.episodes
        ):
            raise ValueError(
                "policy_failures must be between zero and episodes"
            )
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class ValidationCandidateResult:
    """One aggregate server-side Validation outcome."""

    submission_id: str
    program_digest: str
    score: float
    episodes: int
    policy_failures: int

    def __post_init__(self) -> None:
        _non_empty_text(self.submission_id, "submission_id")
        _digest(self.program_digest, "program_digest")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("score must be a finite number")
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if (
            type(self.policy_failures) is not int
            or not 0 <= self.policy_failures <= self.episodes
        ):
            raise ValueError(
                "policy_failures must be between zero and episodes"
            )
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregate server-side evidence and deterministic final selection."""

    split: str
    episodes_per_candidate: int
    primary_metric: str
    score_direction: Literal["maximize", "minimize"]
    candidates: tuple[ValidationCandidateResult, ...]
    selected_submission_id: str

    def __post_init__(self) -> None:
        _non_empty_text(self.split, "split")
        _non_empty_text(self.primary_metric, "primary_metric")
        if (
            type(self.episodes_per_candidate) is not int
            or self.episodes_per_candidate <= 0
        ):
            raise ValueError(
                "episodes_per_candidate must be a positive integer"
            )
        if self.score_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "score_direction must be 'maximize' or 'minimize'"
            )
        candidates = tuple(self.candidates)
        if not candidates or any(
            type(candidate) is not ValidationCandidateResult
            for candidate in candidates
        ):
            raise TypeError(
                "candidates must contain ValidationCandidateResult values"
            )
        identifiers = tuple(
            candidate.submission_id for candidate in candidates
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("validation candidate IDs must be unique")
        if self.selected_submission_id not in identifiers:
            raise ValueError(
                "selected_submission_id must select a validation candidate"
            )
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class RunResult:
    """The detached public outcome of one Coding Agent development run."""

    final_program: Program | None
    final_submission_id: str | None
    submissions: tuple[SubmissionResult, ...]
    terminal_reason: RunTerminalReason
    candidate_submission_ids: tuple[str, ...] = ()
    validation: ValidationResult | None = None
    assessment: AssessmentResult | None = None

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
        candidates = tuple(self.candidate_submission_ids)
        if any(type(item) is not str or not item for item in candidates):
            raise ValueError(
                "candidate_submission_ids must contain non-empty text"
            )
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate submission IDs must be unique")
        if any(item not in submission_ids for item in candidates):
            raise ValueError(
                "candidate submission IDs must select submissions"
            )
        if (
            self.final_submission_id is not None
            and self.final_submission_id not in candidates
        ):
            raise ValueError("final submission must be a finished candidate")
        if self.validation is not None:
            if type(self.validation) is not ValidationResult:
                raise TypeError("validation must be ValidationResult or None")
            validation_ids = tuple(
                item.submission_id for item in self.validation.candidates
            )
            if validation_ids != candidates:
                raise ValueError(
                    "validation candidates must match finished candidates"
                )
            if self.final_submission_id != self.validation.selected_submission_id:
                raise ValueError(
                    "final submission must match validation selection"
                )
        final_reasons = {"finished", "assessment_failed"}
        if self.terminal_reason in final_reasons:
            if self.final_submission_id is None or not candidates:
                raise ValueError(
                    "a post-selection Run requires a final candidate"
                )
        elif self.final_submission_id is not None:
            raise ValueError(
                "only a post-selection Run can contain a final Program"
            )
        if (
            self.validation is not None
            and self.terminal_reason not in final_reasons
        ):
            raise ValueError(
                "Validation results require completed candidate selection"
            )
        if self.assessment is not None:
            if type(self.assessment) is not AssessmentResult:
                raise TypeError("assessment must be AssessmentResult or None")
            if self.terminal_reason != "finished":
                raise ValueError(
                    "Assessment results require a finished Run"
                )
            if (
                self.assessment.submission_id
                != self.final_submission_id
            ):
                raise ValueError(
                    "Assessment must match the final submission"
                )
            assert self.final_program is not None
            if self.assessment.program_digest != self.final_program.digest:
                raise ValueError(
                    "Assessment must match the final Program"
                )
        if (
            candidates
            and self.terminal_reason
            not in {"finished", "validation_failed", "assessment_failed"}
        ):
            raise ValueError(
                "only post-finish phases can contain candidates"
            )
        if self.terminal_reason not in {
            "finished",
            "agent_exited",
            "budget_exhausted",
            "agent_failed",
            "evaluation_failed",
            "validation_failed",
            "assessment_failed",
        }:
            raise ValueError("terminal_reason is invalid")
        object.__setattr__(self, "submissions", submissions)
        object.__setattr__(self, "candidate_submission_ids", candidates)


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
    "AssessmentResult",
    "EpisodeStatus",
    "EpisodeSummary",
    "EvaluationResult",
    "Feedback",
    "PolicyFailureCode",
    "RunResult",
    "RunTerminalReason",
    "SubmissionResult",
    "ValidationCandidateResult",
    "ValidationResult",
]
