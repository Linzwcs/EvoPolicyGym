"""Public Coding Agent development-Run configuration and entry point."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..agents import CodingAgent
from ..benchmark import Benchmark
from ..execution import ProcessExecution
from ..program import Program
from ..results import RunResult
from ..skills import AgentSkill
from .progress import ConsoleProgress, RunEvent, RunObserver

_MAX_AGENT_SKILLS = 16
_MAX_AGENT_SKILL_FILES = 2_048
_MAX_AGENT_SKILL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessmentConfig:
    """Fixed held-out evidence for the selected final Program."""

    episodes: int
    split: str = "test"

    def __post_init__(self) -> None:
        if type(self.split) is not str or not self.split:
            raise ValueError("assessment split must be non-empty text")
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("assessment episodes must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationConfig:
    """Fixed server-side evidence used to select finished candidates."""

    episodes_per_candidate: int
    split: str = "validation"
    max_candidates: int = 3

    def __post_init__(self) -> None:
        if type(self.split) is not str or not self.split:
            raise ValueError("validation split must be non-empty text")
        for name in ("episodes_per_candidate", "max_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """Finite authority granted to one Program Evolution Run."""

    split: str = "train"
    max_submissions: int = 20
    episode_budget: int = 1_000
    episode_pool_size: int | None = None
    max_episodes_per_submission: int | None = None
    validation: ValidationConfig | None = None
    assessment: AssessmentConfig | None = None
    seed: int = 0
    episode_timeout_seconds: float = 30.0
    agent_timeout_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if type(self.split) is not str or not self.split:
            raise ValueError("split must be non-empty text")
        for name in (
            "max_submissions",
            "episode_budget",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        pool_size = self.episode_pool_size
        if pool_size is None:
            pool_size = self.episode_budget
            object.__setattr__(self, "episode_pool_size", pool_size)
        elif type(pool_size) is not int or pool_size <= 0:
            raise ValueError(
                "episode_pool_size must be a positive integer or None"
            )
        submission_limit = self.max_episodes_per_submission
        if submission_limit is not None:
            if type(submission_limit) is not int or submission_limit <= 0:
                raise ValueError(
                    "max_episodes_per_submission must be a positive integer or None"
                )
            if submission_limit > self.episode_budget:
                raise ValueError(
                    "max_episodes_per_submission cannot exceed episode_budget"
                )
            if submission_limit > pool_size:
                raise ValueError(
                    "max_episodes_per_submission cannot exceed "
                    "episode_pool_size"
                )
        if self.validation is not None:
            if type(self.validation) is not ValidationConfig:
                raise TypeError("validation must be ValidationConfig or None")
            if self.validation.max_candidates > self.max_submissions:
                raise ValueError(
                    "validation max_candidates cannot exceed max_submissions"
                )
        if self.assessment is not None:
            if type(self.assessment) is not AssessmentConfig:
                raise TypeError(
                    "assessment must be AssessmentConfig or None"
                )
        if type(self.seed) is not int or not 0 <= self.seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        for name in ("episode_timeout_seconds", "agent_timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, float(value))


def run(
    initial_program: Program,
    benchmark: Benchmark,
    *,
    agent: CodingAgent,
    execution: ProcessExecution,
    record_to: str | os.PathLike[str],
    config: RunConfig | None = None,
    skills: Sequence[AgentSkill] = (),
    observer: RunObserver | None = None,
) -> RunResult:
    """Let one Coding Agent improve a Program through a bounded local Session.

    ``ProcessExecution`` is not a sandbox. The Agent and submitted Policy code
    run with the authority of the current operating-system user.
    """

    if type(initial_program) is not Program:
        raise TypeError("initial_program must be Program")
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must implement Benchmark")
    if not isinstance(agent, CodingAgent):
        raise TypeError("agent must implement CodingAgent")
    if type(execution) is not ProcessExecution:
        raise TypeError("execution must be ProcessExecution.unsafe()")
    selected_config = RunConfig() if config is None else config
    if type(selected_config) is not RunConfig:
        raise TypeError("config must be RunConfig or None")
    selected_skills = _select_skills(skills)
    if observer is not None and not isinstance(observer, RunObserver):
        raise TypeError("observer must implement RunObserver or be None")
    try:
        run_directory = Path(os.fspath(record_to))
    except TypeError:
        raise TypeError("record_to must be a path-like string") from None

    from ._service import run_agent_with_processes

    return run_agent_with_processes(
        initial_program,
        benchmark,
        agent=agent,
        run_directory=run_directory,
        config=selected_config,
        skills=selected_skills,
        observer=observer,
    )


def _select_skills(
    skills: Sequence[AgentSkill],
) -> tuple[AgentSkill, ...]:
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError("skills must be a sequence of AgentSkill values")
    selected = tuple(skills)
    if len(selected) > _MAX_AGENT_SKILLS:
        raise ValueError(
            f"a Run can include at most {_MAX_AGENT_SKILLS} Agent Skills"
        )
    if any(type(skill) is not AgentSkill for skill in selected):
        raise TypeError("skills must contain only AgentSkill values")
    names = tuple(skill.name for skill in selected)
    if len(set(names)) != len(names):
        raise ValueError("skills must have unique names")
    if sum(skill.file_count for skill in selected) > _MAX_AGENT_SKILL_FILES:
        raise ValueError("Agent Skills contain too many files for one Run")
    if sum(skill.total_bytes for skill in selected) > _MAX_AGENT_SKILL_BYTES:
        raise ValueError("Agent Skills exceed the total Run byte limit")
    return selected


__all__ = [
    "AssessmentConfig",
    "ConsoleProgress",
    "RunConfig",
    "RunEvent",
    "RunObserver",
    "RunResult",
    "ValidationConfig",
    "run",
]
