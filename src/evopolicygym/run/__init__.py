"""Public Coding Agent development-Run configuration and entry point."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from ..agents import CodingAgent
from ..benchmark import Benchmark
from ..execution import ProcessExecution
from ..program import Program
from ..results import RunResult
from .progress import ConsoleProgress, RunEvent, RunObserver


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """Finite authority granted to one Program Evolution Run."""

    split: str = "train"
    max_submissions: int = 20
    episode_budget: int = 1_000
    max_episodes_per_submission: int = 100
    seed: int = 0
    episode_timeout_seconds: float = 30.0
    agent_timeout_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if type(self.split) is not str or not self.split:
            raise ValueError("split must be non-empty text")
        for name in (
            "max_submissions",
            "episode_budget",
            "max_episodes_per_submission",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_episodes_per_submission > self.episode_budget:
            raise ValueError(
                "max_episodes_per_submission cannot exceed episode_budget"
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
        observer=observer,
    )


__all__ = [
    "ConsoleProgress",
    "RunConfig",
    "RunEvent",
    "RunObserver",
    "RunResult",
    "run",
]
