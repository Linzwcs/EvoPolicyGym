"""Public direct-Evaluation configuration and entry point."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..benchmark import Benchmark
from ..execution import ProcessExecution
from ..program import Program
from ..results import EvaluationResult


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationConfig:
    """Finite deterministic input for one direct Evaluation."""

    split: str = "validation"
    episodes: int = 1
    seed: int = 0
    episode_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if type(self.split) is not str or not self.split:
            raise ValueError("split must be non-empty text")
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        timeout = self.episode_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("episode_timeout_seconds must be positive and finite")
        object.__setattr__(self, "episode_timeout_seconds", float(timeout))


def evaluate(
    program: Program,
    benchmark: Benchmark,
    *,
    execution: ProcessExecution,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Evaluate one immutable Program using explicitly unsafe local processes.

    ``ProcessExecution`` is not a sandbox. The submitted Program runs with the
    authority of the current operating-system user.
    """

    if type(program) is not Program:
        raise TypeError("program must be Program")
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must implement Benchmark")
    if type(execution) is not ProcessExecution:
        raise TypeError("execution must be ProcessExecution.unsafe()")
    selected_config = EvaluationConfig() if config is None else config
    if type(selected_config) is not EvaluationConfig:
        raise TypeError("config must be EvaluationConfig or None")

    from ..execution.process.policy.runtime import ProcessPolicyRuntimeFactory
    from ._service import EvaluationService

    service = EvaluationService(
        policy_runtimes=ProcessPolicyRuntimeFactory(),
        monotonic=time.monotonic,
    )
    return service.evaluate(program, benchmark, selected_config)


__all__ = ["EvaluationConfig", "EvaluationResult", "evaluate"]
