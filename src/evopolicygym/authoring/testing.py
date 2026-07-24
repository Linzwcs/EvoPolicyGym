"""Public conformance helpers for external Benchmark packages."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import BenchmarkError
from ..policy import PolicyValue, copy_policy_value
from .benchmark import Benchmark
from .environment import Environment, EpisodeSpec, Step


@dataclass(frozen=True, slots=True)
class BenchmarkFixture:
    """One deterministic Environment replay used by conformance checks."""

    episode: EpisodeSpec
    actions: tuple[PolicyValue, ...]

    def __post_init__(self) -> None:
        if type(self.episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        object.__setattr__(
            self,
            "actions",
            tuple(copy_policy_value(action) for action in self.actions),
        )


@dataclass(frozen=True, slots=True)
class ConformanceIssue:
    """One stable checker finding."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """The complete local result of checking a Benchmark."""

    benchmark_id: str
    issues: tuple[ConformanceIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            codes = ", ".join(issue.code for issue in self.issues)
            raise BenchmarkError(f"Benchmark conformance failed: {codes}")


def check_benchmark(
    benchmark: Benchmark,
    *,
    fixtures: tuple[BenchmarkFixture, ...],
) -> ConformanceReport:
    """Replay fixtures twice and report structural or determinism failures."""

    issues: list[ConformanceIssue] = []
    if not isinstance(benchmark, Benchmark):
        return ConformanceReport(
            benchmark_id="<invalid>",
            issues=(
                ConformanceIssue(
                    code="benchmark.interface",
                    message="object does not implement the Benchmark protocol",
                ),
            ),
        )

    benchmark_id = benchmark.spec.id
    for index, fixture in enumerate(fixtures):
        first = _replay(benchmark, fixture, index=index, issues=issues)
        second = _replay(benchmark, fixture, index=index, issues=issues)
        if first is not None and second is not None and first != second:
            issues.append(
                ConformanceIssue(
                    code="environment.nondeterministic",
                    message=f"fixture {index} produced different replay values",
                )
            )
    return ConformanceReport(benchmark_id=benchmark_id, issues=tuple(issues))


def _replay(
    benchmark: Benchmark,
    fixture: BenchmarkFixture,
    *,
    index: int,
    issues: list[ConformanceIssue],
) -> tuple[PolicyValue, tuple[Step, ...]] | None:
    environment: Environment | None = None
    replay: tuple[PolicyValue, tuple[Step, ...]] | None = None
    try:
        environment = benchmark.make_environment(fixture.episode)
        if not isinstance(environment, Environment):
            raise TypeError("make_environment() returned an incompatible object")
        initial = copy_policy_value(environment.reset())
        steps: list[Step] = []
        done = False
        for action in fixture.actions:
            if done:
                raise RuntimeError("fixture contains an Action after termination")
            step = environment.step(copy_policy_value(action))
            if type(step) is not Step:
                raise TypeError("Environment.step() must return Step")
            steps.append(step)
            done = step.done
        replay = (initial, tuple(steps))
    except Exception:
        issues.append(
            ConformanceIssue(
                code="environment.replay",
                message=f"fixture {index} could not be replayed",
            )
        )
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                issues.append(
                    ConformanceIssue(
                        code="environment.close",
                        message=f"fixture {index} could not be closed",
                    )
                )
                replay = None
    return replay


__all__ = [
    "BenchmarkFixture",
    "ConformanceIssue",
    "ConformanceReport",
    "check_benchmark",
]
