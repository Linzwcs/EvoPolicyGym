"""Direct Program Evaluation rules and their narrow runtime contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Protocol

from ..authoring import (
    Benchmark,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    Transition,
)
from ..errors import EvaluationError
from ..policy import PolicyContext, PolicyValue, copy_policy_value
from ..program import Program
from ..results import (
    EpisodeSummary,
    EvaluationResult,
    Feedback,
    PolicyFailureCode,
)
from . import EvaluationConfig

_POLICY_SEED_DOMAIN = b"evopolicygym/policy-seed/v1\0"


class PolicyRuntimeError(Exception):
    """The Host could not reliably control the Policy runtime."""


class PolicyRuntimeCleanupError(PolicyRuntimeError):
    """The Policy runtime could not be reaped or cleaned."""


class PolicyRuntime(Protocol):
    def start(self, *, timeout_seconds: float) -> PolicyFailureCode | None:
        ...

    def act(
        self,
        observation: PolicyValue,
        *,
        timeout_seconds: float,
    ) -> tuple[PolicyValue | None, PolicyFailureCode | None]:
        ...

    def close(self) -> None:
        ...


class PolicyRuntimeFactory(Protocol):
    def create(
        self,
        program: Program,
        context: PolicyContext,
    ) -> PolicyRuntime:
        ...


class EvaluationService:
    """Evaluate Programs without selecting a concrete Policy runtime."""

    def __init__(
        self,
        *,
        policy_runtimes: PolicyRuntimeFactory,
        monotonic: Callable[[], float],
    ) -> None:
        self._policy_runtimes = policy_runtimes
        self._monotonic = monotonic

    def evaluate(
        self,
        program: Program,
        benchmark: Benchmark,
        config: EvaluationConfig,
        *,
        episode_completed: (
            Callable[[int, int, EpisodeSummary], None] | None
        ) = None,
    ) -> EvaluationResult:
        try:
            spec = benchmark.spec
            planned = tuple(
                benchmark.episodes(
                    config.split,
                    seed=config.seed,
                    count=config.episodes,
                )
            )
        except Exception:
            raise EvaluationError("Benchmark could not plan Episodes") from None
        if len(planned) != config.episodes:
            raise EvaluationError("Benchmark returned the wrong Episode count")
        if any(type(episode) is not EpisodeSpec for episode in planned):
            raise EvaluationError("Benchmark returned an invalid Episode plan")

        records: list[EpisodeRecord] = []
        for index, episode in enumerate(planned):
            record = self._evaluate_episode(
                program,
                benchmark,
                episode,
                spec=spec,
                policy_seed=_derive_policy_seed(config.seed, index),
                max_steps=spec.max_episode_steps,
                timeout_seconds=config.episode_timeout_seconds,
            )
            records.append(record)
            if episode_completed is not None:
                episode_completed(
                    index + 1,
                    len(planned),
                    _public_summary(record),
                )

        try:
            feedback = benchmark.feedback(tuple(records))
        except Exception:
            raise EvaluationError("Benchmark could not produce Feedback") from None
        if type(feedback) is not Feedback:
            raise EvaluationError("Benchmark feedback() returned an invalid value")

        summaries = tuple(_public_summary(record) for record in records)
        return EvaluationResult(
            benchmark_id=spec.id,
            program_digest=program.digest,
            feedback=feedback,
            episodes=summaries,
        )

    def _evaluate_episode(
        self,
        program: Program,
        benchmark: Benchmark,
        episode: EpisodeSpec,
        *,
        spec: BenchmarkSpec,
        policy_seed: int,
        max_steps: int,
        timeout_seconds: float,
    ) -> EpisodeRecord:
        environment: Environment | None = None
        policy: PolicyRuntime | None = None
        record: EpisodeRecord | None = None
        trusted_fault: str | None = None
        cleanup_fault = False
        deadline = self._monotonic() + timeout_seconds

        try:
            try:
                candidate = benchmark.make_environment(episode)
            except Exception:
                trusted_fault = "Benchmark could not create an Environment"
                candidate = None
            if trusted_fault is None:
                if not isinstance(candidate, Environment):
                    trusted_fault = "Benchmark returned an invalid Environment"
                else:
                    environment = candidate

            initial: PolicyValue = None
            if trusted_fault is None:
                assert environment is not None
                try:
                    initial = copy_policy_value(environment.reset())
                except Exception:
                    trusted_fault = "Environment reset failed"

            if trusted_fault is None:
                context = PolicyContext(
                    observation_space=spec.observation_space,
                    action_space=spec.action_space,
                    metadata=spec.metadata,
                    policy_seed=policy_seed,
                )
                policy = self._policy_runtimes.create(program, context)
                try:
                    startup_failure = policy.start(
                        timeout_seconds=self._remaining(deadline),
                    )
                except Exception:
                    trusted_fault = "Policy process could not be controlled"
                    startup_failure = None
                if trusted_fault is None and startup_failure is not None:
                    record = EpisodeRecord(
                        episode=episode,
                        policy_seed=policy_seed,
                        initial_observation=initial,
                        transitions=(),
                        policy_failure=startup_failure,
                    )

            if trusted_fault is None and record is None:
                assert environment is not None
                assert policy is not None
                observation = initial
                transitions: list[Transition] = []
                policy_failure = None
                for _ in range(max_steps):
                    try:
                        action, policy_failure = policy.act(
                            observation,
                            timeout_seconds=self._remaining(deadline),
                        )
                    except (TypeError, ValueError, RecursionError):
                        trusted_fault = "Environment observation is invalid"
                        break
                    except PolicyRuntimeError:
                        trusted_fault = "Policy process could not be controlled"
                        break
                    if policy_failure is not None:
                        break
                    try:
                        step = environment.step(action)
                    except InvalidAction:
                        policy_failure = "invalid_action"
                        break
                    except Exception:
                        trusted_fault = "Environment step failed"
                        break
                    if type(step) is not Step:
                        trusted_fault = "Environment step returned an invalid value"
                        break
                    transitions.append(Transition(action=action, step=step))
                    observation = step.observation
                    if step.done:
                        break
                else:
                    trusted_fault = "Environment exceeded max_episode_steps"

                if trusted_fault is None:
                    if policy_failure is None and (
                        not transitions or not transitions[-1].step.done
                    ):
                        trusted_fault = "Environment did not terminate the Episode"
                    else:
                        record = EpisodeRecord(
                            episode=episode,
                            policy_seed=policy_seed,
                            initial_observation=initial,
                            transitions=tuple(transitions),
                            policy_failure=policy_failure,
                        )
        finally:
            if policy is not None:
                try:
                    policy.close()
                except PolicyRuntimeCleanupError:
                    cleanup_fault = True
            if environment is not None:
                try:
                    environment.close()
                except Exception:
                    cleanup_fault = True

        if cleanup_fault:
            raise EvaluationError("Episode cleanup failed")
        if trusted_fault is not None:
            raise EvaluationError(trusted_fault)
        assert record is not None
        return record

    def _remaining(self, deadline: float) -> float:
        return max(deadline - self._monotonic(), 0.0)


def _derive_policy_seed(master_seed: int, episode_index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_POLICY_SEED_DOMAIN)
    digest.update(master_seed.to_bytes(8, "big"))
    digest.update(episode_index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _public_summary(record: EpisodeRecord) -> EpisodeSummary:
    if record.policy_failure is not None:
        return EpisodeSummary(
            status="policy_failed",
            reward=None,
            steps=record.steps,
            failure=record.policy_failure,
        )
    return EpisodeSummary(
        status="completed",
        reward=record.total_reward,
        steps=record.steps,
    )


__all__: list[str] = []
