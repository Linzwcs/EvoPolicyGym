"""Trusted Environment values implemented by external Benchmark packages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..policy import PolicyValue, copy_policy_value
from ..results import PolicyFailureCode


class InvalidAction(Exception):
    """A complete Policy Action is outside the Environment's action domain."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """One trusted scenario and Environment random seed."""

    environment_seed: int
    scenario: PolicyValue = None

    def __post_init__(self) -> None:
        if (
            type(self.environment_seed) is not int
            or not 0 <= self.environment_seed <= 2**64 - 1
        ):
            raise ValueError("environment_seed must be an unsigned 64-bit integer")
        object.__setattr__(self, "scenario", copy_policy_value(self.scenario))


@dataclass(frozen=True, slots=True)
class Step:
    """One trusted Environment response."""

    observation: PolicyValue
    reward: float
    terminated: bool
    truncated: bool = False
    metrics: PolicyValue = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(float(self.reward))
        ):
            raise ValueError("reward must be a finite number")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("terminated and truncated must be exact bool values")
        object.__setattr__(self, "observation", copy_policy_value(self.observation))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "metrics", copy_policy_value(self.metrics))

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class Transition:
    """One unmodified Policy Action and its trusted Environment response."""

    action: PolicyValue
    step: Step

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", copy_policy_value(self.action))
        if type(self.step) is not Step:
            raise TypeError("step must be Step")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Trusted evidence collected from one Episode."""

    episode: EpisodeSpec
    policy_seed: int
    initial_observation: PolicyValue
    transitions: tuple[Transition, ...]
    policy_failure: PolicyFailureCode | None = None

    def __post_init__(self) -> None:
        if type(self.episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(self.policy_seed) is not int or not 0 <= self.policy_seed <= 2**64 - 1:
            raise ValueError("policy_seed must be an unsigned 64-bit integer")
        transitions = tuple(self.transitions)
        if any(type(transition) is not Transition for transition in transitions):
            raise TypeError("transitions must contain Transition values")
        if any(transition.step.done for transition in transitions[:-1]):
            raise ValueError("EpisodeRecord cannot contain transitions after termination")
        if self.policy_failure is not None and self.policy_failure not in {
            "exception",
            "timeout",
            "invalid_action",
            "protocol_error",
        }:
            raise ValueError("policy_failure is invalid")
        if transitions and transitions[-1].step.done and self.policy_failure is not None:
            raise ValueError("a terminated Episode cannot also contain a Policy failure")
        object.__setattr__(
            self,
            "initial_observation",
            copy_policy_value(self.initial_observation),
        )
        object.__setattr__(self, "transitions", transitions)

    @property
    def total_reward(self) -> float:
        return sum(transition.step.reward for transition in self.transitions)

    @property
    def steps(self) -> int:
        return len(self.transitions)


@runtime_checkable
class Environment(Protocol):
    """A fresh Benchmark-owned Environment for one Episode."""

    def reset(self) -> PolicyValue:
        """Start the Episode and return its first observation."""
        ...

    def step(self, action: PolicyValue) -> Step:
        """Apply one unmodified Policy Action."""
        ...

    def close(self) -> None:
        """Release resources; the evaluator calls this on every exit path."""
        ...


__all__ = [
    "Environment",
    "EpisodeRecord",
    "EpisodeSpec",
    "InvalidAction",
    "Step",
    "Transition",
]
