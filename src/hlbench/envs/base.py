"""Environment backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hlbench.core.scenario import ScenarioSpec
from hlbench.core.task import EnvContract


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    max_steps: int | None = None
    backend: str = "gymnasium"


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_record(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "info": self.info,
        }


class EnvironmentInstance(Protocol):
    @property
    def action_count(self) -> int | None:
        ...

    @property
    def action_schema(self) -> dict[str, Any]:
        ...

    def reset(self, seed: int, config: dict[str, Any] | None = None) -> Any:
        ...

    def step(self, action: Any) -> StepResult:
        ...

    def close(self) -> None:
        ...


class EnvironmentBackend(Protocol):
    name: str

    def describe(self, scenario: ScenarioSpec) -> EnvContract:
        ...

    def make(self, scenario: ScenarioSpec) -> EnvironmentInstance:
        ...


EnvBackend = EnvironmentInstance
