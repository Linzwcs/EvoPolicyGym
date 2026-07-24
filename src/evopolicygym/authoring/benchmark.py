"""Structural contract implemented by external Benchmark distributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from ..policy import PolicyValue, copy_policy_value
from ..results import Feedback
from .environment import Environment, EpisodeRecord, EpisodeSpec

type ScoreDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Static, public, execution-independent Benchmark metadata."""

    id: str
    description: str
    observation_space: PolicyValue
    action_space: PolicyValue
    metadata: Mapping[str, PolicyValue]
    max_episode_steps: int
    primary_metric: str
    score_direction: ScoreDirection

    def __post_init__(self) -> None:
        for name in ("id", "description", "primary_metric"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        if type(self.max_episode_steps) is not int or self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        if self.score_direction not in {"maximize", "minimize"}:
            raise ValueError("score_direction must be 'maximize' or 'minimize'")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        metadata: dict[str, PolicyValue] = {}
        for key, value in self.metadata.items():
            if type(key) is not str:
                raise TypeError("metadata keys must be exact strings")
            metadata[key] = copy_policy_value(value)
        object.__setattr__(
            self,
            "observation_space",
            copy_policy_value(self.observation_space),
        )
        object.__setattr__(self, "action_space", copy_policy_value(self.action_space))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@runtime_checkable
class Benchmark(Protocol):
    """The stable structural interface implemented by external Benchmarks."""

    @property
    def spec(self) -> BenchmarkSpec:
        """Return static public metadata without opening an Environment."""
        ...

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        """Deterministically plan exactly ``count`` trusted Episodes."""
        ...

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        """Create one fresh Environment for one Episode."""
        ...

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        """Project trusted Episode evidence into Benchmark-defined Feedback."""
        ...


__all__ = ["Benchmark", "BenchmarkSpec", "ScoreDirection"]
