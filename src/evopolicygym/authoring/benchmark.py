"""Structural contract implemented by external Benchmark distributions."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from ..policy import PolicyValue, TensorValue, copy_policy_value
from ..results import Feedback
from .environment import Environment, EpisodeRecord, EpisodeSpec

type ScoreDirection = Literal["maximize", "minimize"]

_ENVIRONMENT_DIGEST_DOMAIN = b"evopolicygym/environment-parameters/v1\0"


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> object:
        ...


class _EnvironmentParameters(Mapping[str, PolicyValue]):
    """A detached mapping whose mutable nested carriers never escape."""

    __slots__ = ("_digest", "_values")

    def __init__(self, values: Mapping[str, PolicyValue]) -> None:
        copied = _copy_mapping(values, name="environment_parameters")
        self._values = MappingProxyType(copied)
        self._digest = _environment_digest(copied)

    @property
    def digest(self) -> str:
        return self._digest

    def __getitem__(self, key: str) -> PolicyValue:
        return copy_policy_value(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(dict(self.items()))


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
    environment_parameters: Mapping[str, PolicyValue] = field(
        default_factory=dict,
    )
    agent_skill: str | None = None
    _environment_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("id", "description", "primary_metric"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        if type(self.max_episode_steps) is not int or self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        if self.score_direction not in {"maximize", "minimize"}:
            raise ValueError("score_direction must be 'maximize' or 'minimize'")
        if self.agent_skill is not None:
            if type(self.agent_skill) is not str:
                raise TypeError("agent_skill must be text or None")
            if (
                not self.agent_skill
                or len(self.agent_skill.encode("utf-8", errors="strict"))
                > 64 * 1024
                or "\0" in self.agent_skill
            ):
                raise ValueError("agent_skill must be non-empty bounded text")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not isinstance(self.environment_parameters, Mapping):
            raise TypeError("environment_parameters must be a mapping")

        metadata = _copy_mapping(self.metadata, name="metadata")
        environment_parameters = _EnvironmentParameters(
            self.environment_parameters
        )
        object.__setattr__(
            self,
            "observation_space",
            copy_policy_value(self.observation_space),
        )
        object.__setattr__(self, "action_space", copy_policy_value(self.action_space))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(
            self,
            "environment_parameters",
            environment_parameters,
        )
        object.__setattr__(
            self,
            "_environment_digest",
            environment_parameters.digest,
        )

    @property
    def environment_digest(self) -> str:
        """Return the canonical identity of the public Environment parameters."""

        return self._environment_digest


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


def _copy_mapping(
    value: Mapping[str, PolicyValue],
    *,
    name: str,
) -> dict[str, PolicyValue]:
    copied: dict[str, PolicyValue] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError(f"{name} keys must be exact strings")
        copied[key] = copy_policy_value(item)
    return copied


def _environment_digest(parameters: dict[str, PolicyValue]) -> str:
    digest = hashlib.sha256()
    digest.update(_ENVIRONMENT_DIGEST_DOMAIN)
    _update_policy_value(digest, parameters)
    return f"sha256:{digest.hexdigest()}"


def _update_policy_value(digest: _Hasher, value: PolicyValue) -> None:
    if value is None:
        digest.update(b"N")
        return
    if type(value) is bool:
        digest.update(b"B1" if value else b"B0")
        return
    if type(value) is int:
        digest.update(b"I")
        _update_bytes(digest, str(value).encode("ascii"))
        return
    if type(value) is float:
        digest.update(b"F")
        digest.update(struct.pack("!d", value))
        return
    if type(value) is str:
        digest.update(b"S")
        _update_bytes(digest, value.encode("utf-8", errors="strict"))
        return
    if type(value) is bytes:
        digest.update(b"Y")
        _update_bytes(digest, value)
        return
    if type(value) is TensorValue:
        digest.update(b"T")
        _update_bytes(digest, value.dtype.encode("ascii"))
        digest.update(len(value.shape).to_bytes(8, "big"))
        for size in value.shape:
            _update_bytes(digest, str(size).encode("ascii"))
        _update_bytes(digest, value.data)
        return
    if type(value) is list:
        digest.update(b"L")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_policy_value(digest, item)
        return
    if type(value) is tuple:
        digest.update(b"U")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_policy_value(digest, item)
        return
    if type(value) is dict:
        digest.update(b"M")
        digest.update(len(value).to_bytes(8, "big"))
        for key in sorted(value, key=lambda item: item.encode("utf-8")):
            _update_bytes(digest, key.encode("utf-8", errors="strict"))
            _update_policy_value(digest, value[key])
        return
    raise TypeError("unsupported Environment parameter")


def _update_bytes(digest: _Hasher, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


__all__ = ["Benchmark", "BenchmarkSpec", "ScoreDirection"]
