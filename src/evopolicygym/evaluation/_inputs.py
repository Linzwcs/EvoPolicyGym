"""Trusted exact Episode inputs shared by Evaluation and Run rules."""

from __future__ import annotations

from dataclasses import dataclass

from ..authoring import EpisodeSpec


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    """One exact Evaluation input and its Policy random seed."""

    spec: EpisodeSpec
    policy_seed: int

    def __post_init__(self) -> None:
        if type(self.spec) is not EpisodeSpec:
            raise TypeError("spec must be EpisodeSpec")
        if (
            type(self.policy_seed) is not int
            or not 0 <= self.policy_seed <= 2**64 - 1
        ):
            raise ValueError("policy_seed must be an unsigned 64-bit integer")


__all__: list[str] = []
