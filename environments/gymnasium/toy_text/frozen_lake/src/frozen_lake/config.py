"""Typed, public FrozenLake environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAPS: dict[str, tuple[str, ...]] = {
    "4x4": (
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ),
    "8x8": (
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
}


@dataclass(frozen=True, slots=True)
class FrozenLakeConfig:
    """Parameters that define one FrozenLake Benchmark identity."""

    map_name: str = "4x4"
    is_slippery: bool = True
    success_rate: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if type(self.map_name) is not str or self.map_name not in _MAPS:
            raise ValueError("map_name must be '4x4' or '8x8'")
        if type(self.is_slippery) is not bool:
            raise TypeError("is_slippery must be an exact bool")
        if type(self.success_rate) is not float:
            raise TypeError("success_rate must be an exact float")
        if (
            not math.isfinite(self.success_rate)
            or not 0.0 <= self.success_rate <= 1.0
        ):
            raise ValueError("success_rate must be finite and within [0.0, 1.0]")

    @property
    def layout(self) -> tuple[str, ...]:
        """Return the canonical standard map selected by this configuration."""

        return _MAPS[self.map_name]

    @property
    def environment_id(self) -> str:
        """Return the Gymnasium registration with the matching time limit."""

        return "FrozenLake-v1" if self.map_name == "4x4" else "FrozenLake8x8-v1"

    @property
    def max_episode_steps(self) -> int:
        """Return the upstream time limit for the selected standard map."""

        return 100 if self.map_name == "4x4" else 200


__all__ = ["FrozenLakeConfig"]
