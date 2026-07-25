"""Typed, public CarRacing-v3 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CarRacingConfig:
    """Parameters that define one CarRacing Benchmark identity."""

    continuous: bool = True
    lap_complete_percent: float = 0.95
    domain_randomize: bool = False

    def __post_init__(self) -> None:
        if type(self.continuous) is not bool:
            raise TypeError("continuous must be an exact bool")
        if type(self.domain_randomize) is not bool:
            raise TypeError("domain_randomize must be an exact bool")
        if type(self.lap_complete_percent) is not float:
            raise TypeError("lap_complete_percent must be an exact float")
        if (
            not math.isfinite(self.lap_complete_percent)
            or not 0.0 < self.lap_complete_percent <= 1.0
        ):
            raise ValueError(
                "lap_complete_percent must be finite and in (0.0, 1.0]"
            )


__all__ = ["CarRacingConfig"]
