"""Typed, public LunarLander-v3 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LunarLanderConfig:
    """Parameters that define one LunarLander Benchmark identity."""

    continuous: bool = False
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5

    def __post_init__(self) -> None:
        if type(self.continuous) is not bool:
            raise TypeError("continuous must be an exact bool")
        if type(self.enable_wind) is not bool:
            raise TypeError("enable_wind must be an exact bool")
        _finite_float(self.gravity, name="gravity")
        if not -12.0 < self.gravity < 0.0:
            raise ValueError("gravity must be strictly between -12.0 and 0.0")
        _finite_float(self.wind_power, name="wind_power")
        _finite_float(self.turbulence_power, name="turbulence_power")


def _finite_float(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = ["LunarLanderConfig"]
