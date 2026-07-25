"""Typed, public Taxi-v4 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaxiConfig:
    """Parameters that define one Taxi Benchmark identity."""

    is_rainy: bool = False
    fickle_passenger: bool = False
    rainy_probability: float = 0.8
    fickle_probability: float = 0.3

    def __post_init__(self) -> None:
        if type(self.is_rainy) is not bool:
            raise TypeError("is_rainy must be an exact bool")
        if type(self.fickle_passenger) is not bool:
            raise TypeError("fickle_passenger must be an exact bool")
        _probability(self.rainy_probability, name="rainy_probability")
        _probability(self.fickle_probability, name="fickle_probability")


def _probability(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0.0, 1.0]")


__all__ = ["TaxiConfig"]
