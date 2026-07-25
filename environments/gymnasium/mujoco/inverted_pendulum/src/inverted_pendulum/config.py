"""Typed, public InvertedPendulum-v5 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvertedPendulumConfig:
    """Parameters that define one InvertedPendulum Benchmark identity."""

    frame_skip: int = 2
    reset_noise_scale: float = 0.01

    def __post_init__(self) -> None:
        if type(self.frame_skip) is not int:
            raise TypeError("frame_skip must be an exact integer")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if type(self.reset_noise_scale) is not float:
            raise TypeError("reset_noise_scale must be an exact float")
        if (
            not math.isfinite(self.reset_noise_scale)
            or not 0.0 <= self.reset_noise_scale <= 1.0
        ):
            raise ValueError(
                "reset_noise_scale must be finite and in [0.0, 1.0]"
            )


__all__ = ["InvertedPendulumConfig"]
