"""Typed, public InvertedDoublePendulum-v5 configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvertedDoublePendulumConfig:
    """Parameters defining one InvertedDoublePendulum Benchmark identity."""

    frame_skip: int = 5
    healthy_reward: float = 10.0
    reset_noise_scale: float = 0.1

    def __post_init__(self) -> None:
        if type(self.frame_skip) is not int:
            raise TypeError("frame_skip must be an exact integer")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if type(self.healthy_reward) is not float:
            raise TypeError("healthy_reward must be an exact float")
        if (
            not math.isfinite(self.healthy_reward)
            or not 0.0 <= self.healthy_reward <= 1_000_000.0
        ):
            raise ValueError(
                "healthy_reward must be finite and in [0.0, 1000000.0]"
            )
        if type(self.reset_noise_scale) is not float:
            raise TypeError("reset_noise_scale must be an exact float")
        if (
            not math.isfinite(self.reset_noise_scale)
            or not 0.0 <= self.reset_noise_scale <= 1.0
        ):
            raise ValueError(
                "reset_noise_scale must be finite and in [0.0, 1.0]"
            )


__all__ = ["InvertedDoublePendulumConfig"]
