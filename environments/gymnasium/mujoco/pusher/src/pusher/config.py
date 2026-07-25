"""Typed, public Pusher-v5 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PusherConfig:
    """Parameters that define one Pusher Benchmark identity."""

    frame_skip: int = 5
    reward_near_weight: float = 0.5
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 0.1

    def __post_init__(self) -> None:
        if type(self.frame_skip) is not int:
            raise TypeError("frame_skip must be an exact integer")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        _weight(self.reward_near_weight, name="reward_near_weight")
        _weight(self.reward_dist_weight, name="reward_dist_weight")
        _weight(self.reward_control_weight, name="reward_control_weight")


def _weight(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or not 0.0 <= value <= 1_000_000.0:
        raise ValueError(
            f"{name} must be finite and in [0.0, 1000000.0]"
        )


__all__ = ["PusherConfig"]
