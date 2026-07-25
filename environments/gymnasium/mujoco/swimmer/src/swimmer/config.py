"""Typed, public Swimmer-v5 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SwimmerConfig:
    """Parameters that define one Swimmer Benchmark identity."""

    frame_skip: int = 4
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.0001
    reset_noise_scale: float = 0.1
    exclude_current_positions_from_observation: bool = True

    def __post_init__(self) -> None:
        if type(self.frame_skip) is not int:
            raise TypeError("frame_skip must be an exact integer")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        _weight(
            self.forward_reward_weight,
            name="forward_reward_weight",
        )
        _weight(self.ctrl_cost_weight, name="ctrl_cost_weight")
        if type(self.reset_noise_scale) is not float:
            raise TypeError("reset_noise_scale must be an exact float")
        if (
            not math.isfinite(self.reset_noise_scale)
            or not 0.0 <= self.reset_noise_scale <= 1.0
        ):
            raise ValueError(
                "reset_noise_scale must be finite and in [0.0, 1.0]"
            )
        if type(self.exclude_current_positions_from_observation) is not bool:
            raise TypeError(
                "exclude_current_positions_from_observation "
                "must be an exact bool"
            )


def _weight(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or not 0.0 <= value <= 1_000_000.0:
        raise ValueError(
            f"{name} must be finite and in [0.0, 1000000.0]"
        )


__all__ = ["SwimmerConfig"]
