"""Typed, public Ant-v5 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AntConfig:
    """Parameters that define one Ant Benchmark identity."""

    frame_skip: int = 5
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.5
    contact_cost_weight: float = 0.0005
    healthy_reward: float = 1.0
    main_body: int = 1
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (0.2, 1.0)
    contact_force_range: tuple[float, float] = (-1.0, 1.0)
    reset_noise_scale: float = 0.1
    exclude_current_positions_from_observation: bool = True
    include_cfrc_ext_in_observation: bool = True

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
        _weight(self.contact_cost_weight, name="contact_cost_weight")
        _weight(self.healthy_reward, name="healthy_reward")
        if type(self.main_body) is not int:
            raise TypeError("main_body must be an exact integer")
        if not 1 <= self.main_body <= 13:
            raise ValueError("main_body must be in [1, 13]")
        if type(self.terminate_when_unhealthy) is not bool:
            raise TypeError("terminate_when_unhealthy must be an exact bool")
        _finite_range(self.healthy_z_range, name="healthy_z_range")
        _finite_range(
            self.contact_force_range,
            name="contact_force_range",
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
        if type(self.exclude_current_positions_from_observation) is not bool:
            raise TypeError(
                "exclude_current_positions_from_observation "
                "must be an exact bool"
            )
        if type(self.include_cfrc_ext_in_observation) is not bool:
            raise TypeError(
                "include_cfrc_ext_in_observation must be an exact bool"
            )


def _weight(value: float, *, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or not 0.0 <= value <= 1_000_000.0:
        raise ValueError(
            f"{name} must be finite and in [0.0, 1000000.0]"
        )


def _finite_range(value: tuple[float, float], *, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{name} must be an exact two-float tuple")
    lower, upper = value
    if type(lower) is not float or type(upper) is not float:
        raise TypeError(f"{name} must be an exact two-float tuple")
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
    ):
        raise ValueError(f"{name} must contain finite increasing bounds")


__all__ = ["AntConfig"]
