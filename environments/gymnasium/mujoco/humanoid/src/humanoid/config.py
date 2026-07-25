"""Typed, public Humanoid-v5 environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HumanoidConfig:
    """Parameters that define one Humanoid Benchmark identity."""

    frame_skip: int = 5
    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    contact_cost_weight: float = 0.0000005
    contact_cost_range: tuple[float | None, float] = (None, 10.0)
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (1.0, 2.0)
    reset_noise_scale: float = 0.01
    exclude_current_positions_from_observation: bool = True
    include_cinert_in_observation: bool = True
    include_cvel_in_observation: bool = True
    include_qfrc_actuator_in_observation: bool = True
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
        _clamp_range(self.contact_cost_range, name="contact_cost_range")
        if type(self.terminate_when_unhealthy) is not bool:
            raise TypeError("terminate_when_unhealthy must be an exact bool")
        _finite_range(self.healthy_z_range, name="healthy_z_range")
        if type(self.reset_noise_scale) is not float:
            raise TypeError("reset_noise_scale must be an exact float")
        if (
            not math.isfinite(self.reset_noise_scale)
            or not 0.0 <= self.reset_noise_scale <= 1.0
        ):
            raise ValueError(
                "reset_noise_scale must be finite and in [0.0, 1.0]"
            )
        for name in (
            "exclude_current_positions_from_observation",
            "include_cinert_in_observation",
            "include_cvel_in_observation",
            "include_qfrc_actuator_in_observation",
            "include_cfrc_ext_in_observation",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")


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


def _clamp_range(
    value: tuple[float | None, float],
    *,
    name: str,
) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(
            f"{name} must be an exact (float | None, float) tuple"
        )
    lower, upper = value
    if lower is not None and type(lower) is not float:
        raise TypeError(f"{name} lower bound must be a float or None")
    if type(upper) is not float:
        raise TypeError(f"{name} upper bound must be an exact float")
    if lower is not None and not math.isfinite(lower):
        raise ValueError(f"{name} lower bound must be finite")
    if not math.isfinite(upper):
        raise ValueError(f"{name} upper bound must be finite")
    if lower is not None and lower >= upper:
        raise ValueError(f"{name} bounds must be increasing")


__all__ = ["HumanoidConfig"]
