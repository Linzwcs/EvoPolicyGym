"""Baseline Pendulum policy."""

from __future__ import annotations

from typing import Any


class Policy:
    def reset(self, task_config: dict[str, Any]) -> None:
        del task_config

    def act(self, observation: Any, context: dict[str, Any]) -> list[float]:
        del context
        cos_theta = float(observation[0])
        sin_theta = float(observation[1])
        angular_velocity = float(observation[2])
        torque = 2.0 * sin_theta - 0.5 * angular_velocity
        if cos_theta < 0.0:
            torque = 2.0 if sin_theta >= 0.0 else -2.0
        return [max(-2.0, min(2.0, torque))]
