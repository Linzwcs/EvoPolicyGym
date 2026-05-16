"""Baseline CartPole policy."""

from __future__ import annotations

from typing import Any


class Policy:
    def reset(self, task_config: dict[str, Any]) -> None:
        del task_config

    def act(self, observation: Any, context: dict[str, Any]) -> Any:
        del context
        values = observation
        pole_angle = float(values[2]) if len(values) > 2 else 0.0
        return 1 if pole_angle > 0 else 0
