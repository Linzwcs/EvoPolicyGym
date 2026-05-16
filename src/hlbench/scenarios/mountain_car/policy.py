"""Baseline MountainCar policy."""

from __future__ import annotations

from typing import Any


class Policy:
    def reset(self, task_config: dict[str, Any]) -> None:
        del task_config

    def act(self, observation: Any, context: dict[str, Any]) -> int:
        del context
        velocity = float(observation[1])
        return 2 if velocity >= 0.0 else 0
