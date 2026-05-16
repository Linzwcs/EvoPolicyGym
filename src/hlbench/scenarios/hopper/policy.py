"""Baseline Hopper policy."""

from __future__ import annotations

from typing import Any


class Policy:
    def reset(self, task_config: dict[str, Any]) -> None:
        del task_config

    def act(self, observation: Any, context: dict[str, Any]) -> list[float]:
        del observation, context
        return [0.0, 0.0, 0.0]
