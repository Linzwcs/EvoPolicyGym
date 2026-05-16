"""Baseline LunarLander policy."""

from __future__ import annotations

from typing import Any


class Policy:
    def reset(self, task_config: dict[str, Any]) -> None:
        del task_config

    def act(self, observation: Any, context: dict[str, Any]) -> int:
        del context
        x_position = float(observation[0])
        y_position = float(observation[1])
        x_velocity = float(observation[2])
        y_velocity = float(observation[3])
        angle = float(observation[4])
        angular_velocity = float(observation[5])
        left_contact = bool(observation[6])
        right_contact = bool(observation[7])

        target_angle = max(-0.35, min(0.35, 0.55 * x_position + 0.80 * x_velocity))
        angle_error = target_angle - angle
        turn_signal = 0.70 * angle_error - 0.80 * angular_velocity

        target_height = 0.30 + 0.45 * abs(x_position)
        descent_signal = 0.55 * (target_height - y_position) - 0.70 * y_velocity
        if left_contact or right_contact:
            descent_signal = -0.65 * y_velocity
            turn_signal *= 0.25

        if descent_signal > max(0.12, abs(turn_signal)):
            return 2
        if turn_signal > 0.06:
            return 1
        if turn_signal < -0.06:
            return 3
        return 0
