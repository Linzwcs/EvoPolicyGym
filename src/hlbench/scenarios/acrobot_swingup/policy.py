"""Baseline Acrobot policy."""

from __future__ import annotations

import math
from typing import Any


class Policy:
    def reset(self, task_config: dict[str, Any]) -> None:
        del task_config

    def act(self, observation: Any, context: dict[str, Any]) -> int:
        del context
        cos_theta_1 = float(observation[0])
        sin_theta_1 = float(observation[1])
        cos_theta_2 = float(observation[2])
        sin_theta_2 = float(observation[3])
        theta_dot_1 = float(observation[4])
        theta_dot_2 = float(observation[5])

        theta_1 = math.atan2(sin_theta_1, cos_theta_1)
        theta_2 = math.atan2(sin_theta_2, cos_theta_2)
        swing_signal = (
            0.8 * theta_dot_1
            + theta_dot_2
            + 0.4 * math.sin(theta_1)
            + 0.8 * math.sin(theta_1 + theta_2)
        )
        return 2 if swing_signal >= 0.0 else 0
