"""Pendulum-Hardcore starter policy.

Same contract as the pendulum starter; returns zero torque so the
agent has a working skeleton on turn 0 without bias toward any
particular controller.

Iterate by editing ``act()``. See ``TASK.md`` for the parameter
ranges (mass / length / gravity vary per episode within those
ranges; specific values are NOT exposed to the policy).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Pendulum-Hardcore ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        # Cache action bounds for defensive clipping.
        self.action_low = float(action_space["low"][0])
        self.action_high = float(action_space["high"][0])

    def reset(self, episode_index: int) -> None:
        """Per-episode reset. The env's hidden seed determines (m, l, g)
        for this episode within the ranges documented in TASK.md."""
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: np.ndarray shape (3,) dtype float32.
                 [cos(theta), sin(theta), theta_dot], theta=0 is upright.

        Returns:
            np.ndarray shape (1,) dtype float32, torque in [-2.0, 2.0].
        """
        # === Replace this body with your controller ===
        u = 0.0
        return np.array([u], dtype=np.float32)
