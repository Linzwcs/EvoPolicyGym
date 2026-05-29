"""Pendulum-from-Pixels starter policy.

Returns zero torque every step. Iterate by editing ``act()``; see
``TASK.md`` for vision strategies (most important: cache the
previous frame in instance state to estimate angular velocity).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Pendulum-from-Pixels ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        self.action_low = float(action_space["low"][0])
        self.action_high = float(action_space["high"][0])
        # Cache previous frame here for velocity estimation across steps.
        self.prev_obs: np.ndarray | None = None

    def reset(self, episode_index: int) -> None:
        del episode_index
        self.prev_obs = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: np.ndarray shape (64, 64, 3) dtype uint8. RGB pendulum render.

        Returns:
            np.ndarray shape (1,) dtype float32, torque in [-2.0, 2.0].
        """
        # === Replace this body with your controller ===
        # Hint: use self.prev_obs (cached from last step) to estimate
        # angular velocity from the angle change between frames.
        self.prev_obs = obs
        return np.array([0.0], dtype=np.float32)
