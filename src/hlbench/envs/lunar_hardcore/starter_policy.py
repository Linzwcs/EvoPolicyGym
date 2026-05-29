"""Lunar-Hardcore starter policy.

Returns zero on both engines — the lander will free-fall and crash.
The contract is unambiguous; agent gets clean trajectory data on
turn 0 to start iterating from.

Iterate by editing ``act()``. See ``TASK.md`` for wind / turbulence
ranges and the 8-D obs layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Lunar-Hardcore ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        self.action_dim = int(np.prod(action_space["shape"]))
        self.action_low = np.array(action_space["low"], dtype=np.float32)
        self.action_high = np.array(action_space["high"], dtype=np.float32)

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: np.ndarray shape (8,) dtype float32.
                 [x, y, vx, vy, angle, angular_vel, leg1, leg2]

        Returns:
            np.ndarray shape (2,) dtype float32, [main, side] each
            in [-1.0, 1.0]. See TASK.md for engine conventions.
        """
        # === Replace this body with your controller ===
        return np.zeros(self.action_dim, dtype=np.float32)
