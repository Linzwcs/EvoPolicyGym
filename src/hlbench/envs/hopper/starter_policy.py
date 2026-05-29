"""Hopper-v5 starter policy.

Returns zero torque on every joint — robot collapses or stands still.
Edit ``act()`` to implement a controller. See ``TASK.md`` for obs
layout + reward structure + strategy hints.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Hopper-v5 ``Policy`` implementing the SPEC §2 contract."""

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
            obs: np.ndarray shape (11,) dtype float64. See TASK.md.

        Returns:
            np.ndarray shape (3,) dtype float32, joint torques in [-1.0, 1.0].
        """
        # === Replace this body with your controller ===
        return np.zeros(self.action_dim, dtype=np.float32)
