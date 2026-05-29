"""Bipedal-Hardcore starter policy.

Returns zero torque on every joint — the biped will collapse
immediately on the first obstacle, but the contract is unambiguous
and the agent gets clean trajectory data on turn 0.

Iterate by editing ``act()``. See ``TASK.md`` for the obs layout
(LIDAR readings 14-23 are essential for obstacle-aware policies).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Bipedal-Hardcore ``Policy`` implementing the SPEC §2 contract."""

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
            obs: np.ndarray shape (24,) dtype float32. See TASK.md
                 for the index → component mapping.

        Returns:
            np.ndarray shape (4,) dtype float32, joint torques in
            [-1.0, 1.0] (hip-1, knee-1, hip-2, knee-2).
        """
        # === Replace this body with your controller ===
        return np.zeros(self.action_dim, dtype=np.float32)
