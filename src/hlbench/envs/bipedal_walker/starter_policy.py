"""BipedalWalker-v3 starter policy.

Returns zero torques every step. The robot stands briefly under
gravity, then collapses → expected return ≈ random_baseline (-100).

Replace ``act()`` with your controller.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """BipedalWalker-v3 ``Policy`` implementing SPEC §2."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        """Constructed once per submit.

        Args:
            obs_space:    {"type": "Box", "shape": [24], "dtype": "float32"}
            action_space: {"type": "Box", "shape": [4],
                           "low":  [-1, -1, -1, -1],
                           "high": [ 1,  1,  1,  1],
                           "dtype": "float32"}
                          Action = torques on (hip1, knee1, hip2, knee2).
            env_meta:     run-wide context."""
        self.action_dim = int(action_space["shape"][0])

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: ``np.ndarray`` shape ``(24,)`` dtype ``float32``.
                 See TASK.md for the full decomposition. Key indices:
                 obs[0]   = hull angle
                 obs[8]   = leg-1 ground contact (1.0 if touching)
                 obs[13]  = leg-2 ground contact
                 obs[14:] = 10 LIDAR readings

        Returns:
            ``np.ndarray`` shape ``(4,)`` dtype ``float32``, joint
            torques in ``[-1, 1]``.
        """
        # === Replace this body with your controller ===
        return np.zeros(self.action_dim, dtype=np.float32)
