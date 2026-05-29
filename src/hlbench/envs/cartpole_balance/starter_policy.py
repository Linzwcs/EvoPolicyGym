"""CartPole-Balance starter policy.

Always returns action 0 (push left). The cart drifts off the track
within a few hundred steps; score ≈ 100. Iterate by replacing the
body of ``act()`` with an angle-based PD (see TASK.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """CartPole-Balance ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        self.n_actions = int(action_space["n"])

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> int:
        """Per-step action.

        Args:
            obs: np.ndarray shape (4,) dtype float32.
                 [cart_x, cart_v, pole_angle, pole_omega]

        Returns:
            int in {0, 1}: 0 = push left, 1 = push right.
        """
        # === Replace this body with your controller ===
        # Hint: simple angle-based PD reaches 500 immediately.
        # action = 0 if (obs[2] * 1.0 + obs[3] * 0.1) < 0 else 1
        return 0
