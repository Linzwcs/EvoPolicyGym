"""LunarLanderContinuous-v3 starter policy.

Returns ``[0, 0]`` every step (engines off → free-fall under gravity).
Expected return ≈ random_baseline (-150) — the lander crashes.

Replace ``act()`` with your controller. The action encoding is
non-trivial: see the comments in ``act()`` below.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """LunarLanderContinuous-v3 ``Policy`` implementing SPEC §2."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        """Constructed once per submit.

        Args:
            obs_space:    {"type": "Box", "shape": [8], "dtype": "float32"}
            action_space: {"type": "Box", "shape": [2],
                           "low":  [-1, -1], "high": [1, 1],
                           "dtype": "float32"}
            env_meta:     run-wide context."""
        pass

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: ``np.ndarray`` shape ``(8,)`` dtype ``float32``.
                 obs[0..1] = x, y position
                 obs[2..3] = x, y velocity
                 obs[4..5] = angle, angular velocity
                 obs[6..7] = left/right leg ground contact (0 or 1)

        Returns:
            ``np.ndarray`` shape ``(2,)`` dtype ``float32``.
              action[0] = main engine:
                  < 0   → off
                  [0,1] → throttle proportional
              action[1] = side thrusters:
                  < -0.5 → left thruster on
                  [-0.5, 0.5] → both off
                  > 0.5  → right thruster on
        """
        # === Replace this body with your controller ===
        return np.array([0.0, 0.0], dtype=np.float32)
