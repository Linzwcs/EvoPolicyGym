"""MountainCarContinuous-v0 starter policy.

Returns zero engine push every step. The car oscillates briefly under
gravity then settles at the valley bottom — random_baseline-like
behavior (return ≈ 0).

Replace ``act()`` with your controller. The interface comments below
double as the canonical contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """MountainCarContinuous-v0 ``Policy`` implementing SPEC §2."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        """Constructed once per submit.

        Args:
            obs_space:    {"type": "Box", "shape": [2],
                           "low": [-1.2, -0.07],
                           "high": [0.6, 0.07],
                           "dtype": "float32"}
            action_space: {"type": "Box", "shape": [1],
                           "low": [-1.0], "high": [1.0],
                           "dtype": "float32"}
            env_meta:     run-wide context."""
        self.action_low = float(action_space["low"][0])
        self.action_high = float(action_space["high"][0])

    def reset(self, episode_index: int) -> None:
        """Called at the start of every episode in the submit."""
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: ``np.ndarray`` shape ``(2,)`` dtype ``float32``.
                 Components:
                   obs[0] = position in [-1.2, 0.6]
                   obs[1] = velocity in [-0.07, 0.07]

        Returns:
            ``np.ndarray`` shape ``(1,)`` dtype ``float32``, engine
            push in ``[-1.0, 1.0]``.
        """
        # === Replace this body with your controller ===
        # Reference deconstruction:
        #     position = float(obs[0])
        #     velocity = float(obs[1])
        # Bang-bang baseline: u = math.copysign(1.0, velocity) once moving.
        u = 0.0  # zero throttle — see TASK.md "Strategy hints"
        return np.array([u], dtype=np.float32)
