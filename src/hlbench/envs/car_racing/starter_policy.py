"""Car-Racing starter policy.

Returns `[0, 0.5, 0]` — straight steering, half throttle, no brake.
Car drives straight off the track on first turn. Score ≈ -50.
Iterate by editing ``act()``; see ``TASK.md`` for strategy hints.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Car-Racing ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        self.action_dim = int(np.prod(action_space["shape"]))

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: np.ndarray shape (16, 16, 3) dtype uint8. RGB pixels.

        Returns:
            np.ndarray shape (3,) dtype float32: [steering, gas, brake].
        """
        # === Replace this body with your controller ===
        return np.array([0.0, 0.5, 0.0], dtype=np.float32)
