"""Car-Racing-Pixel starter policy.

Returns `[0, 0.5, 0]` — straight, half throttle, no brake. Car
drives straight off the track. Iterate by editing ``act()``.

Note: obs is delivered as a NumPy array via the env's normal step()
return — the external-storage of obs in observations.npy is a
**feedback artifact** (so the agent can inspect old trajectories
later), but during episode execution the agent's act() still sees
the full numpy frame each step.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Car-Racing-Pixel ``Policy`` implementing the SPEC §2 contract."""

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
            obs: np.ndarray shape (96, 96, 3) dtype uint8. RGB pixels.

        Returns:
            np.ndarray shape (3,) dtype float32: [steering, gas, brake].
        """
        # === Replace this body with your controller ===
        return np.array([0.0, 0.5, 0.0], dtype=np.float32)
