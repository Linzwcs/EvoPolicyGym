"""K-Server starter policy.

Always dispatches server 0 — terrible (server 0 racks up huge
distance), but the contract is unambiguous. Iterate by editing
``act()``; see ``TASK.md`` for greedy/anticipation strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """K-Server ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        self.n_servers = int(action_space["n"])

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> int:
        """Per-step action.

        Args:
            obs: np.ndarray shape (8,) dtype float32.
                 [s0.x, s0.y, s1.x, s1.y, s2.x, s2.y, req.x, req.y]

        Returns:
            int in [0, n_servers) — which server to dispatch.
        """
        # === Replace this body with your controller ===
        return 0
