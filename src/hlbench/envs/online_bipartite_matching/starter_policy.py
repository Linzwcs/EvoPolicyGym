"""Online-Bipartite-Matching starter policy.

Always skips — terrible but safe (zero reward, no invalid attempts).
The contract is unambiguous; iterate by editing ``act()``.

See ``TASK.md`` for greedy / RANKING / reservation strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Online-Bipartite-Matching ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        # action_space is Discrete; n = N_LEFT + 1 (last action = skip)
        self.n_actions = int(action_space["n"])
        self.skip_action = self.n_actions - 1  # last is skip
        self.n_left = self.n_actions - 1  # number of left vertices

    def reset(self, episode_index: int) -> None:
        del episode_index

    def act(self, obs: np.ndarray) -> int:
        """Per-step action.

        Args:
            obs: np.ndarray shape (32,) dtype int8.
                 [left_matched_mask (16) | current_arrival_neighbors (16)]

        Returns:
            int in [0, n_actions). [0, n_left) = match to that left
            vertex. n_left = skip.
        """
        # === Replace this body with your controller ===
        return self.skip_action
