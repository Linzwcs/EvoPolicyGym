"""Cache-Replacement starter policy.

Always evicts slot 0 — terrible but safe. The contract is
unambiguous; the agent gets clean trajectory data on turn 0 to
iterate from.

Edit ``act()`` to implement LRU, LFU, ARC, or a custom heuristic.
See ``TASK.md`` for obs layout and strategy hints.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Cache-Replacement ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        # action_space is Discrete; n is the cache capacity.
        self.n_slots = int(action_space["n"])

    def reset(self, episode_index: int) -> None:
        # Per-episode reset: clear any per-slot tracking state here.
        del episode_index

    def act(self, obs: np.ndarray) -> int:
        """Per-step action.

        Args:
            obs: np.ndarray shape (17,) dtype int32. See TASK.md for
                 the [cache_slots | history | current_access] layout.

        Returns:
            int in [0, n_slots) — slot index to evict on miss
            (ignored on hit but must be a valid int).
        """
        # === Replace this body with your controller ===
        return 0
