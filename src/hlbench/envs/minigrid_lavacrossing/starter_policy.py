"""MiniGrid-LavaCrossingS11N5-v0 starter policy.

Always returns action 0 (turn_left). The agent will spin in place and
never finish — score 0. Iterate by editing ``act()``. See ``TASK.md``
for the action set and obs layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """MiniGrid-LavaCrossingS11N5-v0 ``Policy`` implementing the SPEC §2 contract."""

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
            obs: np.ndarray shape (50,) dtype uint16.
                 First 49 = packed 7x7 egocentric grid; each cell is
                 ``type * 100 + color * 10 + state`` (decode below).
                 Position 49 = agent direction in {0, 1, 2, 3}.

                 Decoding example:
                     grid = obs[:49].reshape(7, 7)
                     cell_type  = grid // 100      # MiniGrid object_type id
                     cell_color = (grid // 10) % 10  # color id
                     cell_state = grid % 10           # state id

        Returns:
            int in [0, 7):
                0=turn_left  1=turn_right  2=move_forward
                3=pickup     4=drop         5=toggle (e.g., open door)
                6=done (declare task complete)
        """
        # === Replace this body with your controller ===
        return 0
