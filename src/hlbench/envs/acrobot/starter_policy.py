"""Acrobot-v1 starter policy.

Returns the no-op action (torque=0) every step. Expected return ≈
random baseline (-500) — the system never builds enough energy to
swing up.

Replace ``act()`` with your controller. The interface comments below
double as the canonical contract for ``__init__`` / ``reset`` / ``act``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Acrobot-v1 ``Policy`` implementing the SPEC §2 contract."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        """Constructed once per submit.

        Args:
            obs_space:    {"type": "Box", "shape": [6],
                           "low":  [-1, -1, -1, -1, -12.566, -28.274],
                           "high": [ 1,  1,  1,  1,  12.566,  28.274],
                           "dtype": "float32"}
            action_space: {"type": "Discrete", "n": 3}
                          Maps to torque values: 0→-1, 1→0, 2→+1.
            env_meta:     env / submit_index / n_episodes_this_submit /
                          remaining_budget_after / max_episode_steps /
                          allowed_imports."""
        # Acrobot has Discrete action: cache n for any future random-baseline
        # fallback (also documents the contract).
        self.n_actions = int(action_space["n"])

    def reset(self, episode_index: int) -> None:
        """Called at the start of every episode in the submit."""
        del episode_index

    def act(self, obs: np.ndarray) -> int:
        """Per-step action.

        Args:
            obs: ``np.ndarray`` shape ``(6,)`` dtype ``float32``.
                 Components:
                   obs[0..1] = cos(theta1), sin(theta1)
                   obs[2..3] = cos(theta2), sin(theta2)
                   obs[4]    = theta1_dot
                   obs[5]    = theta2_dot

        Returns:
            ``int`` in ``{0, 1, 2}`` — torque {-1, 0, +1} applied at the
            elbow joint.

        Wall-time limit per call: ``act_wall_ms`` from ``GET /info``.
        """
        # === Replace this body with your controller ===
        # Reference deconstruction of ``obs``:
        #     import math
        #     cos_t1, sin_t1, cos_t2, sin_t2, w1, w2 = obs
        #     theta1 = math.atan2(float(sin_t1), float(cos_t1))
        #     theta2 = math.atan2(float(sin_t2), float(cos_t2))
        return 1  # no-op torque — see TASK.md "Strategy hints"
