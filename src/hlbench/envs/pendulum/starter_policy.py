"""Pendulum-v1 starter policy.

Copied into ``workspace/system/policy.py`` by ``Server.__init__`` so
the agent has a valid skeleton on turn 0 — the interface contract
(class name, method signatures, action shape/dtype) is unambiguous.

This implementation returns **zero torque** every step. That's a
deliberately bad policy (expected return ≈ random baseline) so the
starter doesn't bias the agent toward a working solution; the agent
still has to write the controller. Iterate by editing ``act()``.

Why ship a starter at all (rather than an empty workspace):
  - Eliminates an entire failure mode (typos in __init__ signature,
    wrong action shape/dtype) that wastes a whole submit's budget.
  - The signature comments below double as machine-checkable docs:
    if you change ``obs`` shape you'll see it immediately, etc.
  - First submit is guaranteed to run, so the agent gets real feedback
    on turn 0 instead of debugging the contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Pendulum-v1 ``Policy`` implementing the SPEC §2 contract.

    Replace the body of ``act()`` with your controller. The signatures
    of ``__init__`` / ``reset`` / ``act`` MUST remain as shown — the
    harness imports this class as ``Policy`` and instantiates it once
    per submit (see SPEC §2)."""

    def __init__(
        self,
        obs_space: dict[str, Any],
        action_space: dict[str, Any],
        env_meta: dict[str, Any],
    ) -> None:
        """Constructed once per submit (shared across all episodes in
        the submit). Allowed to read files under ``system/`` for
        persistent state across submits.

        Args:
            obs_space:    {"type": "Box", "shape": [3],
                           "low": [-1.0, -1.0, -8.0],
                           "high": [1.0, 1.0, 8.0],
                           "dtype": "float32"}
            action_space: {"type": "Box", "shape": [1],
                           "low": [-2.0], "high": [2.0],
                           "dtype": "float32"}
            env_meta:     also contains env / submit_index /
                          n_episodes_this_submit / remaining_budget_after /
                          max_episode_steps / allowed_imports."""
        # Cache action bounds so ``act()`` can clip without re-fetching.
        # (Returning out-of-bound actions is allowed — the env clips —
        # but agents that clip themselves get cleaner trajectories.)
        self.action_low = float(action_space["low"][0])
        self.action_high = float(action_space["high"][0])

    def reset(self, episode_index: int) -> None:
        """Called at the start of every episode in the submit.
        ``episode_index`` ranges over ``[0, n_episodes_this_submit)``.

        The episode seed is NOT passed and MUST NOT be inferred."""
        # No per-episode state to reset for this starter.
        del episode_index

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Per-step action.

        Args:
            obs: np.ndarray shape ``(3,)`` dtype ``float32``.
                 Components: ``[cos(theta), sin(theta), theta_dot]``
                 where ``theta = 0`` is upright (target).

        Returns:
            np.ndarray shape ``(1,)`` dtype ``float32``, a torque in
            ``[-2.0, 2.0]`` N·m. (Returning a Python ``list`` of one
            float, e.g. ``[u]``, also works — the env coerces — but
            numpy is the canonical type.)

        Wall-time limit per call: ``act_wall_ms`` from ``GET /info``
        (default 1000 ms in the local config; SPEC default is 10 ms).
        """
        # === Replace this body with your controller ===
        # Reference deconstruction of ``obs``:
        #     cos_theta = float(obs[0])
        #     sin_theta = float(obs[1])
        #     theta_dot = float(obs[2])
        #     theta = math.atan2(sin_theta, cos_theta)  # in [-pi, pi]
        u = 0.0  # zero torque — see TASK.md "Strategy hints"
        return np.array([u], dtype=np.float32)
