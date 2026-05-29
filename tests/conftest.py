"""Shared test fixtures.

Lives in tests/ rather than src/ — these helpers exist for testing the
hlbench machinery, not as part of the runtime API. Real agent policies
ship in `system/policy.py` per SPEC.md §2.
"""

from __future__ import annotations

import math
from typing import Any


class PendulumPDPolicy:
    """Energy-shaping swing-up + PD stabilization for Pendulum-v1.

    Pendulum obs is `[cos(theta), sin(theta), theta_dot]`, where theta=0 is
    upright. Pure linear PD on theta can't swing up from the bottom (control
    torque saturates at +/-2 N*m and gravity dominates), so we run two regimes:

      |theta| < 0.5 rad:  PD on (theta, theta_dot) — stabilize near upright.
      otherwise:          pump/drain energy via u = -K * theta_dot * (E - E_top)
                          where E = 0.5*theta_dot^2 + (g/l)*cos(theta).

    Reliably beats ~-300 mean return on Pendulum-v1 across random init states.

    Implements the Policy interface from SPEC.md §2 (`reset`, `act`).
    """

    KP = 30.0
    KD = 5.0
    K_ENERGY = 1.0
    A_MAX = 2.0       # Pendulum action range is [-2, 2]
    G_OVER_L = 10.0   # Pendulum-v1 uses g=10, l=1, m=1

    def __init__(
        self,
        obs_space: Any = None,
        action_space: Any = None,
        env_meta: dict[str, Any] | None = None,
    ) -> None:
        self._episode_count = 0
        self._last_episode_index: int | None = None

    def reset(self, episode_index: int) -> None:
        self._episode_count += 1
        self._last_episode_index = episode_index

    def act(self, obs: Any) -> Any:
        cos_t, sin_t, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
        theta = math.atan2(sin_t, cos_t)  # in [-pi, pi]; theta=0 is upright

        if abs(theta) < 0.5:
            # PD stabilization near upright.
            u = -self.KP * theta - self.KD * theta_dot
        else:
            # Energy pumping: torque in direction of motion when below E_top.
            E = 0.5 * theta_dot * theta_dot + self.G_OVER_L * cos_t
            E_top = self.G_OVER_L
            u = -self.K_ENERGY * theta_dot * (E - E_top)

        u = max(-self.A_MAX, min(self.A_MAX, u))
        # Pendulum action_space is Box(shape=(1,)); return a 1-elem list/array.
        try:
            import numpy as np

            return np.array([u], dtype=np.float32)
        except ImportError:  # pragma: no cover
            return [u]


class CrashingPolicy:
    """Policy that raises in act() — used to test ended_with_error path."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def reset(self, episode_index: int) -> None:
        pass

    def act(self, obs: Any) -> Any:
        raise RuntimeError("intentional test crash")
