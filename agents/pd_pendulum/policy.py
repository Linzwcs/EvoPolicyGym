"""Reference Pendulum-v1 policy: energy-shaping swing-up + PD stabilize.

This is a hlbench *consumer* (an agent solution), not part of the
hlbench server library. Drop this file into ``workspace/system/policy.py``
to submit it against a running hlbench server.

The two-regime strategy:

  |theta| < 0.5 rad  →  linear PD on (theta, theta_dot)
  otherwise          →  energy pumping  u = -K * theta_dot * (E - E_top)

with  E = 0.5 * theta_dot^2 + (g/l) * cos(theta)  and  E_top = g/l.
Pendulum-v1 uses g=10, l=1, m=1 and an action torque bounded to [-2, 2].

Empirically averages ~ -135 reward over 200-step episodes from random
starts (vs. ~ -150 for an LQR-tuned expert and ~ -1200 for a uniform
random policy)."""

from __future__ import annotations

import math


class Policy:
    """Implements the Policy interface from SPEC.md §2."""

    KP = 30.0
    KD = 5.0
    K_ENERGY = 1.0
    A_MAX = 2.0
    G_OVER_L = 10.0

    def __init__(self, obs_space=None, action_space=None, env_meta=None):
        # MVP: ignore env_meta; the controller is hand-tuned to Pendulum-v1.
        # Real agents would read action_space["high"]/["low"] to set A_MAX
        # and read obs_space to handle obs shape.
        pass

    def reset(self, episode_index: int) -> None:
        # PD has no per-episode state to reset.
        del episode_index

    def act(self, obs):
        cos_t, sin_t, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
        theta = math.atan2(sin_t, cos_t)  # in [-pi, pi]; 0 is upright

        if abs(theta) < 0.5:
            u = -self.KP * theta - self.KD * theta_dot
        else:
            E = 0.5 * theta_dot * theta_dot + self.G_OVER_L * cos_t
            u = -self.K_ENERGY * theta_dot * (E - self.G_OVER_L)

        u = max(-self.A_MAX, min(self.A_MAX, u))
        return [u]
