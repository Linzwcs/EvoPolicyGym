# Pendulum-v1 — Swing-Up and Stabilize

## Goal

Swing a frictionless pendulum to the upright position (angle = 0) and
keep it balanced there with minimum control effort, starting from a
random initial state each episode.

## Observation

`Box(shape=[3])` — `[cos(theta), sin(theta), theta_dot]`. The angle
convention is `theta = 0` upright, `theta = ±π` hanging down. Angular
velocity is in radians/second, bounded to `[-8, 8]`.

## Action

`Box(shape=[1])` — a continuous torque applied at the pivot, bounded
to `[-2.0, 2.0]` N·m. Out-of-bound actions are clipped by the env;
the value returned by your policy is what we record in
`trajectory.jsonl` (pre-clip), per SPEC §4.2.

## Reward

Per step: `-(theta² + 0.1·theta_dot² + 0.001·u²)`. Reward is always
non-positive; the best achievable episode return on a 200-step episode
is bounded above by `0`. There are no `reward_components`.

## Episode structure

- 200 steps per episode (no natural termination — the time limit is
  the only end condition; expect `truncated=true` on the last step).
- Initial state at each `reset()` is drawn uniformly from
  `theta ∈ [-π, π]` and `theta_dot ∈ [-1, 1]` by the seed.
- The seed-to-instance mapping is hidden; you address an env instance
  by integer ID in `[0, n_env_instances)`. Submitting to the same ID
  twice gives you the same trajectory under a deterministic policy.

## Strategy hints

A pure linear PD on `(theta, theta_dot)` cannot swing the pendulum up
from the bottom — control torque saturates and gravity dominates. A
common approach is a two-regime controller: energy pumping when far
from upright, PD when close. Other approaches (LQR, MPC, RL) are
also fair game.
