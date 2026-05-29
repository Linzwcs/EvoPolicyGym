"""MountainCarContinuous-v0 env registration.

Underpowered car at the bottom of a valley must reach the flag on the
right hill. Engine torque is too weak to climb directly — the policy
must oscillate left/right to build momentum. Sparse near-zero reward
plus continuous Box action makes this a meaningful step up from
Pendulum:

  - **Hard exploration** — random actions almost never reach the goal
    in 999 steps; random_baseline ≈ -1 (small ctrl penalty, no
    success bonus).
  - **Continuous Box action** like Pendulum, but a SCALAR signed
    push instead of a torque — exercises the same serialization with
    different bounds.
  - **Variable episode length** — terminates as soon as the goal is
    reached, otherwise truncates at 999 steps.

Side effect: importing this module registers `mountain_car_continuous`
with the registry.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("MountainCarContinuous-v0", render_mode=None)


register_env(
    env_id="mountain_car_continuous",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [2],
        # [position, velocity]
        "low": [-1.2, -0.07],
        "high": [0.6, 0.07],
        "dtype": "float32",
    },
    action_space={
        "type": "Box",
        "shape": [1],
        "low": [-1.0],
        "high": [1.0],
        "dtype": "float32",
    },
    max_episode_steps=999,
    # Random ≈ -1 (small ctrl cost, never reaches goal).
    # Expert ≈ 95 (reaches the flag with minimum total |action|² penalty).
    # Approximate; calibrate.
    expert_baseline=95.0,
    random_baseline=-1.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
