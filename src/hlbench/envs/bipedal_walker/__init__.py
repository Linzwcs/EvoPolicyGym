"""BipedalWalker-v3 env registration.

A 4-DOF bipedal robot must walk forward over slightly uneven terrain
without falling. Continuous Box(4,) action commands torque on each
of the four leg joints; observation is a 24-dim hull/joint/lidar mix.

Why include in the benchmark:
  - **Significantly harder than classic control**: random policies get
    ≈ -100 (falls in a few steps); a working PPO/SAC takes thousands
    of episodes of training to clear +200.
  - **High-dimensional continuous control** (4-D action × 24-D obs).
  - **Mild perception**: 10 of the 24 obs values are LIDAR distance
    readings — the policy has to make sense of them.
  - Long episodes (1600 steps max) so submit_wall_s should be set
    accordingly when running.

Side effect: importing this module registers `bipedal_walker`.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("BipedalWalker-v3", render_mode=None)


register_env(
    env_id="bipedal_walker",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [24],
        # Hull angle/velocities (4) + joint angles/velocities (8) +
        # ground contact (2) + 10 LIDAR readings.
        # Bounds are wide; gym uses ±inf for some, but documenting the
        # nominal float32 ranges here for agent reference.
        "low": [-3.14] + [-5.0] * 13 + [-0.0] + [-0.0] + [0.0] * 10,
        "high": [3.14] + [5.0] * 13 + [1.0] + [1.0] + [1.0] * 10,
        "dtype": "float32",
    },
    action_space={
        "type": "Box",
        "shape": [4],
        # Hip-1, knee-1, hip-2, knee-2 torques.
        "low": [-1.0, -1.0, -1.0, -1.0],
        "high": [1.0, 1.0, 1.0, 1.0],
        "dtype": "float32",
    },
    max_episode_steps=1600,
    # Random ≈ -100 (falls quickly, accumulates ctrl penalty); expert
    # ≈ 300 ("solved" threshold per the gym docs is +300).
    # Approximate; calibrate.
    expert_baseline=300.0,
    random_baseline=-100.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
