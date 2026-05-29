"""Acrobot-v1 env registration.

Two-link pendulum swing-up: a Discrete-action, hard-exploration
classic-control benchmark. Distinct from Pendulum-v1 in three ways
worth covering:
  - **Discrete action space** (3 torques: -1, 0, +1) — exercises the
    Discrete serialization path in env_runner that Pendulum doesn't.
  - **Sparse reward** (-1 per step until tip clears the bar) —
    requires the policy to find a swing-up strategy in a non-shaped
    reward landscape.
  - **Episode length up to 500 steps**, much longer than Pendulum's 200.

Baselines are approximate (published RL benchmark values); calibrate
against a real PPO/SAC reference before publishing comparative scores.

Side effect: importing this module registers `acrobot` with the registry.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("Acrobot-v1", render_mode=None)


register_env(
    env_id="acrobot",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [6],
        # cos/sin of two joint angles + angular velocities
        "low": [-1.0, -1.0, -1.0, -1.0, -12.566, -28.274],
        "high": [1.0, 1.0, 1.0, 1.0, 12.566, 28.274],
        "dtype": "float32",
    },
    action_space={
        "type": "Discrete",
        "n": 3,
        # Encodes torque {-1, 0, +1} applied at the second joint.
    },
    max_episode_steps=500,
    # Server-internal. Random ~ -500 (max possible neg = couldn't solve);
    # expert ~ -80 (consistent swing-up in well under 100 steps).
    # Approximate; calibrate.
    expert_baseline=-80.0,
    random_baseline=-500.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=256,
    obs_storage="inline",
    reward_components=None,
)
