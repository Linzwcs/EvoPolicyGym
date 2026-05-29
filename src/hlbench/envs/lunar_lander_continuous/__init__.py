"""LunarLanderContinuous-v3 env registration.

A lander must descend and touch down softly between two flags on the
moon's surface. Continuous Box(2,) action commands main + side engines;
8-dim obs covers position / velocity / angle / leg contact flags.

Why include:
  - **Reward-shaping landscape distinct from BipedalWalker**: dense
    shaping reward + sparse landing bonus / crash penalty.
  - **2-D action**, smaller than BipedalWalker (4-D) but harder than
    Pendulum (1-D) — covers middle action dimensionality.
  - **Discrete success condition** within continuous action — agent
    must decide WHEN to cut engines.

Side effect: importing this module registers `lunar_lander_continuous`.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("LunarLanderContinuous-v3", render_mode=None)


register_env(
    env_id="lunar_lander_continuous",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [8],
        # [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]
        "low": [-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0],
        "high": [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0],
        "dtype": "float32",
    },
    action_space={
        "type": "Box",
        "shape": [2],
        # [main_engine, side_engine]
        # main:  -1..0 = off,  0..1 = throttle increasing
        # side:  -1..-0.5 = left, -0.5..0.5 = off, 0.5..1 = right
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
        "dtype": "float32",
    },
    max_episode_steps=1000,
    # Random ≈ -150 (crashes); solved threshold ≥ +200 per gym docs.
    # Approximate; calibrate.
    expert_baseline=200.0,
    random_baseline=-150.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
