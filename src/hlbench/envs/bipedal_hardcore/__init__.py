"""Bipedal-Hardcore env registration (v1 roster #16).

Wraps Gymnasium's ``BipedalWalkerHardcore-v3`` — a procedurally
harder terrain variant with stumps, ladder-like obstacles, and pits
(in contrast to ``BipedalWalker-v3`` which uses gentle hills only).

The hardcore variant's procedural terrain is governed by the env's
own RNG seeded via ``reset(seed=...)`` — so train and held-out seed
pools naturally produce different terrain layouts. No wrapper needed.

Why this is in the roster:
  - **Inherent procedural generalization**: each seed yields a unique
    sequence of obstacle types and spacings. Policies that memorize
    a specific obstacle order fail on held-out.
  - **High discrimination expected**: hand-crafted gaits rarely clear
    BipedalWalkerHardcore (PPO/SAC trained 50M+ steps clear it; LLM
    one-shot policies typically score near zero). The FAILURE
    PATTERNS — falls on stumps vs gaps vs steps — differ across
    frontier models, providing model-discrimination signal even when
    absolute scores cluster low.
  - **Zero new infra**: existing Gymnasium env, no observations.npy
    dependency, no per-env act_wall_ms relaxation needed.

Side effect: importing this module registers ``bipedal_hardcore``.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("BipedalWalkerHardcore-v3", render_mode=None)


register_env(
    env_id="bipedal_hardcore",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [24],
        # Same observation layout as BipedalWalker-v3: hull angle/vel,
        # joint angles/vel, ground contact, 10 LIDAR readings.
        "low": [-3.14] + [-5.0] * 13 + [-0.0] + [-0.0] + [0.0] * 10,
        "high": [3.14] + [5.0] * 13 + [1.0] + [1.0] + [1.0] * 10,
        "dtype": "float32",
    },
    action_space={
        "type": "Box",
        "shape": [4],
        "low": [-1.0, -1.0, -1.0, -1.0],
        "high": [1.0, 1.0, 1.0, 1.0],
        "dtype": "float32",
    },
    max_episode_steps=2000,
    # Random ≈ -100 (falls within first stump). Solved threshold per
    # gym docs is +300 over 100 episodes — virtually no hand-crafted
    # policy clears this; PPO/SAC need 50M+ steps. The expected score
    # range for LLM-written policies is roughly [-100, +50].
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
