"""Hopper-v5 env registration (Hopper-v5).

MuJoCo locomotion task: Hop forward on a single leg without falling.

Direct wrap of Gymnasium's ``Hopper-v5``. Per-seed variation comes from
the env's RNG (initial joint perturbations); no domain-randomization
wrapper is applied. Held-out seeds therefore differ only in initial
state, not in physics parameters — generalization here is "robustness
to initial-state distribution" rather than parameter OOD.

Side effect: importing this module registers ``hopper``.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("Hopper-v5", render_mode=None)


register_env(
    env_id="hopper",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [11],
        "low": [-float("inf")] * 11,
        "high": [float("inf")] * 11,
        "dtype": "float64",
    },
    action_space={
        "type": "Box",
        "shape": [3],
        "low": [-1.0] * 3,
        "high": [1.0] * 3,
        "dtype": "float32",
    },
    max_episode_steps=1000,
    expert_baseline=3500.0,
    random_baseline=5.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
