"""Ant-v5 env registration (Ant-v5).

MuJoCo locomotion task: Locomote a 4-legged ant forward.

Direct wrap of Gymnasium's ``Ant-v5``. Per-seed variation comes from
the env's RNG (initial joint perturbations); no domain-randomization
wrapper is applied. Held-out seeds therefore differ only in initial
state, not in physics parameters — generalization here is "robustness
to initial-state distribution" rather than parameter OOD.

Side effect: importing this module registers ``ant``.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("Ant-v5", render_mode=None)


register_env(
    env_id="ant",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [105],
        "low": -float("inf"),
        "high": float("inf"),
        "dtype": "float64",
    },
    action_space={
        "type": "Box",
        "shape": [8],
        "low": [-1.0] * 8,
        "high": [1.0] * 8,
        "dtype": "float32",
    },
    max_episode_steps=1000,
    expert_baseline=6000.0,
    random_baseline=-50.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
