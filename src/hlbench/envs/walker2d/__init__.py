"""Walker2d-v5 env registration (Walker2d-v5).

MuJoCo locomotion task: Walk forward on two legs without falling.

Direct wrap of Gymnasium's ``Walker2d-v5``. Per-seed variation comes from
the env's RNG (initial joint perturbations); no domain-randomization
wrapper is applied. Held-out seeds therefore differ only in initial
state, not in physics parameters — generalization here is "robustness
to initial-state distribution" rather than parameter OOD.

Side effect: importing this module registers ``walker2d``.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("Walker2d-v5", render_mode=None)


register_env(
    env_id="walker2d",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [17],
        "low": -float("inf"),
        "high": float("inf"),
        "dtype": "float64",
    },
    action_space={
        "type": "Box",
        "shape": [6],
        "low": [-1.0] * 6,
        "high": [1.0] * 6,
        "dtype": "float32",
    },
    max_episode_steps=1000,
    expert_baseline=4500.0,
    random_baseline=5.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
