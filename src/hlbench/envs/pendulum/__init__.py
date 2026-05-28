"""Pendulum-v1 env registration.

Side effect: importing this module registers `pendulum` with the registry.
"""

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    """Lazy import gymnasium to avoid forcing it as a hard dep at server init."""
    import gymnasium

    return gymnasium.make("Pendulum-v1", render_mode=None)


register_env(
    env_id="pendulum",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [3],
        "low": [-1.0, -1.0, -8.0],
        "high": [1.0, 1.0, 8.0],
        "dtype": "float32",
    },
    action_space={
        "type": "Box",
        "shape": [1],
        "low": [-2.0],
        "high": [2.0],
        "dtype": "float32",
    },
    max_episode_steps=200,
    # Server-internal scoring references. Not exposed to agent.
    # Random ~ -1200 (uniform action), expert ~ -150 (LQR / good PD).
    expert_baseline=-150.0,
    random_baseline=-1200.0,
    train_seeds_path=_HERE / "train.json",
    heldout_seeds_path=_HERE / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    n_env_instances=256,
    obs_storage="inline",
    reward_components=None,
)
