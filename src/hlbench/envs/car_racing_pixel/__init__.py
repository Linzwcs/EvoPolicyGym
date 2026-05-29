"""Car-Racing-Pixel env registration (full 96x96x3 CarRacing-v3).

This is the full-resolution variant of CarRacing, in contrast to
``car_racing`` (which down-samples to 16x16 to fit inline obs).

Uses ``obs_storage="external"``: per-step 96x96x3 uint8 frames are
written to ``feedback/submit_NNN/episodes/ep_<XXX>/observations.npy``
side-car files; ``trajectory.jsonl`` carries ``"obs": null`` for
each step. The agent reads observations via numpy:

    obs = np.load("feedback/submit_005/episodes/ep_042/observations.npy")
    frame_100 = obs[100]   # shape (96, 96, 3) uint8

Side effect: importing this module registers ``car_racing_pixel``.
"""

from __future__ import annotations

from pathlib import Path

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent


def _factory() -> object:
    import gymnasium

    return gymnasium.make("CarRacing-v3", render_mode=None, continuous=True)


register_env(
    env_id="car_racing_pixel",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [96, 96, 3],
        "low": [[[0] * 3] * 96] * 96,
        "high": [[[255] * 3] * 96] * 96,
        "dtype": "uint8",
    },
    action_space={
        "type": "Box",
        "shape": [3],
        # [steering, gas, brake]
        "low": [-1.0, 0.0, 0.0],
        "high": [1.0, 1.0, 1.0],
        "dtype": "float32",
    },
    max_episode_steps=1000,
    # Random ≈ -100; expert (~RL trained) ≈ +900 on full resolution.
    expert_baseline=900.0,
    random_baseline=-100.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="external",   # 96*96*3 = 27,648 bytes per step, too big for inline JSON
    reward_components=None,
)
