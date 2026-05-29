"""MiniGrid-KeyCorridorS6R3-v0 env registration (MiniGrid-KeyCorridorS6R3-v0).

MiniGrid task: Navigate a corridor of 3 rooms per side; locked door require

Wraps the Gymnasium MiniGrid env. MiniGrid returns a Dict obs
``{image: (7,7,3) uint8, direction: int, mission: str}``. Our
wrapper flattens this to a uint8 ``Box(0, 255, (148,))``:
``image.flatten() + [direction]``. The mission string is static
per env (see TASK.md) and not included in obs.

Side effect: importing this module registers ``minigrid_keycorridor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

OBS_DIM: int = 7 * 7 * 3 + 1  # 148


def _factory() -> object:
    import gymnasium
    import minigrid  # noqa: F401 — registers MiniGrid envs

    base = gymnasium.make("MiniGrid-KeyCorridorS6R3-v0", render_mode=None)

    class _MiniGridFlattenWrapper(gymnasium.Wrapper[Any, Any, Any, Any]):
        """Flatten MiniGrid Dict obs to a uint8 Box(148,)."""

        def __init__(self, env: gymnasium.Env[Any, Any]) -> None:
            super().__init__(env)
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=(OBS_DIM,), dtype=np.uint8,
            )

        def _flatten(self, obs: dict[str, Any]) -> np.ndarray:
            img = np.asarray(obs["image"], dtype=np.uint8).flatten()
            direction = np.array([obs["direction"]], dtype=np.uint8)
            return np.concatenate([img, direction]).astype(np.uint8)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            obs, info = self.env.reset(seed=seed, options=options)
            return self._flatten(obs), info

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            obs, r, term, trunc, info = self.env.step(action)
            return self._flatten(obs), float(r), bool(term), bool(trunc), info

    return _MiniGridFlattenWrapper(base)


register_env(
    env_id="minigrid_keycorridor",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [OBS_DIM],
        "low": [0] * OBS_DIM,
        "high": [255] * OBS_DIM,
        "dtype": "uint8",
    },
    action_space={
        "type": "Discrete",
        "n": 7,
    },
    max_episode_steps=1080,
    expert_baseline=0.9,
    random_baseline=0.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
