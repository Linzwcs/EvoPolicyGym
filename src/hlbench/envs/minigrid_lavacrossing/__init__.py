"""MiniGrid-LavaCrossingS11N5-v0 env registration (MiniGrid-LavaCrossingS11N5-v0).

MiniGrid task: Cross 5 lava rivers without stepping on lava to reach the goal in an 11x11 grid.

Wraps the Gymnasium MiniGrid env. MiniGrid returns a Dict obs
``{image: (7,7,3) uint8, direction: int, mission: str}``. Our
wrapper packs each cell's (type, color, state) into a single
uint16 (``type * 100 + color * 10 + state``, max ~1200) and
flattens to ``Box(0, 1500, (50,))``: ``packed_image.flatten() +
[direction]``. Information-preserving (no channel loss); the
mission string is static per env (see TASK.md) and not in obs.

Side effect: importing this module registers ``minigrid_lavacrossing``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

OBS_DIM: int = 7 * 7 + 1  # 50: 49 packed cells + 1 direction


def _factory() -> object:
    import gymnasium
    import minigrid  # noqa: F401 — registers MiniGrid envs

    base = gymnasium.make("MiniGrid-LavaCrossingS11N5-v0", render_mode=None)

    class _MiniGridPackedWrapper(gymnasium.Wrapper[Any, Any, Any, Any]):
        """Pack MiniGrid Dict obs into a uint16 Box(50,).

        Each of 49 cells gets a single uint16 = type * 100 + color * 10 + state.
        Position 49 is the agent direction (0-3).
        """

        def __init__(self, env: gymnasium.Env[Any, Any]) -> None:
            super().__init__(env)
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=1500, shape=(OBS_DIM,), dtype=np.uint16,
            )

        def _pack(self, obs: dict[str, Any]) -> np.ndarray:
            img = np.asarray(obs["image"], dtype=np.uint16)  # (7, 7, 3)
            packed = (
                img[:, :, 0] * 100 +  # type
                img[:, :, 1] * 10 +   # color
                img[:, :, 2]           # state
            ).flatten()  # (49,)
            direction = np.array([obs["direction"]], dtype=np.uint16)
            return np.concatenate([packed, direction]).astype(np.uint16)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            obs, info = self.env.reset(seed=seed, options=options)
            return self._pack(obs), info

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            obs, r, term, trunc, info = self.env.step(action)
            return self._pack(obs), float(r), bool(term), bool(trunc), info

    return _MiniGridPackedWrapper(base)


register_env(
    env_id="minigrid_lavacrossing",
    env_version="0.2",  # bumped from 0.1 — obs encoding changed
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [OBS_DIM],
        "low": 0,
        "high": 1500,
        "dtype": "uint16",
    },
    action_space={
        "type": "Discrete",
        "n": 7,
    },
    max_episode_steps=880,
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
