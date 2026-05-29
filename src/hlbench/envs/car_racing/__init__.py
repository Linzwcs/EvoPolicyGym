"""Car-Racing-lite env registration (downsampled CarRacing-v3).

Wraps Gymnasium's ``CarRacing-v3`` with a 96x96x3 -> 16x16x3
nearest-neighbor downsample (PIL not required — pure numpy slicing).
The downsampled obs fits inline in trajectory.jsonl (~3 KB serialized
per step, well under the 10 KB cap).

This is a "lite" variant. The full 96x96 CarRacing requires
``observations.npy`` side-car storage (SPEC §4.6), which is deferred
to a separate infrastructure PR. We ship the downsampled version so
the visual-control category has at least one representative in v1
and so the LLM can be evaluated on color-based road following with
limited spatial resolution.

The agent sees enough information to detect:
- Road vs grass (color)
- Approximate road direction (gradient of road pixels)
- Lateral position relative to road center

But not enough to read precise track geometry far ahead.

Side effect: importing this module registers ``car_racing``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

DOWNSAMPLE_H: int = 16
DOWNSAMPLE_W: int = 16
OBS_SHAPE: tuple[int, int, int] = (DOWNSAMPLE_H, DOWNSAMPLE_W, 3)


def _factory() -> object:
    import gymnasium

    base = gymnasium.make(
        "CarRacing-v3",
        render_mode=None,
        continuous=True,
    )

    class _DownsampleWrapper(gymnasium.Wrapper[Any, Any, Any, Any]):
        """96x96x3 -> DOWNSAMPLE_H x DOWNSAMPLE_W x 3 via uniform
        block-averaging (pure numpy).
        """

        def __init__(self, env: gymnasium.Env[Any, Any]) -> None:
            super().__init__(env)
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8,
            )

        def _downsample(self, obs: np.ndarray) -> np.ndarray:
            # CarRacing returns 96x96x3 uint8. Block-average 6x6 -> 1.
            h, w, c = obs.shape
            block_h = h // DOWNSAMPLE_H
            block_w = w // DOWNSAMPLE_W
            # Crop to even-divisible region
            obs = obs[: block_h * DOWNSAMPLE_H, : block_w * DOWNSAMPLE_W]
            # Reshape and mean
            ds = (
                obs.reshape(DOWNSAMPLE_H, block_h, DOWNSAMPLE_W, block_w, c)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )
            return ds  # type: ignore[no-any-return]

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            obs, info = self.env.reset(seed=seed, options=options)
            return self._downsample(obs), info

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            obs, r, term, trunc, info = self.env.step(action)
            return self._downsample(obs), float(r), bool(term), bool(trunc), info

    return _DownsampleWrapper(base)


register_env(
    env_id="car_racing",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [DOWNSAMPLE_H, DOWNSAMPLE_W, 3],
        "low": 0,
        "high": 255,
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
    # Random ≈ -100 (drives off track immediately, accumulates negative
    # reward). Expert ≈ +900 (consistent track completion). With
    # 16x16 downsampling, expert ceiling is lower — call it +500.
    # Approximate; calibrate.
    expert_baseline=500.0,
    random_baseline=-100.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
