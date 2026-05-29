"""Pendulum-from-Pixels env registration (visual Pendulum-v1).

Same swing-up + stabilize task as ``pendulum``, but the agent sees a
64x64x3 RGB rendering of the pendulum instead of the state vector
``[cos, sin, dot]``. The agent must extract physics state (angle and
angular velocity) from images.

A single frame contains the angle but not the angular velocity —
agents need at least a 2-frame history (or maintained internal state
across ``act()`` calls) to compute velocity.

Uses ``obs_storage="external"``: per-step 64x64x3 frames are written
to ``observations.npy`` side-cars (3 KB per step inline JSON would
just barely fit, but we use external for consistency with the other
pixel envs and to test the infra).

Side effect: importing this module registers ``pendulum_from_pixels``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

OBS_H: int = 64
OBS_W: int = 64


def _factory() -> object:
    import gymnasium

    base = gymnasium.make("Pendulum-v1", render_mode="rgb_array")

    class _PendulumPixelWrapper(gymnasium.Wrapper[Any, Any, Any, Any]):
        """Replace state-vector obs with rendered RGB image, downsampled to 64x64."""

        def __init__(self, env: gymnasium.Env[Any, Any]) -> None:
            super().__init__(env)
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=(OBS_H, OBS_W, 3), dtype=np.uint8,
            )

        def _render_obs(self) -> np.ndarray:
            """Render and downsample to OBS_H x OBS_W."""
            frame = self.env.render()
            assert frame is not None, "Pendulum env returned no frame"
            # Pendulum default render is 500x500x3. Block-average down to OBS_H x OBS_W.
            arr = np.asarray(frame, dtype=np.uint8)
            h, w, c = arr.shape
            block_h = h // OBS_H
            block_w = w // OBS_W
            arr = arr[: block_h * OBS_H, : block_w * OBS_W]
            ds = (
                arr.reshape(OBS_H, block_h, OBS_W, block_w, c)
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
            _, info = self.env.reset(seed=seed, options=options)
            return self._render_obs(), info

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            _, r, term, trunc, info = self.env.step(action)
            return self._render_obs(), float(r), bool(term), bool(trunc), info

    return _PendulumPixelWrapper(base)


register_env(
    env_id="pendulum_from_pixels",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [OBS_H, OBS_W, 3],
        "low": 0,
        "high": 255,
        "dtype": "uint8",
    },
    action_space={
        "type": "Box",
        "shape": [1],
        "low": [-2.0],
        "high": [2.0],
        "dtype": "float32",
    },
    max_episode_steps=200,
    # Random ≈ -1200 (same as state-based Pendulum). Expert ≈ -200
    # (slightly worse than state-based -150 because pixel extraction
    # adds noise + 1-frame velocity inference is imperfect).
    expert_baseline=-200.0,
    random_baseline=-1200.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="external",
    reward_components=None,
)
