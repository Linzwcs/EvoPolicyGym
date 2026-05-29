"""Lunar-Hardcore env registration (v1 roster #15).

LunarLanderContinuous-v3 with wind + turbulence enabled and their
strengths randomized per ``reset(seed=...)``. Train pool sees mild
wind; held-out pool sees stronger wind disjoint from train range.

A hand-tuned PID that lands cleanly without wind will drift and crash
on stronger held-out wind — unless the policy reads obs velocity
and angle to react adaptively.

Gravity is left at gymnasium's default (-10 m/s²). Per-seed gravity
randomization would require Box2D world rebuild and is deferred.

Side effect: importing this module registers ``lunar_hardcore``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

# Train: mild wind. Held-out: stronger wind (disjoint).
_TRAIN_RANGES: dict[str, tuple[float, float]] = {
    "wind_power":       (10.0, 15.0),
    "turbulence_power": (1.0, 1.5),
}
_HELDOUT_RANGES: dict[str, tuple[float, float]] = {
    "wind_power":       (15.0, 20.0),   # stronger (disjoint upper)
    "turbulence_power": (1.5, 2.0),     # more turbulent (disjoint upper)
}

_HELDOUT_SEED_FLOOR: int = 1_000_000


def _sample_params(seed: int) -> tuple[float, float]:
    """Deterministic (wind_power, turbulence_power) for a real seed.

    Train pool seeds (< floor) → train range; held-out (>= floor) →
    disjoint OOD range. The wrapper applies these in ``reset()``.

    Gymnasium's LunarLander reads ``self.wind_power`` and
    ``self.turbulence_power`` inside ``step()`` for the wind force
    integration, so reassigning between resets is sufficient.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    ranges = _HELDOUT_RANGES if seed >= _HELDOUT_SEED_FLOOR else _TRAIN_RANGES
    wp = float(rng.uniform(*ranges["wind_power"]))
    tp = float(rng.uniform(*ranges["turbulence_power"]))
    return wp, tp


def _factory() -> object:
    """Wrapped LunarLanderContinuous-v3 with wind enabled and per-seed
    wind / turbulence strength.
    """
    import gymnasium

    # enable_wind=True is the env-construction switch; specific powers
    # are reassigned per reset.
    base_env = gymnasium.make(
        "LunarLanderContinuous-v3",
        render_mode=None,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    class _DomainRandomizedLunar(gymnasium.Wrapper[Any, Any, Any, Any]):
        """Reassigns wind_power and turbulence_power on each reset(seed=)."""

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> Any:
            if seed is not None:
                wp, tp = _sample_params(seed)
                inner = self.env.unwrapped
                inner.wind_power = wp          # type: ignore[attr-defined]
                inner.turbulence_power = tp    # type: ignore[attr-defined]
            return self.env.reset(seed=seed, options=options)

    return _DomainRandomizedLunar(base_env)


register_env(
    env_id="lunar_hardcore",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [8],
        "low": [-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0],
        "high": [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0],
        "dtype": "float32",
    },
    action_space={
        "type": "Box",
        "shape": [2],
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
        "dtype": "float32",
    },
    max_episode_steps=1000,
    # With wind enabled, baselines shift. Approximate; calibrate.
    # Random ≈ -200 (crashes faster with wind). Expert ≈ +150 (lower
    # than vanilla +200 because wind makes precision landing harder).
    expert_baseline=150.0,
    random_baseline=-200.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
