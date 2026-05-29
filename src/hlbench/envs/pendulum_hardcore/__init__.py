"""Pendulum-Hardcore env registration (v1 roster #14).

Domain-randomized Pendulum-v1: mass, length, and gravity are sampled
per ``reset(seed=...)`` from declared ranges. Train pool draws from
nominal ranges; held-out pool draws from OOD ranges (heavier mass,
longer rod, stronger gravity).

A fixed-gain PD that wins on the original Pendulum will fail here on
held-out OOD seeds. Agent must either:
  - Read parameter ranges from ``TASK.md`` / ``env_meta.extras`` and
    schedule gains as a function of (m, l, g).
  - Do brief online identification at episode start.
  - Use a sufficiently robust controller (energy shaping with adaptive
    gain).

Specific seed → (m, l, g) values are NOT exposed — only the *ranges*
the seed pool was drawn from. See ``TASK.md``.

Side effect: importing this module registers ``pendulum_hardcore``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

# Train pool: nominal parameter ranges. Held-out: disjoint OOD ranges
# (heavier mass, longer rod, weaker gravity — all individually harder).
# By disjointness, an overfit-to-train policy provably fails on held-out.
_TRAIN_RANGES: dict[str, tuple[float, float]] = {
    "mass":    (0.5, 2.0),
    "length":  (0.7, 1.5),
    "gravity": (8.0, 12.0),
}
_HELDOUT_RANGES: dict[str, tuple[float, float]] = {
    "mass":    (2.0, 3.5),   # heavier (disjoint upper from train)
    "length":  (1.5, 2.2),   # longer (disjoint upper)
    "gravity": (4.0, 8.0),   # weaker (disjoint lower — harder swing-up)
}

# Seed-magnitude convention: train seeds in [0, _HELDOUT_SEED_FLOOR),
# held-out seeds in [_HELDOUT_SEED_FLOOR, ...). The wrapper uses this
# to dispatch to the correct range. Seed pool files (data/train.json,
# data/heldout.json) MUST respect this split.
_HELDOUT_SEED_FLOOR: int = 1_000_000


def _sample_params(seed: int) -> tuple[float, float, float]:
    """Deterministic (m, l, g) for a real seed.

    Train seeds map to ``_TRAIN_RANGES``; held-out seeds (>= floor) map
    to ``_HELDOUT_RANGES`` (disjoint OOD). The wrapper applies these in
    ``reset()`` — Gymnasium's PendulumEnv reads ``self.m / .l / .g`` in
    every step's dynamics integration, so reassigning between resets
    is sufficient.

    The split is encoded in the seed value itself (rather than tracked
    by the wrapper) so the same wrapper handles both train and heldout
    correctly without knowing which pool a seed came from.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    ranges = _HELDOUT_RANGES if seed >= _HELDOUT_SEED_FLOOR else _TRAIN_RANGES
    mass = float(rng.uniform(*ranges["mass"]))
    length = float(rng.uniform(*ranges["length"]))
    gravity = float(rng.uniform(*ranges["gravity"]))
    return mass, length, gravity


def _factory() -> object:
    """Wrapped Pendulum-v1 with per-seed (mass, length, gravity) sampling."""
    import gymnasium

    base_env = gymnasium.make("Pendulum-v1", render_mode=None)

    class _DomainRandomizedPendulum(gymnasium.Wrapper[Any, Any, Any, Any]):
        """Reassigns Pendulum's m / l / g on each reset(seed=...).

        Gymnasium's PendulumEnv reads ``self.m``, ``self.l``, ``self.g``
        in ``step()`` for every dynamics integration, so reassigning
        them between resets is sufficient (no Box2D world rebuild
        needed).
        """

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> Any:
            if seed is not None:
                mass, length, gravity = _sample_params(seed)
                inner = self.env.unwrapped
                inner.m = mass     # type: ignore[attr-defined]
                inner.l = length   # type: ignore[attr-defined]
                inner.g = gravity  # type: ignore[attr-defined]
            return self.env.reset(seed=seed, options=options)

    return _DomainRandomizedPendulum(base_env)


register_env(
    env_id="pendulum_hardcore",
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
    # Random baseline ≈ -1700 (worse than vanilla pendulum because
    # heavier rod + stronger g amplify error). Expert ≈ -200 (good
    # adaptive controller; harder to hit -150 like vanilla).
    # Approximate; calibrate before paper submission.
    expert_baseline=-200.0,
    random_baseline=-1700.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
    # extras: exposed in EnvDefinition.extras (server-internal for v1;
    # agents read ranges from TASK.md instead).
    train_param_ranges=_TRAIN_RANGES,
    heldout_param_ranges=_HELDOUT_RANGES,
)
