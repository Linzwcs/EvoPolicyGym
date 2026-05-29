"""Smoke tests for the 3 hardcore env variants (v1 roster #14-16).

Coverage:
- Registration metadata (all 3 envs)
- Per-seed wrapper for pendulum_hardcore (in-process: no Box2D needed)
- Per-seed sampler for lunar_hardcore (function call only, no factory)
- Seed pool split-by-floor invariant (file inspection only)
- Bipedal_hardcore registration (no factory call — Box2D in pytest's
  main process segfaults on macOS; the env is exercised end-to-end via
  Sandbox-spawned subprocesses in integration tests)

Box2D's macOS wheel segfaults when its C extension is imported in
pytest's main process. Existing tests for ``bipedal_walker`` and
``lunar_lander_continuous`` work because they invoke ``SubmitHandler``
which spawns sandbox subprocesses where the Box2D import succeeds.
We follow the same pattern here.
"""

from __future__ import annotations

import pytest

from hlbench.envs.registry import get_env


@pytest.fixture(autouse=True)
def _ensure_envs_registered() -> None:
    """Importing hlbench.envs eagerly registers all envs (side effect)."""
    import hlbench.envs  # noqa: F401


HARDCORE_ENVS = ["pendulum_hardcore", "lunar_hardcore", "bipedal_hardcore"]


@pytest.mark.parametrize("env_id", HARDCORE_ENVS)
def test_hardcore_env_registered(env_id: str) -> None:
    """Each hardcore env must register cleanly with correct metadata."""
    d = get_env(env_id)
    assert d.env_id == env_id
    assert d.env_version == "0.1"
    assert d.n_env_instances == 10000
    assert d.obs_storage == "inline"
    assert d.train_seeds_path.exists()
    assert d.heldout_seeds_path.exists()
    assert d.starter_policy_path is not None and d.starter_policy_path.exists()
    assert d.task_md_path is not None and d.task_md_path.exists()


def test_pendulum_hardcore_per_seed_randomization() -> None:
    """Train seeds → train (m, l, g) ranges; held-out → disjoint OOD.

    Pendulum has no Box2D dependency, so we can safely call factory() +
    reset() in pytest's main process.
    """
    from hlbench.envs.pendulum_hardcore import (
        _HELDOUT_RANGES,
        _HELDOUT_SEED_FLOOR,
        _TRAIN_RANGES,
    )

    d = get_env("pendulum_hardcore")
    env = d.factory()
    try:
        # Train seeds: 100 samples must land within train ranges
        for seed in range(0, 1000, 10):
            assert seed < _HELDOUT_SEED_FLOOR
            env.reset(seed=seed)
            inner = env.unwrapped
            for axis, attr in (("mass", "m"), ("length", "l"), ("gravity", "g")):
                lo, hi = _TRAIN_RANGES[axis]
                v = float(getattr(inner, attr))
                assert lo <= v <= hi, f"seed {seed} {axis}={v} outside train [{lo}, {hi}]"

        # Held-out seeds: must land within held-out (OOD) ranges
        for seed in range(_HELDOUT_SEED_FLOOR, _HELDOUT_SEED_FLOOR + 1000, 10):
            env.reset(seed=seed)
            inner = env.unwrapped
            for axis, attr in (("mass", "m"), ("length", "l"), ("gravity", "g")):
                lo, hi = _HELDOUT_RANGES[axis]
                v = float(getattr(inner, attr))
                assert lo <= v <= hi, f"seed {seed} {axis}={v} outside held-out [{lo}, {hi}]"
    finally:
        env.close()


def test_lunar_hardcore_sampler_function() -> None:
    """Sampler function maps seeds → params correctly. No factory call —
    Box2D import would segfault in pytest's main process on macOS."""
    from hlbench.envs.lunar_hardcore import (
        _HELDOUT_RANGES,
        _HELDOUT_SEED_FLOOR,
        _TRAIN_RANGES,
        _sample_params,
    )

    for seed in range(0, 1000, 10):
        assert seed < _HELDOUT_SEED_FLOOR
        wp, tp = _sample_params(seed)
        wp_lo, wp_hi = _TRAIN_RANGES["wind_power"]
        tp_lo, tp_hi = _TRAIN_RANGES["turbulence_power"]
        assert wp_lo <= wp <= wp_hi, f"seed {seed} wind_power={wp} outside train"
        assert tp_lo <= tp <= tp_hi, f"seed {seed} turbulence_power={tp} outside train"

    for seed in range(_HELDOUT_SEED_FLOOR, _HELDOUT_SEED_FLOOR + 1000, 10):
        wp, tp = _sample_params(seed)
        wp_lo, wp_hi = _HELDOUT_RANGES["wind_power"]
        tp_lo, tp_hi = _HELDOUT_RANGES["turbulence_power"]
        assert wp_lo <= wp <= wp_hi, f"seed {seed} wind_power={wp} outside held-out"
        assert tp_lo <= tp <= tp_hi, f"seed {seed} turbulence_power={tp} outside held-out"


def test_pendulum_hardcore_sampler_disjoint_ranges() -> None:
    """Train and held-out parameter ranges must be disjoint on every axis."""
    from hlbench.envs.pendulum_hardcore import _HELDOUT_RANGES, _TRAIN_RANGES

    for axis in _TRAIN_RANGES:
        t_lo, t_hi = _TRAIN_RANGES[axis]
        h_lo, h_hi = _HELDOUT_RANGES[axis]
        # Adjacent-but-disjoint: train_hi <= heldout_lo OR heldout_hi <= train_lo
        assert t_hi <= h_lo or h_hi <= t_lo, (
            f"pendulum_hardcore {axis}: train [{t_lo}, {t_hi}] overlaps "
            f"heldout [{h_lo}, {h_hi}]"
        )


def test_lunar_hardcore_sampler_disjoint_ranges() -> None:
    """Train and held-out parameter ranges must be disjoint."""
    from hlbench.envs.lunar_hardcore import _HELDOUT_RANGES, _TRAIN_RANGES

    for axis in _TRAIN_RANGES:
        t_lo, t_hi = _TRAIN_RANGES[axis]
        h_lo, h_hi = _HELDOUT_RANGES[axis]
        assert t_hi <= h_lo or h_hi <= t_lo, (
            f"lunar_hardcore {axis}: train [{t_lo}, {t_hi}] overlaps "
            f"heldout [{h_lo}, {h_hi}]"
        )


def test_hardcore_seed_pools_split_by_floor() -> None:
    """For pendulum and lunar: train.json seeds < floor, heldout.json >= floor.

    Bipedal_hardcore doesn't use the floor convention (it relies on
    BipedalWalkerHardcore-v3's built-in procedural terrain rather than
    parameter randomization), so it's excluded.
    """
    import json

    for env_id in ["pendulum_hardcore", "lunar_hardcore"]:
        d = get_env(env_id)
        train = json.loads(d.train_seeds_path.read_text())["real_seeds"]
        heldout = json.loads(d.heldout_seeds_path.read_text())["real_seeds"]

        assert max(train) < 1_000_000, f"{env_id}: train seed exceeds floor"
        assert min(heldout) >= 1_000_000, f"{env_id}: held-out seed below floor"
        assert set(train).isdisjoint(set(heldout)), f"{env_id}: train/heldout overlap"


def test_bipedal_hardcore_factory_targets_correct_gym_id() -> None:
    """Bipedal_hardcore's factory must reference BipedalWalkerHardcore-v3
    (not BipedalWalker-v3). Verified via source inspection rather than
    factory call to avoid in-process Box2D loading."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "src/hlbench/envs/bipedal_hardcore/__init__.py"
    text = src.read_text()
    assert "BipedalWalkerHardcore-v3" in text
    assert '"BipedalWalker-v3"' not in text  # must not target the easy variant
