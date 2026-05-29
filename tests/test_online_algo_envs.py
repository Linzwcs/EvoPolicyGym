"""Smoke tests for the 3 online algorithm envs (v1 roster #11-13).

Coverage:
- Registration metadata (cache_replacement, k_server, online_bipartite_matching)
- Factory + reset + step round-trip (no Box2D — these are pure-Python
  envs with no segfault risk; we can run them in pytest's main process)
- Train/held-out distribution divergence detected by running both
  pools and comparing the trace generators
- Determinism: same seed → identical trace
"""

from __future__ import annotations

import numpy as np
import pytest

from hlbench.envs.registry import get_env


@pytest.fixture(autouse=True)
def _ensure_envs_registered() -> None:
    import hlbench.envs  # noqa: F401


ONLINE_ALGO_ENVS = ["cache_replacement", "k_server", "online_bipartite_matching"]


@pytest.mark.parametrize("env_id", ONLINE_ALGO_ENVS)
def test_online_algo_env_registered(env_id: str) -> None:
    d = get_env(env_id)
    assert d.env_id == env_id
    assert d.env_version == "0.1"
    assert d.n_env_instances == 10000
    assert d.obs_storage == "inline"
    assert d.train_seeds_path.exists()
    assert d.heldout_seeds_path.exists()
    assert d.starter_policy_path is not None and d.starter_policy_path.exists()
    assert d.task_md_path is not None and d.task_md_path.exists()


@pytest.mark.parametrize("env_id", ONLINE_ALGO_ENVS)
def test_online_algo_env_factory_step(env_id: str) -> None:
    """Factory + reset + step round-trip (pure-Python envs are safe
    in pytest's main process)."""
    d = get_env(env_id)
    env = d.factory()
    try:
        obs, _info = env.reset(seed=42)
        assert tuple(obs.shape) == tuple(d.obs_space["shape"])
        # Take a couple steps with sampled actions
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(int(action))
            assert tuple(obs.shape) == tuple(d.obs_space["shape"])
            assert isinstance(reward, float)
            assert not terminated  # episodes only truncate, never terminate
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ONLINE_ALGO_ENVS)
def test_online_algo_env_determinism(env_id: str) -> None:
    """Same seed → same trace. Take 5 steps with action=0; trajectories
    must match across two reset+step sequences."""
    d = get_env(env_id)
    env1 = d.factory()
    env2 = d.factory()
    try:
        env1.reset(seed=123)
        env2.reset(seed=123)
        for _ in range(5):
            o1, r1, *_ = env1.step(0)
            o2, r2, *_ = env2.step(0)
            assert np.array_equal(o1, o2)
            assert r1 == r2
    finally:
        env1.close()
        env2.close()


def test_cache_replacement_train_vs_heldout_distribution() -> None:
    """Train traces (Zipfian) have higher locality than held-out scans.

    Measure: in the first 100 accesses, count how many are repeats.
    Zipfian: small set of hot objects → many repeats early.
    Scan: cycles through permutation → few/no early repeats."""
    from hlbench.envs.cache_replacement import _generate_trace

    train_trace = _generate_trace(seed=42, n_steps=200, n_objects=64)
    heldout_trace = _generate_trace(seed=1_000_042, n_steps=200, n_objects=64)

    train_unique = len(set(train_trace[:100].tolist()))
    heldout_unique = len(set(heldout_trace[:100].tolist()))
    # Zipfian first-100 has fewer unique objects than scan first-100.
    # (Strict inequality is statistical but holds for any reasonable seed
    # at n_objects=64.)
    assert train_unique < heldout_unique, (
        f"expected train to have fewer unique IDs in first 100 accesses; "
        f"got train={train_unique} heldout={heldout_unique}"
    )


def test_k_server_train_vs_heldout_distribution() -> None:
    """Held-out concentrates 75% of requests at one corner; train spreads
    across two centers. Verify by counting requests in the
    [0.5, 1] x [0.5, 1] quadrant."""
    from hlbench.envs.k_server import _generate_requests

    train_reqs = _generate_requests(seed=42, n_requests=200)
    heldout_reqs = _generate_requests(seed=1_000_042, n_requests=200)

    def upper_right_fraction(reqs: np.ndarray) -> float:
        in_quadrant = (reqs[:, 0] > 0.5) & (reqs[:, 1] > 0.5)
        return float(in_quadrant.sum() / len(reqs))

    train_frac = upper_right_fraction(train_reqs)
    heldout_frac = upper_right_fraction(heldout_reqs)
    # Train: ~0% in the upper-right (centers at ±0.4). Held-out: ~75%.
    assert heldout_frac > 0.5, (
        f"held-out should concentrate in upper-right corner; got "
        f"{heldout_frac:.2f}"
    )
    assert train_frac < 0.1, f"train should not concentrate; got {train_frac:.2f}"


def test_online_matching_train_vs_heldout_structure() -> None:
    """Held-out arrivals show structured asymmetry (first half connects
    to left half only); train is uniform random."""
    from hlbench.envs.online_bipartite_matching import N_LEFT, _generate_arrivals

    train_arr = _generate_arrivals(seed=42)
    heldout_arr = _generate_arrivals(seed=1_000_042)

    half = N_LEFT // 2
    # Held-out: first M/2 arrivals must NOT touch the right half of left
    # vertices. Train: should touch both halves more uniformly.
    heldout_first_half_right_half_edges = heldout_arr[: heldout_arr.shape[0] // 2, half:].sum()
    assert heldout_first_half_right_half_edges == 0, (
        "held-out: first M/2 arrivals must not have edges to right half of "
        f"left vertices, got {heldout_first_half_right_half_edges}"
    )
    train_first_half_right_half_edges = train_arr[: train_arr.shape[0] // 2, half:].sum()
    assert train_first_half_right_half_edges > 0, (
        "train: first M/2 arrivals should have some edges to right half"
    )


@pytest.mark.parametrize("env_id", ONLINE_ALGO_ENVS)
def test_online_algo_seed_pools_split_by_floor(env_id: str) -> None:
    """train.json seeds < floor; heldout.json seeds >= floor; disjoint."""
    import json

    d = get_env(env_id)
    train = json.loads(d.train_seeds_path.read_text())["real_seeds"]
    heldout = json.loads(d.heldout_seeds_path.read_text())["real_seeds"]

    assert max(train) < 1_000_000, f"{env_id}: train seed exceeds floor"
    assert min(heldout) >= 1_000_000, f"{env_id}: held-out seed below floor"
    assert set(train).isdisjoint(set(heldout)), f"{env_id}: train/heldout overlap"
