"""Cache-Replacement env registration (v1 roster #11).

Stream of memory accesses with capacity-bounded cache; on miss, the
policy chooses which cached object to evict. Classical online-algorithm
territory (LRU, LFU, ARC, LIRS, Belady's OPT) with strong textbook
baselines that the LLM is expected to know.

Train pool: Zipfian-distributed access traces (high locality;
LRU/LFU work well, ARC near-optimal).
Held-out pool: scan-heavy traces where the working set exceeds cache
capacity, defeating LRU (it evicts items right before they're reused).
Held-out generalization requires the policy to either recognize the
distribution shift and adapt, or to use a strategy robust to both.

Seed-magnitude convention: train in [0, 1_000_000); held-out in
[1_000_000, 2_000_000). The env reads the seed in reset() to decide
which trace distribution to generate.

Side effect: importing this module registers ``cache_replacement``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

# Config constants exposed for tests + TASK.md consistency.
CACHE_CAPACITY: int = 8
HISTORY_WINDOW: int = 8
N_OBJECTS: int = 64
TRACE_LENGTH: int = 500

_HELDOUT_SEED_FLOOR: int = 1_000_000


def _generate_trace(seed: int, n_steps: int, n_objects: int) -> np.ndarray:
    """Deterministic access trace for a given seed.

    Train seeds (< floor) produce Zipfian (high-locality, LRU-friendly).
    Held-out seeds (>= floor) produce scanning traces (cycles through
    a random permutation of all objects, LRU-hostile because every
    access is to the object LRU just evicted)."""
    rng = np.random.default_rng(seed)
    if seed >= _HELDOUT_SEED_FLOOR:
        # Scan-heavy: cycle through a permutation
        perm = rng.permutation(n_objects)
        # Add some noise (10% random Zipfian over the same N objects)
        n_noise = n_steps // 10
        n_scan = n_steps - n_noise
        scan_part = np.tile(perm, (n_scan + n_objects - 1) // n_objects)[:n_scan]
        noise_part = (rng.zipf(1.5, size=n_noise) - 1) % n_objects
        trace = np.concatenate([scan_part, noise_part])
        # Shuffle the noise into the scan
        idx = rng.permutation(n_steps)
        trace = trace[idx]
    else:
        # Zipfian: small set of hot objects accessed most often
        trace = (rng.zipf(1.5, size=n_steps) - 1) % n_objects
    return trace.astype(np.int32)


class CacheReplacementEnv:
    """Custom Gymnasium-compatible env (duck-typed; reset/step/spec).

    Observation: int32 vector of length CACHE_CAPACITY + HISTORY_WINDOW + 1
      [cache_slots, recent_access_history, current_access]
      -1 padding for empty cache slots / history slots before they fill.

    Action: Discrete(CACHE_CAPACITY) — slot index to evict (ignored on
    cache hit but must still be a valid value).

    Reward: +1.0 per cache hit, 0.0 per miss.
    Terminates after TRACE_LENGTH steps (truncated=True).
    """

    metadata: dict[str, Any] = {"render_modes": []}
    spec = None  # set by gymnasium.make wrappers; we don't use it

    def __init__(self) -> None:
        import gymnasium as gym
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=N_OBJECTS - 1,
            shape=(CACHE_CAPACITY + HISTORY_WINDOW + 1,),
            dtype=np.int32,
        )
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(CACHE_CAPACITY)  # type: ignore[type-arg]
        self._trace: np.ndarray | None = None
        self._step: int = 0
        self._cache: list[int] = []
        self._history: list[int] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = 0
        self._trace = _generate_trace(seed, TRACE_LENGTH, N_OBJECTS)
        self._step = 0
        self._cache = []
        self._history = []
        return self._make_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._trace is not None
        access = int(self._trace[self._step])
        is_hit = access in self._cache
        if is_hit:
            reward = 1.0
        else:
            reward = 0.0
            if len(self._cache) >= CACHE_CAPACITY:
                evict_slot = int(action) % CACHE_CAPACITY
                self._cache[evict_slot] = access
            else:
                self._cache.append(access)

        self._history.append(access)
        if len(self._history) > HISTORY_WINDOW:
            self._history = self._history[-HISTORY_WINDOW:]

        self._step += 1
        terminated = False
        truncated = self._step >= TRACE_LENGTH
        return self._make_obs(), reward, terminated, truncated, {"hit": is_hit}

    def close(self) -> None:
        pass

    def _make_obs(self) -> np.ndarray:
        assert self._trace is not None
        cache_arr = list(self._cache) + [-1] * (CACHE_CAPACITY - len(self._cache))
        hist_arr = [-1] * (HISTORY_WINDOW - len(self._history)) + self._history[-HISTORY_WINDOW:]
        current = int(self._trace[self._step]) if self._step < TRACE_LENGTH else -1
        return np.array(cache_arr + hist_arr + [current], dtype=np.int32)

    @property
    def unwrapped(self) -> CacheReplacementEnv:
        return self


def _factory() -> object:
    return CacheReplacementEnv()


register_env(
    env_id="cache_replacement",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [CACHE_CAPACITY + HISTORY_WINDOW + 1],
        "low": [-1] * (CACHE_CAPACITY + HISTORY_WINDOW + 1),
        "high": [N_OBJECTS - 1] * (CACHE_CAPACITY + HISTORY_WINDOW + 1),
        "dtype": "int32",
    },
    action_space={
        "type": "Discrete",
        "n": CACHE_CAPACITY,
    },
    max_episode_steps=TRACE_LENGTH,
    # Random eviction on missing ≈ ~5% hit rate on Zipfian (very few hits).
    # LRU/ARC ≈ ~70-85% hit rate on Zipfian, ~10-20% on scan-heavy held-out.
    # Mean returns: random ≈ 25 (5% of 500), expert ≈ 425 (LRU on Zipfian).
    # Approximate; calibrate.
    expert_baseline=425.0,
    random_baseline=25.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
