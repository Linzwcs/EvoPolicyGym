"""Online-Bipartite-Matching env registration (v1 roster #13).

N left vertices are fixed at the start of every episode. Right
vertices arrive online; each carries a set of edges to left
vertices. The policy must decide immediately whether to match the
arrival to one of its (still-unmatched) left neighbors, or skip
(unmatched).

Train pool: random bipartite graphs (each arrival has each left
vertex as a neighbor with probability `p_train`). RANKING algorithm
achieves the (1 - 1/e) ≈ 0.632 competitive ratio.

Held-out pool: structured graphs where greedy is suboptimal — the
adversarial example known as "perfect graph with online labels": the
first `n/2` arrivals only connect to left vertices `{0, ..., n/2-1}`,
the next `n/2` only connect to `{n/2, ..., n-1}` AND to a "honey
trap" subset `{0, ..., n/2-1}`. Greedy that grabs the honey trap
strands the second-half left vertices unmatched.

Seed-magnitude convention: train in [0, 1_000_000); held-out in
[1_000_000, 2_000_000).

Side effect: importing this module registers
``online_bipartite_matching``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

N_LEFT: int = 16
M_RIGHT: int = 24        # M >= N; some arrivals will go unmatched
EDGE_PROB_TRAIN: float = 0.25

_HELDOUT_SEED_FLOOR: int = 1_000_000


def _generate_arrivals(seed: int) -> np.ndarray:
    """Deterministic arrival sequence — an (M, N) bool matrix where
    row i is the neighbor mask of the i-th arriving right vertex.

    Train: Erdős-Rényi G(N, M, p) — each (left, right) edge
    independently present with probability EDGE_PROB_TRAIN.
    Held-out: structured adversarial — first M/2 arrivals only
    connect to left half; remaining M/2 connect to both halves with
    bias toward the left half (the "honey trap")."""
    rng = np.random.default_rng(seed)
    if seed >= _HELDOUT_SEED_FLOOR:
        arrivals = np.zeros((M_RIGHT, N_LEFT), dtype=bool)
        left_half = list(range(N_LEFT // 2))
        right_half = list(range(N_LEFT // 2, N_LEFT))
        # Phase 1: first half only sees left_half (1-2 random connections each)
        for i in range(M_RIGHT // 2):
            n_conn = rng.integers(1, 3)
            for v in rng.choice(left_half, size=n_conn, replace=False):
                arrivals[i, v] = True
        # Phase 2: second half sees right_half PLUS some left_half "trap"
        for i in range(M_RIGHT // 2, M_RIGHT):
            n_trap = rng.integers(1, 3)
            for v in rng.choice(left_half, size=n_trap, replace=False):
                arrivals[i, v] = True
            n_right = rng.integers(1, 3)
            for v in rng.choice(right_half, size=n_right, replace=False):
                arrivals[i, v] = True
        return arrivals
    else:
        return rng.random(size=(M_RIGHT, N_LEFT)) < EDGE_PROB_TRAIN


class OnlineBipartiteMatchingEnv:
    """Custom env (duck-typed).

    Observation: int8 vector of length 2*N_LEFT:
      [left_matched_mask (N), current_arrival_neighbors_mask (N)]

    Action: Discrete(N + 1) — left vertex index in [0, N) to match
    to, OR N for "skip this arrival" (unmatched).

    Reward: +1.0 per successful match (valid neighbor AND not already
    matched), 0.0 for skip or invalid attempt (we don't penalize
    invalid attempts to keep the design simple; the action just has
    no effect).
    Terminates after M_RIGHT arrivals.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    spec = None

    def __init__(self) -> None:
        import gymnasium as gym
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0, high=1, shape=(2 * N_LEFT,), dtype=np.int8,
        )
        # N actions = pick a left vertex; +1 = skip
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(N_LEFT + 1)  # type: ignore[type-arg]
        self._arrivals: np.ndarray | None = None
        self._matched: np.ndarray | None = None
        self._step: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = 0
        self._arrivals = _generate_arrivals(seed)
        self._matched = np.zeros(N_LEFT, dtype=bool)
        self._step = 0
        return self._make_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._arrivals is not None and self._matched is not None
        reward = 0.0
        info: dict[str, Any] = {"matched": False}
        if 0 <= int(action) < N_LEFT:
            v = int(action)
            neighbors = self._arrivals[self._step]
            if neighbors[v] and not self._matched[v]:
                self._matched[v] = True
                reward = 1.0
                info["matched"] = True
            # else: invalid attempt — no effect, no penalty
        # action == N_LEFT or out-of-range: skip
        self._step += 1
        terminated = False
        truncated = self._step >= M_RIGHT
        return self._make_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        pass

    def _make_obs(self) -> np.ndarray:
        assert self._arrivals is not None and self._matched is not None
        matched_mask = self._matched.astype(np.int8)
        if self._step < M_RIGHT:
            neighbors_mask = self._arrivals[self._step].astype(np.int8)
        else:
            neighbors_mask = np.zeros(N_LEFT, dtype=np.int8)
        return np.concatenate([matched_mask, neighbors_mask]).astype(np.int8)

    @property
    def unwrapped(self) -> OnlineBipartiteMatchingEnv:
        return self


def _factory() -> object:
    return OnlineBipartiteMatchingEnv()


register_env(
    env_id="online_bipartite_matching",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [2 * N_LEFT],
        "low": 0,
        "high": 1,
        "dtype": "int8",
    },
    action_space={
        "type": "Discrete",
        "n": N_LEFT + 1,
    },
    max_episode_steps=M_RIGHT,
    # Baselines: random ≈ 4-6 matches on train (random attempts often
    # invalid). Greedy first-available ≈ 11-13 on train, ≈ 6-8 on
    # held-out. RANKING ≈ 12-14 on train, 9-11 on held-out. Max
    # possible = N_LEFT = 16 (perfect matching).
    expert_baseline=13.0,
    random_baseline=5.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
