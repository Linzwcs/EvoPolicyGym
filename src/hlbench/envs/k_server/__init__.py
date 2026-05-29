"""K-Server env registration (v1 roster #12).

K servers on a 2D Euclidean plane respond to a stream of requests
arriving at points. The policy must dispatch one server to each
request; chosen server moves from its current position to the
request point. Reward is the negative total distance moved.

Train pool: requests drawn from a 2-cluster Gaussian mixture.
Held-out: requests adversarially clustered far from any "natural"
server position — punishes greedy nearest-server policies that
overcommit one server to a hot region.

Seed-magnitude convention: train in [0, 1_000_000); held-out in
[1_000_000, 2_000_000).

Theoretical background:
- The Work Function Algorithm (WFA) is (2k-1)-competitive (Koutsoupias-
  Papadimitriou 1995), but its compute is O(k! n) — intractable for
  the per-step 10ms budget at any meaningful k.
- The Double Coverage algorithm is 2-competitive on trees but doesn't
  apply to a 2D Euclidean plane in general.
- Greedy (nearest available server) is unbounded competitive but
  often performs well empirically on benign distributions.

Side effect: importing this module registers ``k_server``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hlbench.envs.registry import register_env

_HERE = Path(__file__).parent

K_SERVERS: int = 3
N_REQUESTS: int = 200
PLANE_BOUND: float = 1.0  # Requests + servers live in [-PLANE_BOUND, PLANE_BOUND]^2

_HELDOUT_SEED_FLOOR: int = 1_000_000


def _generate_requests(seed: int, n_requests: int) -> np.ndarray:
    """Deterministic request sequence for a seed.

    Train: 2-cluster Gaussian mixture, balanced. Greedy works reasonably.
    Held-out: 4 corners with adversarial weighting (75% of requests at
    one corner that's far from any 'natural' server start). Greedy
    overcommits one server to the hot corner; smarter policies anticipate.
    """
    rng = np.random.default_rng(seed)
    if seed >= _HELDOUT_SEED_FLOOR:
        corners = np.array(
            [[0.7, 0.7], [-0.7, 0.7], [-0.7, -0.7], [0.7, -0.7]],
            dtype=np.float32,
        )
        # 75% at corner 0, 25% spread across the others
        choices = rng.choice(
            len(corners), size=n_requests, p=[0.75, 0.0833, 0.0833, 0.0834]
        )
        centers = corners[choices]
    else:
        centers = np.array(
            [[0.4, 0.4], [-0.4, -0.4]], dtype=np.float32
        )
        # 50/50 between two centers
        choices = rng.integers(0, 2, size=n_requests)
        centers = centers[choices]
    noise = rng.normal(0, 0.05, size=(n_requests, 2)).astype(np.float32)
    requests = centers + noise
    return np.clip(requests, -PLANE_BOUND, PLANE_BOUND).astype(np.float32)  # type: ignore[no-any-return]


def _initial_server_positions() -> np.ndarray:
    """Servers start at fixed canonical positions (the K vertices of a
    regular polygon inscribed in the unit circle). Deterministic across
    seeds so the env's only seed-driven variability is the requests."""
    angles = np.linspace(0, 2 * np.pi, K_SERVERS, endpoint=False)
    return np.stack(
        [0.3 * np.cos(angles), 0.3 * np.sin(angles)], axis=1
    ).astype(np.float32)


class KServerEnv:
    """Custom env (duck-typed).

    Observation: float32 vector of length 2*K + 2:
      [server_0.x, server_0.y, ..., server_{K-1}.x, server_{K-1}.y,
       request.x, request.y]

    Action: Discrete(K) — index of server to dispatch.

    Reward: negative Euclidean distance moved by the chosen server.
    Terminates after N_REQUESTS steps.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    spec = None

    def __init__(self) -> None:
        import gymnasium as gym
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-PLANE_BOUND, high=PLANE_BOUND,
            shape=(2 * K_SERVERS + 2,), dtype=np.float32,
        )
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(K_SERVERS)  # type: ignore[type-arg]
        self._requests: np.ndarray | None = None
        self._servers: np.ndarray | None = None
        self._step: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = 0
        self._requests = _generate_requests(seed, N_REQUESTS)
        self._servers = _initial_server_positions()
        self._step = 0
        return self._make_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._requests is not None and self._servers is not None
        server_idx = int(action) % K_SERVERS
        target = self._requests[self._step]
        old_pos = self._servers[server_idx]
        distance = float(np.linalg.norm(target - old_pos))
        self._servers[server_idx] = target
        reward = -distance
        self._step += 1
        terminated = False
        truncated = self._step >= N_REQUESTS
        return self._make_obs(), reward, terminated, truncated, {"distance": distance}

    def close(self) -> None:
        pass

    def _make_obs(self) -> np.ndarray:
        assert self._requests is not None and self._servers is not None
        server_flat = self._servers.flatten()
        if self._step < N_REQUESTS:
            req = self._requests[self._step]
        else:
            req = np.zeros(2, dtype=np.float32)
        return np.concatenate([server_flat, req]).astype(np.float32)

    @property
    def unwrapped(self) -> KServerEnv:
        return self


def _factory() -> object:
    return KServerEnv()


register_env(
    env_id="k_server",
    env_version="0.1",
    factory=_factory,
    obs_space={
        "type": "Box",
        "shape": [2 * K_SERVERS + 2],
        "low": [-PLANE_BOUND] * (2 * K_SERVERS + 2),
        "high": [PLANE_BOUND] * (2 * K_SERVERS + 2),
        "dtype": "float32",
    },
    action_space={
        "type": "Discrete",
        "n": K_SERVERS,
    },
    max_episode_steps=N_REQUESTS,
    # Baselines: random dispatch ≈ -100 (lots of movement). Greedy
    # nearest ≈ -30 on train, ≈ -80 on held-out. WFA-equivalent ≈ -25.
    # Approximate; calibrate.
    expert_baseline=-25.0,
    random_baseline=-100.0,
    train_seeds_path=_HERE / "data" / "train.json",
    heldout_seeds_path=_HERE / "data" / "heldout.json",
    task_md_path=_HERE / "TASK.md",
    starter_policy_path=_HERE / "starter_policy.py",
    n_env_instances=10000,
    obs_storage="inline",
    reward_components=None,
)
