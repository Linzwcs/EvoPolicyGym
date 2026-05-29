"""Tests for the observations.npy infrastructure (SPEC §4.6) and the
two visual envs that exercise it end-to-end:

- ``car_racing_pixel`` (full 96x96x3, obs_storage=external)
- ``pendulum_from_pixels`` (rendered Pendulum at 64x64x3, external)

Coverage:
- Registration + obs_space metadata
- ``EpisodeRecord.observations`` is populated when ``record_obs=False``
- ``write_observations`` writes a valid .npy file with correct shape
- End-to-end: SubmitHandler-driven pendulum_from_pixels submit produces
  ``observations.npy`` files of shape ``(episode_length, 64, 64, 3)``
  alongside ``trajectory.jsonl`` whose obs fields are all null.

The car_racing_pixel end-to-end path is NOT exercised in pytest's main
process (Box2D segfault risk on macOS); registration + obs_space + the
shared infra is verified, and the env is validated via ad-hoc
standalone scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hlbench.envs.registry import get_env


@pytest.fixture(autouse=True)
def _ensure_envs_registered() -> None:
    import hlbench.envs  # noqa: F401


VISUAL_PIXEL_ENVS = ["car_racing_pixel", "pendulum_from_pixels"]


@pytest.mark.parametrize("env_id", VISUAL_PIXEL_ENVS)
def test_visual_env_registered_external_storage(env_id: str) -> None:
    d = get_env(env_id)
    assert d.env_id == env_id
    assert d.obs_storage == "external", (
        f"{env_id} must use external obs_storage; got {d.obs_storage}"
    )
    assert d.train_seeds_path.exists()
    assert d.heldout_seeds_path.exists()


def test_pendulum_from_pixels_obs_shape() -> None:
    d = get_env("pendulum_from_pixels")
    assert d.obs_space["shape"] == [64, 64, 3]
    assert d.obs_space["dtype"] == "uint8"
    assert d.action_space["shape"] == [1]


def test_car_racing_pixel_obs_shape() -> None:
    d = get_env("car_racing_pixel")
    assert d.obs_space["shape"] == [96, 96, 3]
    assert d.obs_space["dtype"] == "uint8"
    assert d.action_space["shape"] == [3]


def test_episode_record_observations_field_default() -> None:
    """EpisodeRecord.observations defaults to None (inline mode)."""
    from hlbench.core.env_runner import EpisodeRecord

    rec = EpisodeRecord(
        trajectory=[], return_=0.0, length=0,
        terminated=False, truncated=False,
    )
    assert rec.observations is None


def test_run_episode_accumulates_observations_when_external() -> None:
    """run_episode with record_obs=False populates EpisodeRecord.observations."""
    from hlbench.core.env_runner import run_episode

    # Use pendulum_from_pixels factory directly (Pendulum render is
    # safe in pytest main process; no Box2D involved).
    d = get_env("pendulum_from_pixels")
    env = d.factory()

    class _ZeroPolicy:
        def reset(self, episode_index: int) -> None:
            del episode_index

        def act(self, obs: Any) -> Any:
            return np.array([0.0], dtype=np.float32)

    rec = run_episode(
        _ZeroPolicy(), env,
        real_seed=42, episode_index=0,
        action_space_type="Box",
        max_steps=5,
        record_obs=False,
    )
    env.close()
    assert rec.observations is not None
    assert len(rec.observations) == rec.length
    # Each obs is the rendered 64x64x3 frame
    assert all(o.shape == (64, 64, 3) for o in rec.observations)
    # Trajectory entries all have obs=None
    assert all(e["obs"] is None for e in rec.trajectory)


def test_run_episode_inline_mode_no_observations() -> None:
    """run_episode with record_obs=True leaves observations as None."""
    from hlbench.core.env_runner import run_episode

    d = get_env("pendulum")  # state-based env; record_obs=True is fine inline
    env = d.factory()

    class _ZeroPolicy:
        def reset(self, episode_index: int) -> None:
            del episode_index

        def act(self, obs: Any) -> Any:
            return np.array([0.0], dtype=np.float32)

    rec = run_episode(
        _ZeroPolicy(), env,
        real_seed=42, episode_index=0,
        action_space_type="Box",
        max_steps=3,
        record_obs=True,
    )
    env.close()
    assert rec.observations is None
    # Trajectory has obs values
    assert all(e["obs"] is not None for e in rec.trajectory)


def test_write_observations_round_trip(tmp_path: Path) -> None:
    """write_observations + np.load round-trip preserves shape + dtype."""
    from hlbench.core.feedback import write_observations

    obs_list = [
        np.full((4, 4, 3), i, dtype=np.uint8) for i in range(7)
    ]
    out = tmp_path / "observations.npy"
    write_observations(out, obs_list)
    assert out.exists()

    loaded = np.load(out)
    assert loaded.shape == (7, 4, 4, 3)
    assert loaded.dtype == np.uint8
    # Frame 3 should be all-3s
    assert np.all(loaded[3] == 3)


def test_write_observations_empty_list(tmp_path: Path) -> None:
    """Empty obs list (e.g. reset_error before any step) writes a 0-row array."""
    from hlbench.core.feedback import write_observations

    out = tmp_path / "observations.npy"
    write_observations(out, [])
    assert out.exists()
    loaded = np.load(out)
    assert loaded.shape == (0,)


def test_pendulum_from_pixels_e2e_writes_observations_npy(tmp_path: Path) -> None:
    """End-to-end: a submit on pendulum_from_pixels produces an
    observations.npy alongside each episode's trajectory.jsonl, with
    matching length and shape (episode_length, 64, 64, 3) uint8."""
    from hlbench.core.server import Server

    d = get_env("pendulum_from_pixels")
    runs_root = tmp_path / "runs"
    server = Server(
        env_id=d.env_id, runs_root=runs_root,
        model="test-zero", exp_id="obs-npy-e2e",
    )

    # Stage starter policy (Server.__init__ already auto-staged it).
    # Run a tiny submit: 2 episodes with very short max_steps via env_meta.
    # max_episode_steps in registration is 200; we let it run a few steps.
    result = server.submit(env_instances=[0])
    assert result.status == "ok", result

    # Check that observations.npy exists for the episode and has the
    # right shape relative to trajectory.jsonl
    workspace = server.workspace_dir
    submit_dir = workspace / "feedback" / "submit_000"
    ep_dirs = sorted((submit_dir / "episodes").iterdir())
    assert len(ep_dirs) == 1
    ep = ep_dirs[0]

    obs_path = ep / "observations.npy"
    assert obs_path.exists(), f"observations.npy missing at {obs_path}"

    obs_arr = np.load(obs_path)
    # Shape: (episode_length, 64, 64, 3)
    traj_path = ep / "trajectory.jsonl"
    n_steps = sum(1 for _ in traj_path.read_text().strip().splitlines() if _)
    assert obs_arr.shape == (n_steps, 64, 64, 3), (
        f"obs shape {obs_arr.shape} != ({n_steps}, 64, 64, 3)"
    )
    assert obs_arr.dtype == np.uint8

    # Trajectory entries must have obs=null (external mode)
    for line in traj_path.read_text().strip().splitlines():
        entry = json.loads(line)
        assert entry["obs"] is None, "external mode must null obs in trajectory"

    server.finalize()


def test_pendulum_from_pixels_factory_uses_render() -> None:
    """Source inspection: factory must use render_mode='rgb_array'."""
    src = Path(__file__).parent.parent / "src/hlbench/envs/pendulum_from_pixels/__init__.py"
    text = src.read_text()
    assert 'render_mode="rgb_array"' in text
    assert "_PendulumPixelWrapper" in text


def test_car_racing_pixel_uses_external_storage() -> None:
    """Source inspection: car_racing_pixel must declare external storage."""
    src = Path(__file__).parent.parent / "src/hlbench/envs/car_racing_pixel/__init__.py"
    text = src.read_text()
    assert 'obs_storage="external"' in text
