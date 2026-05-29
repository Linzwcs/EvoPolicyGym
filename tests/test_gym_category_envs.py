"""Smoke tests for the 9 new gym-category envs.

Coverage:
- Registration metadata (all 9)
- Factory + reset (skipped for envs that segfault in pytest main process;
  validated end-to-end via Sandbox in test_submit_handler.py)
- Seed pool sanity (10000 train + 256 heldout, disjoint)

MuJoCo, MiniGrid, and CarRacing (Box2D) all load native extensions
that may not be safe in pytest's main process on macOS — for those
we test only the registration + sampler logic, not factory + reset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlbench.envs.registry import get_env


@pytest.fixture(autouse=True)
def _ensure_envs_registered() -> None:
    import hlbench.envs  # noqa: F401


GYM_CATEGORY_ENVS = [
    # MuJoCo
    "half_cheetah", "hopper", "walker2d", "ant",
    # MiniGrid
    "minigrid_doorkey", "minigrid_keycorridor",
    "minigrid_lavacrossing", "minigrid_obstructedmaze",
    # Box2D pixel (downsampled)
    "car_racing",
]


@pytest.mark.parametrize("env_id", GYM_CATEGORY_ENVS)
def test_gym_env_registered(env_id: str) -> None:
    """Each env must register cleanly with correct metadata."""
    d = get_env(env_id)
    assert d.env_id == env_id
    assert d.env_version == "0.1"
    assert d.n_env_instances == 10000
    assert d.train_seeds_path.exists()
    assert d.heldout_seeds_path.exists()
    assert d.starter_policy_path is not None and d.starter_policy_path.exists()
    assert d.task_md_path is not None and d.task_md_path.exists()


@pytest.mark.parametrize("env_id", GYM_CATEGORY_ENVS)
def test_gym_env_seed_pools(env_id: str) -> None:
    """train.json has 10000 entries; heldout.json has 256; disjoint."""
    d = get_env(env_id)
    train = json.loads(d.train_seeds_path.read_text())["real_seeds"]
    heldout = json.loads(d.heldout_seeds_path.read_text())["real_seeds"]
    assert len(train) == 10000
    assert len(heldout) == 256
    assert set(train).isdisjoint(set(heldout))


def test_minigrid_obs_dim() -> None:
    """All 4 MiniGrid envs expose flat 148-D obs (7*7*3 image + 1 direction)."""
    for env_id in [
        "minigrid_doorkey", "minigrid_keycorridor",
        "minigrid_lavacrossing", "minigrid_obstructedmaze",
    ]:
        d = get_env(env_id)
        assert d.obs_space["shape"] == [148], f"{env_id} obs shape mismatch"
        assert d.obs_space["dtype"] == "uint8"
        assert d.action_space["type"] == "Discrete"
        assert d.action_space["n"] == 7


def test_car_racing_downsampled_obs() -> None:
    """car_racing obs is 16x16x3 uint8 (downsampled from CarRacing-v3's 96x96x3)."""
    d = get_env("car_racing")
    assert d.obs_space["shape"] == [16, 16, 3]
    assert d.obs_space["dtype"] == "uint8"
    assert d.action_space["type"] == "Box"
    assert d.action_space["shape"] == [3]


def test_mujoco_action_shapes() -> None:
    """MuJoCo action shapes match Gymnasium documentation."""
    for env_id, expected_act_shape in [
        ("half_cheetah", [6]),
        ("hopper", [3]),
        ("walker2d", [6]),
        ("ant", [8]),
    ]:
        d = get_env(env_id)
        assert d.action_space["shape"] == expected_act_shape, f"{env_id} act shape"
        assert d.action_space["type"] == "Box"


def test_mujoco_obs_shapes_match_gym() -> None:
    """MuJoCo obs shapes match what Gymnasium returns."""
    # Verified via direct gymnasium.make in the env's factory; if these
    # mismatch, the registered env_meta lies and downstream consumers break.
    for env_id, expected_obs_shape in [
        ("half_cheetah", [17]),
        ("hopper", [11]),
        ("walker2d", [17]),
        ("ant", [105]),
    ]:
        d = get_env(env_id)
        assert d.obs_space["shape"] == expected_obs_shape


def test_total_env_count_after_landing() -> None:
    """After this batch lands, registry has 20 envs: 5 v0 + 6 v1-batch1
    (hardcore + online algo) + 9 v1-batch2 (this PR: mujoco + minigrid +
    car_racing)."""
    from hlbench.envs.registry import _REGISTRY
    assert len(_REGISTRY) == 20, f"expected 20 envs, got {len(_REGISTRY)}"


def test_minigrid_factory_returns_wrapped_env() -> None:
    """MiniGrid factory must produce an env with flat 148-D Box obs space
    (not the raw Dict obs). Verified via source inspection rather than
    factory call to avoid in-process MiniGrid/numpy interaction risk
    on some pytest configurations."""
    src = Path(__file__).parent.parent / "src/hlbench/envs/minigrid_doorkey/__init__.py"
    text = src.read_text()
    assert "OBS_DIM" in text
    assert "_MiniGridFlattenWrapper" in text
    assert "import minigrid" in text


def test_car_racing_factory_targets_downsample_wrapper() -> None:
    """car_racing factory must use the downsample wrapper, not raw 96x96."""
    src = Path(__file__).parent.parent / "src/hlbench/envs/car_racing/__init__.py"
    text = src.read_text()
    assert "DOWNSAMPLE_H" in text
    assert "DOWNSAMPLE_W" in text
    assert "_DownsampleWrapper" in text
