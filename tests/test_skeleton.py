"""Day 1 smoke tests — package layout, registry import, seed file integrity.

These verify the skeleton actually loads. Real tests for each module
come in their respective days (see docs/architecture.md §5).
"""

from __future__ import annotations


def test_package_imports() -> None:
    """The top-level package imports cleanly."""
    import hlbench

    assert hlbench.__version__ == "0.1.0a1"


def test_envs_registers_pendulum() -> None:
    """Importing hlbench.envs registers the pendulum env."""
    from hlbench.envs.registry import get_env, list_envs

    assert "pendulum" in list_envs()
    env = get_env("pendulum")
    assert env.env_id == "pendulum"
    assert env.env_version == "0.1"
    assert env.n_env_instances == 256
    assert env.obs_storage == "inline"
    # Server-internal baselines exist but never appear in public_env_meta.
    assert env.expert_baseline == -150.0
    assert env.random_baseline == -1200.0


def test_envs_registers_classic_control_set() -> None:
    """Acrobot + MountainCarContinuous register alongside Pendulum.

    These exercise the Discrete action serialization (acrobot) and a
    different Box action range (mountain_car_continuous) that
    pendulum doesn't cover."""
    import hlbench.envs  # noqa: F401  side effect: trigger registrations
    from hlbench.envs.registry import get_env, list_envs

    ids = set(list_envs())
    assert {"pendulum", "acrobot", "mountain_car_continuous"} <= ids

    acro = get_env("acrobot")
    assert acro.action_space["type"] == "Discrete"
    assert acro.action_space["n"] == 3
    assert acro.obs_space["shape"] == [6]
    assert acro.max_episode_steps == 500

    mcc = get_env("mountain_car_continuous")
    assert mcc.action_space["type"] == "Box"
    assert mcc.action_space["shape"] == [1]
    assert mcc.action_space["low"] == [-1.0] and mcc.action_space["high"] == [1.0]
    assert mcc.obs_space["shape"] == [2]
    assert mcc.max_episode_steps == 999


def test_classic_control_envs_have_starter_and_task_md() -> None:
    """Every env ships TASK.md + starter_policy.py + seed pools per the
    convention in [[feedback_lib_consumer_separation]] / env packaging.
    Regression guard so a new env can't ship missing these."""
    from hlbench.envs.registry import get_env

    for env_id in ("acrobot", "mountain_car_continuous"):
        env = get_env(env_id)
        assert env.task_md_path is not None and env.task_md_path.is_file()
        assert env.starter_policy_path is not None and env.starter_policy_path.is_file()
        assert env.train_seeds_path.is_file()
        assert env.heldout_seeds_path.is_file()


def test_envs_registers_box2d_set() -> None:
    """BipedalWalker + LunarLanderContinuous register. Cover the
    higher-dim continuous action space (action_dim 4 and 2)."""
    import hlbench.envs  # noqa: F401
    from hlbench.envs.registry import get_env, list_envs

    assert {"bipedal_walker", "lunar_lander_continuous"} <= set(list_envs())

    bw = get_env("bipedal_walker")
    assert bw.action_space["type"] == "Box"
    assert bw.action_space["shape"] == [4]
    assert bw.obs_space["shape"] == [24]
    assert bw.max_episode_steps == 1600

    ll = get_env("lunar_lander_continuous")
    assert ll.action_space["type"] == "Box"
    assert ll.action_space["shape"] == [2]
    assert ll.obs_space["shape"] == [8]
    assert ll.max_episode_steps == 1000


def test_box2d_envs_have_starter_and_task_md() -> None:
    from hlbench.envs.registry import get_env

    for env_id in ("bipedal_walker", "lunar_lander_continuous"):
        env = get_env(env_id)
        assert env.task_md_path is not None and env.task_md_path.is_file()
        assert env.starter_policy_path is not None and env.starter_policy_path.is_file()
        assert env.train_seeds_path.is_file()
        assert env.heldout_seeds_path.is_file()


def test_public_env_meta_hides_baselines() -> None:
    """expert_baseline / random_baseline MUST NOT leak via public_env_meta.

    Per CLAUDE.md invariant 2: baselines are server-internal.
    """
    from hlbench.envs.registry import get_env

    meta = get_env("pendulum").public_env_meta()
    assert "expert_baseline" not in meta
    assert "random_baseline" not in meta
    # But the fields the agent legitimately needs are present.
    for required in ("obs_space", "action_space", "max_episode_steps",
                     "n_env_instances", "obs_storage"):
        assert required in meta, f"missing {required}"


def test_seed_resolver_loads_pendulum() -> None:
    """train.json and heldout.json load and have the expected sizes."""
    from hlbench.core.seed_resolver import SeedResolver
    from hlbench.envs.registry import get_env

    env = get_env("pendulum")
    sm = SeedResolver(env.train_seeds_path, env.heldout_seeds_path)

    assert sm.n_env_instances == 256
    assert sm.n_held_out == 256

    # Range check works.
    assert isinstance(sm.real_seed_for_instance(0), int)
    assert isinstance(sm.real_seed_for_instance(255), int)


def test_seed_resolver_rejects_out_of_range() -> None:
    """env_instance 256 (out of [0, 256)) → ValueError → invalid_env_instance verdict."""
    import pytest

    from hlbench.core.seed_resolver import SeedResolver
    from hlbench.envs.registry import get_env

    env = get_env("pendulum")
    sm = SeedResolver(env.train_seeds_path, env.heldout_seeds_path)

    with pytest.raises(ValueError):
        sm.real_seed_for_instance(256)
    with pytest.raises(ValueError):
        sm.real_seed_for_instance(-1)


def test_train_and_heldout_seeds_disjoint() -> None:
    """Train and held-out pools never overlap.

    Spec invariant: held-out is a separate pool, not a slice of train.
    """
    from hlbench.core.seed_resolver import SeedResolver
    from hlbench.envs.registry import get_env

    env = get_env("pendulum")
    sm = SeedResolver(env.train_seeds_path, env.heldout_seeds_path)

    train_set = {sm.real_seed_for_instance(i) for i in range(sm.n_env_instances)}
    heldout_set = set(sm.held_out_seeds())
    assert train_set.isdisjoint(heldout_set), \
        "train and held-out pools must not overlap"
