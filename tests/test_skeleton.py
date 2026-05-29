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
