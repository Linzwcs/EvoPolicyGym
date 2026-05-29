"""EnvRunner tests — verifies trajectory schema and PD-on-Pendulum sanity.

These require gymnasium installed. Pendulum-v1 is part of gymnasium's
classic_control suite, no extra deps.
"""

from __future__ import annotations

import pytest

# Skip the whole module if gymnasium isn't installed.
gym = pytest.importorskip("gymnasium")

from hlbench.core.env_runner import run_episode  # noqa: E402
from hlbench.envs.registry import get_env  # noqa: E402
from tests.conftest import CrashingPolicy, PendulumPDPolicy  # noqa: E402


@pytest.fixture(scope="module")
def pendulum_env_def() -> object:
    """Resolved EnvDefinition for pendulum (loads its real seeds)."""
    # Import triggers registration.
    import hlbench.envs  # noqa: F401

    return get_env("pendulum")


@pytest.fixture
def pendulum_env(pendulum_env_def: object) -> object:
    """A fresh gymnasium env instance per test."""
    env = pendulum_env_def.factory()
    yield env
    env.close()


# ---------- shape / schema tests ----------


def test_run_episode_returns_valid_record(pendulum_env: object) -> None:
    """Smoke test: PD on Pendulum produces a well-formed EpisodeRecord."""
    policy = PendulumPDPolicy()
    rec = run_episode(
        policy,
        pendulum_env,
        real_seed=42,
        episode_index=0,
        action_space_type="Box",
        max_steps=200,
    )

    assert rec.length == 200, "Pendulum has no natural termination → should hit max_steps"
    assert len(rec.trajectory) == rec.length
    assert isinstance(rec.return_, float)
    # Pendulum's last step typically has truncated=True (TimeLimit wrapper).
    # The actual terminated/truncated values depend on the gym version,
    # so we only assert that exactly one of them set the last step's flag.
    last = rec.trajectory[-1]
    assert last["terminated"] or last["truncated"], \
        "max_steps reached → last step should be terminated or truncated"


def test_trajectory_schema_matches_spec(pendulum_env: object) -> None:
    """Each trajectory entry has the SPEC §4.2 fields with correct types."""
    policy = PendulumPDPolicy()
    rec = run_episode(
        policy,
        pendulum_env,
        real_seed=0,
        episode_index=0,
        action_space_type="Box",
        max_steps=10,
    )

    assert len(rec.trajectory) == 10
    for i, step in enumerate(rec.trajectory):
        # Required keys
        for k in ("t", "obs", "action", "reward", "terminated", "truncated", "info"):
            assert k in step, f"step {i} missing key {k}"
        # Types
        assert step["t"] == i, "step.t should be 0-based step index"
        assert isinstance(step["obs"], list), "Pendulum obs is Box → JSON list"
        assert len(step["obs"]) == 3, "Pendulum obs has shape [3]"
        assert isinstance(step["action"], list), "Pendulum action is Box → JSON list"
        assert len(step["action"]) == 1, "Pendulum action has shape [1]"
        assert isinstance(step["reward"], float)
        assert isinstance(step["terminated"], bool)
        assert isinstance(step["truncated"], bool)
        assert isinstance(step["info"], dict)


def test_obs_none_when_record_obs_false(pendulum_env: object) -> None:
    """external obs storage → every trajectory entry's obs is None."""
    policy = PendulumPDPolicy()
    rec = run_episode(
        policy,
        pendulum_env,
        real_seed=0,
        episode_index=0,
        action_space_type="Box",
        max_steps=5,
        record_obs=False,
    )
    for step in rec.trajectory:
        assert step["obs"] is None


def test_episode_index_is_passed_to_policy_reset(pendulum_env: object) -> None:
    """policy.reset(episode_index) is called with the value we pass in."""
    policy = PendulumPDPolicy()
    run_episode(
        policy,
        pendulum_env,
        real_seed=0,
        episode_index=7,
        action_space_type="Box",
        max_steps=5,
    )
    assert policy._last_episode_index == 7
    assert policy._episode_count == 1


# ---------- behavior tests ----------


def test_pd_policy_beats_random_baseline_on_pendulum(
    pendulum_env_def: object,
) -> None:
    """PD controller should easily beat the random baseline (~-1200).

    Per CLAUDE.md invariant 2, baselines are server-internal so we read
    `pendulum_env_def.random_baseline` from the registry (which is allowed
    from inside hlbench code; agents would not have this access).
    """
    policy = PendulumPDPolicy()
    returns: list[float] = []
    for seed in [42, 100, 200, 300, 400]:
        env = pendulum_env_def.factory()
        rec = run_episode(
            policy,
            env,
            real_seed=seed,
            episode_index=0,
            action_space_type="Box",
            max_steps=200,
        )
        returns.append(rec.return_)
        env.close()

    mean = sum(returns) / len(returns)
    # Random baseline is ~-1200. PD should be much better (typically ~-200 to -400).
    assert mean > -800, f"PD mean return {mean:.1f} is suspiciously bad (random ~ {pendulum_env_def.random_baseline})"
    # And we should be at least within striking distance of expert (~-150).
    assert mean < 0, f"Pendulum returns are always negative; got mean={mean}"


def test_deterministic_seed_gives_same_return(pendulum_env_def: object) -> None:
    """Same (policy, env, seed) → same return. Reproducibility check."""
    policy = PendulumPDPolicy()
    returns: list[float] = []
    for _ in range(3):
        env = pendulum_env_def.factory()
        rec = run_episode(
            policy,
            env,
            real_seed=12345,
            episode_index=0,
            action_space_type="Box",
            max_steps=200,
        )
        returns.append(rec.return_)
        env.close()
    # All three should be identical to floating-point precision.
    assert returns[0] == returns[1] == returns[2], f"non-deterministic returns: {returns}"


# ---------- error handling ----------


def test_act_exception_is_captured_not_raised(pendulum_env: object) -> None:
    """policy.act() raising → ended_with_error=True, no exception bubbles."""
    policy = CrashingPolicy()
    rec = run_episode(
        policy,
        pendulum_env,
        real_seed=0,
        episode_index=0,
        action_space_type="Box",
        max_steps=200,
    )
    assert rec.ended_with_error is True
    assert rec.length == 0, "first act() crashed → no trajectory entries"
    assert rec.return_ == 0.0
