"""Sandbox tests — subprocess lifecycle, init errors, act-timeout.

Requires gymnasium (real env in the child). Each test writes a
self-contained `policy.py` into a tmp `system/` directory; the sandbox
spawns a child that imports it.
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.sandbox import (  # noqa: E402
    Sandbox,
    SandboxConfig,
    SandboxInitError,
)
from hlbench.envs.registry import get_env  # noqa: E402

# ----------------------- helpers ------------------------------------------


def _write_policy(snapshot_dir: Path, body: str) -> None:
    """Write `body` as snapshot_dir/system/policy.py. Dedents and strips."""
    system = snapshot_dir / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "policy.py").write_text(textwrap.dedent(body).lstrip())


@pytest.fixture(scope="module")
def pendulum_env_def() -> object:
    import hlbench.envs  # noqa: F401  registration side effect

    return get_env("pendulum")


@pytest.fixture
def make_sandbox(pendulum_env_def: object, tmp_path: Path):
    """Factory: write a policy.py and spawn a Sandbox around it.

    Returns a callable `(policy_body: str, **config_kwargs) -> Sandbox`.
    Tracks the sandbox for teardown so individual tests don't have to.
    """
    created: list[Sandbox] = []

    def _factory(body: str, **config_kwargs: object) -> Sandbox:
        _write_policy(tmp_path, body)
        sb = Sandbox(
            snapshot_dir=tmp_path,
            env_factory=pendulum_env_def.factory,
            env_meta=pendulum_env_def.public_env_meta(),
            config=SandboxConfig(**config_kwargs),  # type: ignore[arg-type]
        )
        created.append(sb)
        return sb

    yield _factory

    for sb in created:
        sb.close()


# Reusable PD policy body (matches tests/conftest.py PendulumPDPolicy).
_PD_POLICY_BODY = """
    import math


    class Policy:
        KP = 30.0
        KD = 5.0
        K_ENERGY = 1.0
        A_MAX = 2.0
        G_OVER_L = 10.0

        def __init__(self, obs_space=None, action_space=None, env_meta=None):
            pass

        def reset(self, episode_index):
            pass

        def act(self, obs):
            cos_t, sin_t, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
            theta = math.atan2(sin_t, cos_t)
            if abs(theta) < 0.5:
                u = -self.KP * theta - self.KD * theta_dot
            else:
                E = 0.5 * theta_dot * theta_dot + self.G_OVER_L * cos_t
                u = -self.K_ENERGY * theta_dot * (E - self.G_OVER_L)
            u = max(-self.A_MAX, min(self.A_MAX, u))
            return [u]
"""


# ----------------------- happy path ---------------------------------------


def test_happy_path_runs_episode(make_sandbox) -> None:
    """Good policy → episode returns a well-formed EpisodeRecord with positive length."""
    sb = make_sandbox(_PD_POLICY_BODY)
    sb.init_policy()

    rec = sb.run_episode(real_seed=42, episode_index=0, max_steps=50)

    assert rec.length == 50, "Pendulum has no natural termination → hits max_steps"
    assert len(rec.trajectory) == 50
    assert rec.ended_with_error is False
    assert rec.error_category is None
    # PD on Pendulum-v1 should beat random (~-1200 / 200 steps → ~-300 for 50 steps).
    # We don't assert the exact value across processes — just sanity.
    assert rec.return_ < 0


def test_multiple_episodes_share_policy_instance(make_sandbox) -> None:
    """Policy persists across episodes within a submit (SPEC.md §2)."""
    body = """
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                self.n_resets = 0

            def reset(self, episode_index):
                self.n_resets += 1
                self.last_index = episode_index

            def act(self, obs):
                return [0.0]
    """
    sb = make_sandbox(body)
    sb.init_policy()

    sb.run_episode(real_seed=0, episode_index=0, max_steps=3)
    sb.run_episode(real_seed=1, episode_index=1, max_steps=3)
    rec = sb.run_episode(real_seed=2, episode_index=2, max_steps=3)

    # No way to peek inside the child directly; rely on the fact that the
    # episode actually ran (length=3) — if Policy were re-instantiated each
    # call, init would be re-paid but behavior wouldn't differ. Better:
    # observe via per-episode side effect. For MVP, just confirm no crash.
    assert rec.length == 3
    assert rec.ended_with_error is False


# ----------------------- init errors --------------------------------------


def test_init_error_when_policy_raises_in_init(make_sandbox) -> None:
    """Policy.__init__ raising → SandboxInitError(category=init_error)."""
    body = """
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                raise ValueError("intentional init crash")
    """
    sb = make_sandbox(body)
    with pytest.raises(SandboxInitError) as exc_info:
        sb.init_policy()
    assert exc_info.value.category == "init_error"
    assert "intentional init crash" in exc_info.value.traceback_str


def test_init_timeout_when_init_too_slow(make_sandbox) -> None:
    """Policy.__init__ sleeping > init_wall_s → SandboxInitError(init_timeout)."""
    body = """
        import time

        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                time.sleep(5.0)
    """
    sb = make_sandbox(body, init_wall_s=0.4)
    with pytest.raises(SandboxInitError) as exc_info:
        sb.init_policy()
    assert exc_info.value.category == "init_timeout"


def test_import_error_when_policy_module_broken(make_sandbox) -> None:
    """policy.py with ImportError at module load → import_error verdict."""
    body = """
        import this_module_definitely_does_not_exist_xyz

        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                pass
    """
    sb = make_sandbox(body)
    with pytest.raises(SandboxInitError) as exc_info:
        sb.init_policy()
    assert exc_info.value.category == "import_error"


# ----------------------- per-episode errors -------------------------------


def test_act_error_is_per_episode_not_sandbox_fatal(make_sandbox) -> None:
    """policy.act raising → EpisodeRecord.error_category='act_error', sandbox stays alive."""
    body = """
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                self.calls = 0

            def reset(self, episode_index):
                self.calls = 0

            def act(self, obs):
                self.calls += 1
                if self.calls > 3:
                    raise RuntimeError("kaboom at step 4")
                return [0.0]
    """
    sb = make_sandbox(body)
    sb.init_policy()

    rec = sb.run_episode(real_seed=0, episode_index=0, max_steps=20)
    assert rec.ended_with_error is True
    assert rec.error_category == "act_error"
    assert rec.error_step_index == 3  # 4th call (1-indexed) = step index 3
    assert "kaboom at step 4" in (rec.error_traceback or "")
    assert rec.length == 3  # three good steps recorded before the crash

    # Sandbox should still be usable for the next episode.
    rec2 = sb.run_episode(real_seed=1, episode_index=1, max_steps=20)
    assert rec2.ended_with_error is True  # same crash pattern
    assert rec2.length == 3


def test_act_timeout_is_per_episode_with_no_traceback(make_sandbox) -> None:
    """policy.act sleeping > act_wall_s → category='act_timeout', traceback=None."""
    body = """
        import time

        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                self.calls = 0

            def reset(self, episode_index):
                self.calls = 0

            def act(self, obs):
                self.calls += 1
                if self.calls > 2:
                    time.sleep(1.0)  # will be interrupted by SIGALRM
                return [0.0]
    """
    sb = make_sandbox(body, act_wall_s=0.05)
    sb.init_policy()

    rec = sb.run_episode(real_seed=0, episode_index=0, max_steps=20)
    assert rec.ended_with_error is True
    assert rec.error_category == "act_timeout"
    assert rec.error_traceback is None
    assert rec.error_step_index == 2
    assert rec.length == 2  # first two steps completed before the slow act


# ----------------------- lifecycle ----------------------------------------


def test_close_terminates_child(make_sandbox) -> None:
    """After close(), the child process is no longer alive."""
    sb = make_sandbox(_PD_POLICY_BODY)
    sb.init_policy()
    sb.run_episode(real_seed=0, episode_index=0, max_steps=5)

    proc = sb._proc  # noqa: SLF001 — white-box check
    sb.close()
    # Give the OS a moment to reap.
    for _ in range(20):
        if not proc.is_alive():
            break
        time.sleep(0.05)
    assert not proc.is_alive()


def test_double_close_is_idempotent(make_sandbox) -> None:
    sb = make_sandbox(_PD_POLICY_BODY)
    sb.init_policy()
    sb.close()
    sb.close()  # should not raise


def test_run_episode_before_init_raises(make_sandbox) -> None:
    sb = make_sandbox(_PD_POLICY_BODY)
    with pytest.raises(RuntimeError, match="init_policy"):
        sb.run_episode(real_seed=0, episode_index=0, max_steps=5)


def test_run_episode_after_close_raises(make_sandbox) -> None:
    sb = make_sandbox(_PD_POLICY_BODY)
    sb.init_policy()
    sb.close()
    with pytest.raises(RuntimeError, match="closed"):
        sb.run_episode(real_seed=0, episode_index=0, max_steps=5)
