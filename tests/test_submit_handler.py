"""SubmitHandler integration tests — covers the 7-phase lifecycle end-to-end.

Each test sets up a tmp workspace (workspace/system/policy.py + empty
feedback/), runs `SubmitHandler.handle()` with a real Sandbox + Pendulum
env, and asserts on the on-disk feedback layout.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.sandbox import SandboxConfig  # noqa: E402
from hlbench.core.seed_resolver import SeedResolver  # noqa: E402
from hlbench.core.submit_handler import (  # noqa: E402
    SubmitConfig,
    SubmitHandler,
    SubmitState,
    _SubmitOutcome,
)
from hlbench.envs.registry import get_env  # noqa: E402

# -------------------- fixtures --------------------------------------------


@pytest.fixture(scope="module")
def pendulum_env_def():
    import hlbench.envs  # noqa: F401  side-effect: registration

    return get_env("pendulum")


def _write_workspace(tmp_path: Path, policy_body: str | None) -> Path:
    """Create tmp workspace with system/policy.py (or no system/ at all)
    plus an empty feedback/. Returns the workspace dir."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "feedback").mkdir()
    if policy_body is not None:
        (ws / "system").mkdir()
        (ws / "system" / "policy.py").write_text(textwrap.dedent(policy_body).lstrip())
    return ws


def _make_handler(
    pendulum_env_def,
    workspace: Path,
    *,
    episode_budget: int = 256,
    max_per_submit: int = 256,
    act_wall_s: float = 1.0,
    init_wall_s: float = 30.0,
    episode_wall_s: float = 60.0,
    submit_wall_s: float = 300.0,
) -> SubmitHandler:
    sm = SeedResolver(
        pendulum_env_def.train_seeds_path,
        pendulum_env_def.heldout_seeds_path,
    )
    return SubmitHandler(
        env_def=pendulum_env_def,
        seed_resolver=sm,
        workspace_dir=workspace,
        config=SubmitConfig(
            episode_budget=episode_budget,
            max_episodes_per_submit=max_per_submit,
            submit_wall_s=submit_wall_s,
            sandbox=SandboxConfig(
                act_wall_s=act_wall_s,
                init_wall_s=init_wall_s,
                episode_wall_s=episode_wall_s,
            ),
        ),
    )


_GOOD_PD_POLICY = """
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


# -------------------- happy path ------------------------------------------


def test_ok_path_writes_full_feedback_layout(tmp_path, pendulum_env_def):
    """4 PD episodes → status=ok, summary fields right, trajectory files valid."""
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=256)
    outcome = handler.handle([0, 1, 2, 3], state)

    assert isinstance(outcome, _SubmitOutcome)
    assert outcome.status == "ok"
    assert outcome.submit_index == 0

    submit_dir = ws / "feedback" / "submit_000"
    assert submit_dir.is_dir()

    summary_path = submit_dir / "summary.json"
    summary = json.loads(summary_path.read_text())

    # Schema fields per SPEC §4.1.
    assert summary["schema_version"] == "0.1"
    assert summary["submit_index"] == 0
    assert summary["env"] == "pendulum"
    assert summary["status"] == "ok"
    assert summary["n_episodes"] == 4
    assert summary["first_global_episode"] == 0
    assert summary["env_instances"] == [0, 1, 2, 3]
    assert summary["remaining_budget"] == 256 - 4
    assert summary["timeouts"] == []
    assert summary["errors"] == []
    assert len(summary["returns"]) == 4
    assert len(summary["episode_lengths"]) == 4
    assert summary["mean_return"] == pytest.approx(
        sum(summary["returns"]) / 4, rel=1e-9
    )
    assert summary["min_return"] == min(summary["returns"])
    assert summary["max_return"] == max(summary["returns"])
    # PD on Pendulum reliably beats -800 (random ~ -1200).
    assert summary["mean_return"] > -800

    # No submit-level errors.txt on success (SPEC §4.4.4 mutual exclusion).
    assert not (submit_dir / "errors.txt").exists()

    # Episodes laid out as ep_000..ep_003 (width 3 since budget=256).
    episodes_dir = submit_dir / "episodes"
    assert sorted(p.name for p in episodes_dir.iterdir()) == [
        "ep_000", "ep_001", "ep_002", "ep_003"
    ]
    for i in range(4):
        traj_path = episodes_dir / f"ep_{i:03d}" / "trajectory.jsonl"
        assert traj_path.exists()
        # Pendulum has no natural termination → 200 steps per episode.
        lines = traj_path.read_text().strip().split("\n")
        assert len(lines) == 200
        # Each line is valid JSON with the required schema keys.
        first = json.loads(lines[0])
        for k in ("t", "obs", "action", "reward", "terminated", "truncated", "info"):
            assert k in first
        assert first["t"] == 0
        assert isinstance(first["obs"], list) and len(first["obs"]) == 3
        # No per-episode error.txt on success.
        assert not (episodes_dir / f"ep_{i:03d}" / "error.txt").exists()

    # State advanced correctly.
    assert outcome.new_state.remaining_budget == 252
    assert outcome.new_state.n_submits == 1
    assert outcome.new_state.n_successful_submits == 1
    assert outcome.new_state.n_episodes_executed == 4
    assert outcome.new_state.last_submit_index == 0
    assert outcome.new_state.last_submit_status == "ok"


def test_consecutive_submits_advance_global_episode_counter(
    tmp_path, pendulum_env_def
):
    """Second submit's first_global_episode = first submit's n_episodes."""
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=256)
    out1 = handler.handle([0, 1], state)
    out2 = handler.handle([2, 3, 4], out1.new_state)

    assert out1.summary["first_global_episode"] == 0
    assert out2.summary["first_global_episode"] == 2
    assert out2.submit_index == 1
    # Episode dirs use the global index across submits.
    assert (ws / "feedback" / "submit_001" / "episodes" / "ep_002").is_dir()
    assert (ws / "feedback" / "submit_001" / "episodes" / "ep_004").is_dir()
    # No collision with submit_000's episodes.
    assert not (ws / "feedback" / "submit_001" / "episodes" / "ep_000").exists()
    assert out2.new_state.remaining_budget == 256 - 2 - 3


# -------------------- Phase 1 failures (no budget consumed) ---------------


def test_invalid_env_instance(tmp_path, pendulum_env_def):
    """env_instance 999 → status=invalid_env_instance, budget UNCHANGED."""
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=256)
    outcome = handler.handle([0, 999, 1], state)

    assert outcome.status == "invalid_env_instance"
    submit_dir = ws / "feedback" / "submit_000"
    summary = json.loads((submit_dir / "summary.json").read_text())
    assert summary["status"] == "invalid_env_instance"
    assert summary["returns"] is None
    assert summary["first_global_episode"] is None
    # Budget UNCHANGED on Phase 1 failure.
    assert summary["remaining_budget"] == 256
    assert outcome.new_state.remaining_budget == 256
    assert outcome.new_state.n_successful_submits == 0
    assert outcome.new_state.n_episodes_executed == 0
    assert outcome.new_state.n_submits == 1
    assert outcome.new_state.last_submit_status == "invalid_env_instance"

    # No episodes/ dir; one-line errors.txt with category invalid_env_instance.
    assert not (submit_dir / "episodes").exists()
    err = json.loads((submit_dir / "errors.txt").read_text().strip())
    assert err["category"] == "invalid_env_instance"
    assert "999" in err["message"]


def test_budget_invalid_when_over_remaining(tmp_path, pendulum_env_def):
    """Requesting 5 episodes with remaining=3 → budget_invalid, no consume."""
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=3)
    outcome = handler.handle([0, 1, 2, 3, 4], state)

    assert outcome.status == "budget_invalid"
    assert outcome.new_state.remaining_budget == 3  # unchanged
    err = json.loads(
        (ws / "feedback" / "submit_000" / "errors.txt").read_text().strip()
    )
    assert err["category"] == "budget_invalid"


def test_budget_invalid_when_over_max_per_submit(tmp_path, pendulum_env_def):
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws, max_per_submit=2)

    state = SubmitState(remaining_budget=256)
    outcome = handler.handle([0, 1, 2], state)
    assert outcome.status == "budget_invalid"
    assert outcome.new_state.remaining_budget == 256


# -------------------- Phase 2-5 failures (budget consumed) ---------------


def test_missing_policy_file(tmp_path, pendulum_env_def):
    """No system/policy.py → missing_policy verdict, budget consumed."""
    ws = _write_workspace(tmp_path, policy_body=None)  # no system/ at all
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=256)
    outcome = handler.handle([0, 1], state)

    assert outcome.status == "missing_policy"
    submit_dir = ws / "feedback" / "submit_000"
    summary = json.loads((submit_dir / "summary.json").read_text())
    assert summary["status"] == "missing_policy"
    assert summary["returns"] is None
    # Budget IS consumed once we've passed Phase 1.
    assert summary["remaining_budget"] == 256 - 2
    assert outcome.new_state.remaining_budget == 256 - 2
    assert outcome.new_state.n_successful_submits == 0
    assert not (submit_dir / "episodes").exists()
    err = json.loads((submit_dir / "errors.txt").read_text().strip())
    assert err["category"] == "missing_policy"


def test_init_error(tmp_path, pendulum_env_def):
    """Policy.__init__ raises → init_error, budget consumed."""
    body = """
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                raise ValueError("no good")
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0, 1], state)

    assert outcome.status == "init_error"
    summary = json.loads(
        (ws / "feedback" / "submit_000" / "summary.json").read_text()
    )
    assert summary["status"] == "init_error"
    assert summary["remaining_budget"] == 8
    err = json.loads(
        (ws / "feedback" / "submit_000" / "errors.txt").read_text().strip()
    )
    assert err["category"] == "init_error"
    assert "no good" in (err["traceback"] or "")


def test_init_timeout(tmp_path, pendulum_env_def):
    body = """
        import time

        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                time.sleep(5.0)
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws, init_wall_s=0.4)

    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0], state)
    assert outcome.status == "init_timeout"


def test_denied_import_verdict_and_errors_txt(tmp_path, pendulum_env_def):
    """Policy importing a denied module → status=denied_import, budget
    consumed (Phase 4 failure), errors.txt carries the verdict."""
    body = """
        import transformers  # denied per AGENTS.md §3.2

        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                pass
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0, 1, 2], state)

    assert outcome.status == "denied_import"
    summary = outcome.summary
    assert summary["status"] == "denied_import"
    assert summary["returns"] is None
    # Phase 4 (Compile) failure: full N consumed.
    assert summary["remaining_budget"] == 7
    assert outcome.new_state.remaining_budget == 7
    submit_dir = ws / "feedback" / "submit_000"
    assert not (submit_dir / "episodes").exists()
    err = json.loads((submit_dir / "errors.txt").read_text().strip())
    assert err["category"] == "denied_import"
    assert "transformers" in (err["traceback"] or "")


# -------------------- Phase 6: per-episode errors -------------------------


def test_act_error_marks_episode_in_errors_array(tmp_path, pendulum_env_def):
    """policy.act raises mid-episode → submit OK, episode in summary.errors[]."""
    body = """
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                self.calls = 0
            def reset(self, episode_index):
                self.calls = 0
                self.ep = episode_index
            def act(self, obs):
                self.calls += 1
                # Crash only in episode index 1, after 5 steps.
                if self.ep == 1 and self.calls > 5:
                    raise RuntimeError("crash at step 6")
                return [0.0]
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=256)
    outcome = handler.handle([0, 1, 2], state)

    assert outcome.status == "ok"
    summary = outcome.summary
    assert summary["errors"] == [1]
    assert summary["timeouts"] == []
    # Episode 1 has both trajectory and error.txt.
    ep1 = ws / "feedback" / "submit_000" / "episodes" / "ep_001"
    assert (ep1 / "trajectory.jsonl").exists()
    err = json.loads((ep1 / "error.txt").read_text().strip())
    assert err["category"] == "act_error"
    assert err["step_index"] == 5
    assert "crash at step 6" in (err["traceback"] or "")
    # Other episodes are clean.
    assert not (
        ws / "feedback" / "submit_000" / "episodes" / "ep_000" / "error.txt"
    ).exists()
    assert not (
        ws / "feedback" / "submit_000" / "episodes" / "ep_002" / "error.txt"
    ).exists()


def test_act_timeout_marks_episode_in_timeouts_array(tmp_path, pendulum_env_def):
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
                    time.sleep(1.0)  # interrupted by SIGALRM
                return [0.0]
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws, act_wall_s=0.05)

    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0, 1], state)

    assert outcome.status == "ok"
    assert outcome.summary["timeouts"] == [0, 1]
    assert outcome.summary["errors"] == []
    ep0 = ws / "feedback" / "submit_000" / "episodes" / "ep_000"
    err = json.loads((ep0 / "error.txt").read_text().strip())
    assert err["category"] == "act_timeout"
    assert err["traceback"] is None
    assert err["step_index"] == 2
    assert "wall time" in err["message"]


# -------------------- stdout / stderr capture (SPEC §4.5) ----------------


def test_stdout_stderr_always_created_even_when_empty(tmp_path, pendulum_env_def):
    """Every successful episode dir must contain stdout.txt and stderr.txt,
    even if the policy printed nothing (zero-byte files)."""
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0, 1], state)
    assert outcome.status == "ok"

    for global_ep in (0, 1):
        ep_dir = ws / "feedback" / "submit_000" / "episodes" / f"ep_{global_ep:03d}"
        assert (ep_dir / "stdout.txt").exists()
        assert (ep_dir / "stderr.txt").exists()
        # PD policy is silent; expect empty.
        assert (ep_dir / "stdout.txt").stat().st_size == 0
        assert (ep_dir / "stderr.txt").stat().st_size == 0


def test_policy_print_captured_in_stdout(tmp_path, pendulum_env_def):
    body = """
        import sys
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                pass
            def reset(self, episode_index):
                print(f"reset called with index={episode_index}")
                print("on stderr", file=sys.stderr)
            def act(self, obs):
                return [0.0]
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=5)
    outcome = handler.handle([0, 1], state)
    assert outcome.status == "ok"

    ep0_dir = ws / "feedback" / "submit_000" / "episodes" / "ep_000"
    ep1_dir = ws / "feedback" / "submit_000" / "episodes" / "ep_001"
    assert "reset called with index=0" in (ep0_dir / "stdout.txt").read_text()
    assert "on stderr" in (ep0_dir / "stderr.txt").read_text()
    # Each episode's capture is isolated (per SPEC §4.5).
    assert "reset called with index=0" not in (ep1_dir / "stdout.txt").read_text()
    assert "reset called with index=1" in (ep1_dir / "stdout.txt").read_text()


def test_init_print_folded_into_first_episode_stdout(tmp_path, pendulum_env_def):
    """Policy.__init__ runs once before episodes; its output goes into
    the FIRST episode's stdout (SPEC §4.5)."""
    body = """
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                print("hello from __init__")
            def reset(self, episode_index):
                pass
            def act(self, obs):
                return [0.0]
    """
    ws = _write_workspace(tmp_path, body)
    handler = _make_handler(pendulum_env_def, ws)

    state = SubmitState(remaining_budget=5)
    outcome = handler.handle([0, 1], state)
    assert outcome.status == "ok"

    ep0_dir = ws / "feedback" / "submit_000" / "episodes" / "ep_000"
    ep1_dir = ws / "feedback" / "submit_000" / "episodes" / "ep_001"
    assert "hello from __init__" in (ep0_dir / "stdout.txt").read_text()
    # Second episode doesn't re-capture init output.
    assert "hello from __init__" not in (ep1_dir / "stdout.txt").read_text()


# -------------------- submit_wall_s enforcement (SPEC §4.1) --------------


def test_submit_wall_exceeded_aborts_after_partial_completion(
    tmp_path, pendulum_env_def,
):
    """SPEC §4.1 + submit-protocol §3.3: when total Phase 6 wall time
    exceeds submit_wall_s, remaining episodes are aborted with verdict
    submit_wall_exceeded; episodes that already completed retain their
    artifacts and the partial summary."""
    body = """
        import time
        class Policy:
            def __init__(self, obs_space=None, action_space=None, env_meta=None):
                pass
            def reset(self, episode_index):
                # ~1.5s per episode → 2 episodes fit in a 3s budget,
                # 3rd trips the cap.
                time.sleep(1.5)
            def act(self, obs):
                return [0.0]
    """
    ws = _write_workspace(tmp_path, body)
    # episode_wall_s generous (must exceed sleep + 200 fast act() calls);
    # submit_wall_s tight (kills the loop after ~2 episodes).
    handler = _make_handler(
        pendulum_env_def, ws,
        episode_wall_s=10.0,
        submit_wall_s=3.0,
    )

    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0, 1, 2, 3, 4], state)

    assert outcome.status == "submit_wall_exceeded"
    # Full N consumed regardless of partial completion (SPEC §4.1 budget rule).
    assert outcome.new_state.remaining_budget == 5
    # At least 1 episode completed before the cap; not all 5.
    summary = outcome.summary
    assert summary["returns"] is not None
    assert 1 <= len(summary["returns"]) < 5, (
        f"unexpected partial count: {len(summary['returns'])}"
    )
    # Both feedback artifacts coexist: errors.txt + episodes/ (the only
    # verdict-pair where this is allowed, submit-protocol §3.3).
    submit_dir = ws / "feedback" / "submit_000"
    assert (submit_dir / "errors.txt").exists()
    assert (submit_dir / "episodes").is_dir()
    err = json.loads((submit_dir / "errors.txt").read_text().strip())
    assert err["category"] == "submit_wall_exceeded"
    assert "3.0s" in err["message"] or "3s" in err["message"]
    # Each completed episode has a full directory.
    n_completed = len(summary["returns"])
    assert sorted((submit_dir / "episodes").iterdir()) == sorted(
        (submit_dir / "episodes" / f"ep_{i:03d}") for i in range(n_completed)
    )


def test_submit_wall_s_default_does_not_fire_in_normal_run(
    tmp_path, pendulum_env_def,
):
    """Sanity: with default 300s submit_wall_s, a normal PD submit does
    NOT trip the cap."""
    ws = _write_workspace(tmp_path, _GOOD_PD_POLICY)
    handler = _make_handler(pendulum_env_def, ws)
    state = SubmitState(remaining_budget=10)
    outcome = handler.handle([0, 1, 2], state)
    assert outcome.status == "ok"
    assert outcome.summary["wall_time_seconds"] < 60.0  # well under 300s


# -------------------- new env smoke tests --------------------------------
# Each new env's starter_policy is auto-staged by Server, so an end-to-end
# submit through SubmitHandler exercises (1) env factory wiring,
# (2) action-space serialization (Discrete for acrobot, Box for mcc),
# (3) starter_policy import-and-instantiate path, (4) full episode loop.


@pytest.fixture(scope="module")
def acrobot_env_def():
    import hlbench.envs  # noqa: F401  registration side effect
    return get_env("acrobot")


@pytest.fixture(scope="module")
def mcc_env_def():
    import hlbench.envs  # noqa: F401  registration side effect
    return get_env("mountain_car_continuous")


def _stage_env_starter(env_def, tmp_path: Path) -> Path:
    """Mimic what Server.__init__ does for the starter: copy
    env.starter_policy_path into workspace/system/policy.py."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "feedback").mkdir()
    (ws / "system").mkdir()
    assert env_def.starter_policy_path is not None
    (ws / "system" / "policy.py").write_text(
        env_def.starter_policy_path.read_text()
    )
    return ws


def test_acrobot_discrete_action_smoke(tmp_path, acrobot_env_def):
    """Acrobot starter (no-op torque) runs 2 episodes to completion;
    trajectory records `action` as an int (Discrete serialization path)."""
    ws = _stage_env_starter(acrobot_env_def, tmp_path)
    handler = _make_handler(
        acrobot_env_def, ws,
        episode_budget=4, max_per_submit=4,
        # Acrobot can run up to 500 steps; give the submit room.
        submit_wall_s=120.0, episode_wall_s=60.0,
    )
    state = SubmitState(remaining_budget=4)
    outcome = handler.handle([0, 1], state)
    assert outcome.status == "ok"
    assert outcome.summary["n_episodes"] == 2
    # Each action in trajectory.jsonl is an int (Discrete serialization).
    traj_path = ws / "feedback" / "submit_000" / "episodes" / "ep_000" / "trajectory.jsonl"
    lines = traj_path.read_text().splitlines()
    first = json.loads(lines[0])
    assert isinstance(first["action"], int)
    assert first["action"] in (0, 1, 2)


def test_mountain_car_continuous_box_action_smoke(tmp_path, mcc_env_def):
    """MountainCarContinuous starter (u=0) runs 2 episodes; trajectory
    records `action` as a 1-elem list (Box[1] serialization)."""
    ws = _stage_env_starter(mcc_env_def, tmp_path)
    handler = _make_handler(
        mcc_env_def, ws,
        episode_budget=4, max_per_submit=4,
        # 999-step episodes; budget the wall time accordingly.
        submit_wall_s=180.0, episode_wall_s=120.0,
    )
    state = SubmitState(remaining_budget=4)
    outcome = handler.handle([0, 1], state)
    assert outcome.status == "ok"
    assert outcome.summary["n_episodes"] == 2
    traj_path = ws / "feedback" / "submit_000" / "episodes" / "ep_000" / "trajectory.jsonl"
    first = json.loads(traj_path.read_text().splitlines()[0])
    assert isinstance(first["action"], list)
    assert len(first["action"]) == 1
    assert -1.0 <= first["action"][0] <= 1.0
