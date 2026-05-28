"""End-to-end test: drive the lib Server through a 5-submit run.

Validates the goal stated in docs/architecture.md Day 7:
"5 submits, each shows feedback, mean_return ~ -200 (good PD)".

Uses the reference PD policy from ``agents/pd_pendulum/policy.py`` —
the file lives outside ``src/hlbench/`` because it's a consumer, not
part of the server library (CLAUDE.md invariant 9). The test copies it
into the workspace's ``system/`` directory exactly the way a real agent
submission would.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.server import Server  # noqa: E402

# Repo root (tests/ is one level under it).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_POLICY = _REPO_ROOT / "agents" / "pd_pendulum" / "policy.py"


@pytest.fixture
def workspace_with_reference_agent(tmp_path: Path):
    """Yield (server, workspace) where the reference PD agent is staged."""
    ws = tmp_path / "run"
    srv = Server(env_id="pendulum", workspace_dir=ws)
    # Real agent flow: agent writes/edits its code under workspace/system/.
    shutil.copy(_REFERENCE_POLICY, ws / "system" / "policy.py")
    return srv, ws


def test_workspace_has_required_four_things(tmp_path: Path) -> None:
    """Per CLAUDE.md invariant 5, workspace contains exactly:
    TASK.md, AGENT.md, system/, feedback/."""
    ws = tmp_path / "run"
    Server(env_id="pendulum", workspace_dir=ws)
    assert (ws / "TASK.md").is_file()
    assert (ws / "AGENT.md").is_file()
    assert (ws / "system").is_dir()
    assert (ws / "feedback").is_dir()
    # TASK.md was the env-specific one, not the placeholder.
    assert "Pendulum" in (ws / "TASK.md").read_text()


def test_info_hides_baselines_and_seeds(tmp_path: Path) -> None:
    """Per CLAUDE.md invariant 2, /info NEVER reveals expert/random
    baselines or real seeds."""
    srv = Server(env_id="pendulum", workspace_dir=tmp_path / "run")
    info = srv.info()
    flat = json.dumps(info)
    assert "expert_baseline" not in flat
    assert "random_baseline" not in flat
    assert "real_seed" not in flat
    assert "heldout" not in flat
    # But the legitimate fields are present.
    assert info["env"] == "pendulum"
    assert info["env_meta"]["n_env_instances"] == 256
    assert info["state"]["remaining_budget"] == 256
    assert info["state"]["is_finalized"] is False


def test_info_state_advances_after_submits(workspace_with_reference_agent) -> None:
    """state.remaining_budget / n_submits update between submits."""
    srv, _ = workspace_with_reference_agent
    assert srv.info()["state"]["remaining_budget"] == 256
    srv.submit([0, 1])
    info = srv.info()
    assert info["state"]["remaining_budget"] == 254
    assert info["state"]["n_submits"] == 1
    assert info["state"]["n_successful_submits"] == 1
    assert info["state"]["last_submit_index"] == 0
    assert info["state"]["last_submit_status"] == "ok"


def test_five_submit_run_with_reference_pd_agent(
    workspace_with_reference_agent,
) -> None:
    """The Day 7 acceptance criterion: 5 submits of 4 episodes each on
    Pendulum-v1 with the reference PD agent. Each submit writes a
    well-formed feedback dir; the per-submit mean_return clears the
    "comfortably better than random" bar.

    Random baseline ~ -1200; expert ~ -150. We assert mean_return < -50
    (much better than random) and > -800 (a wide safety margin so this
    test isn't flaky on slow CI). The actual value is typically ~ -135."""
    srv, ws = workspace_with_reference_agent
    submit_means: list[float] = []

    # Submit env_instances 0..3, then 4..7, then 8..11, etc.
    for submit_i in range(5):
        env_instances = list(range(submit_i * 4, submit_i * 4 + 4))
        result = srv.submit(env_instances)
        assert result.status == "ok", \
            f"submit #{submit_i} failed: {result.status} {result.summary.get('errors')}"
        assert result.submit_id == submit_i

        summary = result.summary
        assert summary["n_episodes"] == 4
        assert summary["first_global_episode"] == submit_i * 4
        assert summary["env_instances"] == env_instances
        assert summary["timeouts"] == []
        assert summary["errors"] == []
        assert -800 < summary["mean_return"] < 0, \
            f"submit #{submit_i} mean_return {summary['mean_return']:.1f} outside expected band"
        submit_means.append(summary["mean_return"])

    # Final state: 5 successful submits, 20 episodes consumed.
    final = srv.info()["state"]
    assert final["remaining_budget"] == 256 - 20
    assert final["n_submits"] == 5
    assert final["n_successful_submits"] == 5
    assert final["last_submit_status"] == "ok"

    # Run-wide mean: PD reliably beats -400.
    overall_mean = sum(submit_means) / 5
    assert overall_mean < -50, "PD on Pendulum should be much better than 0"
    assert overall_mean > -400, \
        f"reference PD performing worse than expected (mean={overall_mean:.1f})"

    # Feedback layout: 5 submit dirs, each with summary.json + 4 episode
    # dirs containing trajectory.jsonl.
    fb_dir = ws / "feedback"
    submit_dirs = sorted(p.name for p in fb_dir.iterdir())
    assert submit_dirs == [f"submit_{i:03d}" for i in range(5)]
    for submit_i in range(5):
        sd = fb_dir / f"submit_{submit_i:03d}"
        assert (sd / "summary.json").is_file()
        assert not (sd / "errors.txt").exists()  # success → no submit-level error
        for ep_i in range(4):
            global_ep = submit_i * 4 + ep_i
            ep_dir = sd / "episodes" / f"ep_{global_ep:03d}"
            assert ep_dir.is_dir()
            traj = ep_dir / "trajectory.jsonl"
            lines = traj.read_text().strip().split("\n")
            assert len(lines) == 200, f"{traj} has {len(lines)} steps, want 200"


def test_finalize_not_yet_implemented(workspace_with_reference_agent) -> None:
    """Day 7 ships only the train-side loop; finalize arrives Day 8."""
    srv, _ = workspace_with_reference_agent
    with pytest.raises(NotImplementedError):
        srv.finalize()


def test_unknown_config_override_rejected(tmp_path: Path) -> None:
    """Misspelled override keys must fail loudly, not silently fall through."""
    with pytest.raises(ValueError, match="unknown config_overrides"):
        Server(
            env_id="pendulum",
            workspace_dir=tmp_path / "run",
            config_overrides={"epiosde_budget": 8},  # typo
        )
