"""HarnessRunner tests with a FakeAgent.

The runner integrates Server + AgentLike + state-observer. We don't
shell out to ``claude`` here; FakeAgent satisfies AgentLike and
performs scripted Server-side actions per turn (submits, finalize,
fail). This lets us test the loop control (termination conditions,
consecutive-failure cap, force-finalize, prompt routing) without an
LLM in the loop."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.server import Server  # noqa: E402
from hlbench_harness.runner import HarnessRunner  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_POLICY = _REPO_ROOT / "agents" / "pd_pendulum" / "policy.py"


# --------------------------- FakeAgent -----------------------------------


@dataclass
class _TurnResult:
    """Minimal AgentLike result; mirrors ClaudeAgent.TurnResult."""

    turn_index: int
    session_id: str
    exit_code: int = 0
    timed_out: bool = False
    duration_seconds: float = 0.01
    text: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# Action signatures: (server) -> bool. Return False to signal "agent
# turn was unhealthy" (will be reported as exit_code=1).
Action = Callable[[Server], bool]


class FakeAgent:
    """Test-only agent. Holds a list of actions; one per ``run_turn``.

    Each action receives the live Server and may submit / finalize on
    it, simulating what the inner Claude would have curl'd to do."""

    def __init__(self, *, server: Server, actions: list[Action], session_id: str = "test-uuid") -> None:
        self.session_id = session_id
        self.turn_count = 0
        self.prompts_seen: list[str] = []
        self._server = server
        self._actions = actions

    def run_turn(self, prompt: str) -> _TurnResult:
        idx = self.turn_count
        self.turn_count += 1
        self.prompts_seen.append(prompt)
        action = self._actions[idx] if idx < len(self._actions) else _noop
        ok = True
        try:
            ok = action(self._server)
        except Exception:  # pragma: no cover (test misconfiguration)
            ok = False
        return _TurnResult(
            turn_index=idx, session_id=self.session_id,
            exit_code=0 if ok else 1, timed_out=False, text=f"turn {idx}",
        )


def _noop(_srv: Server) -> bool:
    return True


def _submit(ids: list[int]) -> Action:
    def _do(srv: Server) -> bool:
        srv.submit(ids)
        return True
    return _do


def _finalize_action(srv: Server) -> bool:
    srv.finalize()
    return True


def _fail(_srv: Server) -> bool:
    return False


# --------------------------- fixtures -----------------------------------


@pytest.fixture
def server_with_policy(tmp_path: Path) -> Server:
    """Server with a small budget and the reference PD agent already
    staged so submits succeed."""
    srv = Server(
        env_id="pendulum",
        runs_root=tmp_path / "runs",
        config_overrides={"episode_budget": 8, "max_episodes_per_submit": 4},
    )
    shutil.copy(_REFERENCE_POLICY, srv.workspace_dir / "system" / "policy.py")
    return srv


# --------------------------- tests --------------------------------------


def test_agent_submits_then_finalizes(server_with_policy: Server) -> None:
    """Happy path: 2 submits + agent calls finalize → finalized_by_agent."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _submit([4, 5, 6, 7]),
        _finalize_action,
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
    )
    summary = runner.run()
    assert summary.finalized_by_agent is True
    assert summary.forced_finalize is False
    assert summary.max_turns_reached is False
    assert summary.n_turns == 3
    assert agent.turn_count == 3
    assert summary.final_result is not None
    assert summary.final_result["status"] == "completed"
    assert summary.final_result["final_score"] is not None


def test_force_finalize_when_max_turns_exhausted(server_with_policy: Server) -> None:
    """Agent never finalizes → harness force-finalizes after max_turns."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _submit([4, 5, 6, 7]),
        # No finalize; agent does nothing on turns 2+.
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=4,
    )
    summary = runner.run()
    assert summary.finalized_by_agent is False
    assert summary.forced_finalize is True
    assert summary.max_turns_reached is True
    assert summary.n_turns == 4  # all 4 turns executed
    # run.json was still written.
    assert summary.final_result is not None
    assert summary.final_result["status"] == "completed"


def test_consecutive_failures_break_loop(server_with_policy: Server) -> None:
    """3 consecutive failed turns → loop bails out (default cap=3)."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _fail, _fail, _fail,  # noqa: E501  three back-to-back failures
        _submit([0, 1, 2, 3]),  # would succeed but loop bailed
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
        max_consecutive_failures=3,
    )
    summary = runner.run()
    assert summary.consecutive_failures_hit is True
    assert summary.n_turns == 3
    # Force-finalize still fires; status reflects no successful submit.
    assert summary.forced_finalize is True


def test_failure_counter_resets_on_success(server_with_policy: Server) -> None:
    """A successful turn between failures keeps the loop alive."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _fail,
        _submit([0, 1, 2, 3]),  # resets counter
        _fail,
        _fail,
        _finalize_action,
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
        max_consecutive_failures=3,
    )
    summary = runner.run()
    assert summary.consecutive_failures_hit is False
    assert summary.finalized_by_agent is True
    assert summary.n_turns == 5


def test_initial_prompt_then_continuations(server_with_policy: Server) -> None:
    """Turn 0 receives the initial prompt (with task md / info json /
    rules); turn 1+ receive the much shorter continuation prompt."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _submit([4, 5, 6, 7]),
        _finalize_action,
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://x:1", max_turns=10,
    )
    runner.run()

    initial = agent.prompts_seen[0]
    cont1 = agent.prompts_seen[1]
    cont2 = agent.prompts_seen[2]

    # Initial: full task / rules / info.
    assert "GET /info" in initial or '"episode_budget"' in initial
    assert "transformers" in initial  # rules excerpt
    assert str(server_with_policy.workspace_dir) in initial

    # Continuations: concise turn header + last submit recap.
    assert "Turn 1/" in cont1
    assert "remaining_budget" in cont1
    assert "Turn 2/" in cont2
    # Continuations should NOT re-embed the full task/rules (they're way
    # shorter than the initial prompt).
    assert len(cont1) < len(initial) // 2
    assert len(cont2) < len(initial) // 2


def test_summary_persisted_to_logs(server_with_policy: Server) -> None:
    """``logs/harness_runner.json`` is written for analyst tooling."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _finalize_action,
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://x", max_turns=4,
    )
    summary = runner.run()
    persisted = server_with_policy.run_dir / "logs" / "harness_runner.json"
    assert persisted.is_file()
    data = json.loads(persisted.read_text())
    assert data["session_id"] == agent.session_id
    assert data["n_turns"] == summary.n_turns
    assert data["finalized_by_agent"] is True
    assert isinstance(data["turns"], list)
    assert len(data["turns"]) == 2
    # Each turn entry has post-turn server state snapshot.
    assert "remaining_budget" in data["turns"][0]["state_after"]


def test_no_submits_at_all_still_finalizes_without_crashing(server_with_policy: Server) -> None:
    """Agent never submits anything → force-finalize completes (the
    workspace has the reference PD policy staged so heldout runs fine)."""
    agent = FakeAgent(server=server_with_policy, actions=[_noop, _noop])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://x", max_turns=2,
    )
    summary = runner.run()
    assert summary.forced_finalize is True
    assert summary.final_result is not None
    # Pendulum reference PD will score normally on held-out even with
    # zero successful submits; final_submit_index is None but final_score
    # comes from the staged policy.
    assert summary.final_result["status"] == "completed"
    assert summary.final_result["final_submit_index"] is None
