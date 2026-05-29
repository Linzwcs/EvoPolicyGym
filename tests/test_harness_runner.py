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


def test_budget_exhausted_terminates_naturally(server_with_policy: Server) -> None:
    """Preferred path: agent uses the whole budget, harness auto-finalizes.

    budget=8 with 4 episodes per submit → 2 submits drains the budget.
    After turn 1 (the 2nd submit), remaining_budget==0 so the loop ends
    with termination_reason=budget_exhausted. The agent never calls
    /finalize itself."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _submit([4, 5, 6, 7]),
        # max_turns=10 leaves headroom; loop must STILL break at turn 2
        # because budget is gone.
        _submit([0, 1, 2, 3]),  # never executed
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
    )
    summary = runner.run()
    assert summary.termination_reason == "budget_exhausted"
    assert summary.n_turns == 2
    assert agent.turn_count == 2  # third action never invoked
    assert summary.final_result is not None
    assert summary.final_result["status"] == "completed"
    assert summary.final_result["final_score"] is not None


def test_max_turns_caps_the_loop_when_budget_unused(server_with_policy: Server) -> None:
    """Safety net: agent does nothing useful → max_turns ends the loop,
    harness still auto-finalizes."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),  # 4 of 8 budget
        # No more submits; budget never reaches 0.
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=4,
    )
    summary = runner.run()
    assert summary.termination_reason == "max_turns"
    assert summary.n_turns == 4  # all 4 turns ran (agent submits once, noops 3x)
    assert summary.final_result is not None
    assert summary.final_result["status"] == "completed"


def test_consecutive_failures_break_loop(server_with_policy: Server) -> None:
    """3 consecutive failed turns → loop bails with the
    ``consecutive_failures`` reason; harness still auto-finalizes."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _fail, _fail, _fail,
        _submit([0, 1, 2, 3]),  # would succeed but loop bailed
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
        max_consecutive_failures=3,
    )
    summary = runner.run()
    assert summary.termination_reason == "consecutive_failures"
    assert summary.n_turns == 3


def test_failure_counter_resets_on_success(server_with_policy: Server) -> None:
    """A successful turn between failures keeps the loop alive until
    budget hits 0."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _fail,
        _submit([0, 1, 2, 3]),  # resets counter; budget now 4 of 8
        _fail,
        _fail,
        _submit([4, 5, 6, 7]),  # drains budget
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
        max_consecutive_failures=3,
    )
    summary = runner.run()
    assert summary.termination_reason == "budget_exhausted"
    assert summary.n_turns == 5


def test_agent_finalized_honored_defensively(server_with_policy: Server) -> None:
    """The prompt tells the agent NOT to call /finalize, but if the
    agent does anyway we honor it (Server.finalize is idempotent)."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _finalize_action,  # agent prematurely finalizes despite prompt
        _submit([4, 5, 6, 7]),  # never executes — run is over
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://test", max_turns=10,
    )
    summary = runner.run()
    assert summary.termination_reason == "agent_finalized"
    assert summary.n_turns == 2
    assert summary.final_result is not None
    assert summary.final_result["status"] == "completed"


def test_initial_prompt_then_continuations(server_with_policy: Server) -> None:
    """Turn 0 receives the initial prompt (with task md / info json /
    rules); turn 1+ receive the much shorter continuation prompt."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _submit([4, 5, 6, 7]),
    ])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://x:1", max_turns=10,
    )
    runner.run()

    initial = agent.prompts_seen[0]
    cont1 = agent.prompts_seen[1]

    # Initial: full task / rules / info.
    assert "GET /info" in initial or '"episode_budget"' in initial
    assert "transformers" in initial  # rules excerpt
    assert str(server_with_policy.workspace_dir) in initial
    # Initial prompt must NOT instruct the agent to call /finalize —
    # the harness handles that.
    assert "POST /finalize" not in initial
    assert "POST ``/finalize``" not in initial

    # Continuation: concise turn header + last submit recap.
    assert "Turn 1/" in cont1
    assert "remaining_budget" in cont1
    # Continuation should NOT re-embed the full task/rules.
    assert len(cont1) < len(initial) // 2


def test_summary_persisted_to_logs(server_with_policy: Server) -> None:
    """``logs/harness_runner.json`` is written for analyst tooling."""
    agent = FakeAgent(server=server_with_policy, actions=[
        _submit([0, 1, 2, 3]),
        _submit([4, 5, 6, 7]),
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
    assert data["termination_reason"] == "budget_exhausted"
    assert isinstance(data["turns"], list)
    assert len(data["turns"]) == 2
    # Each turn entry has post-turn server state snapshot.
    assert "remaining_budget" in data["turns"][0]["state_after"]


def test_no_submits_at_all_still_finalizes_without_crashing(server_with_policy: Server) -> None:
    """Agent never submits anything → max_turns ends the loop; harness
    still auto-finalizes (workspace has the reference PD policy staged
    so heldout runs)."""
    agent = FakeAgent(server=server_with_policy, actions=[_noop, _noop])
    runner = HarnessRunner(
        server=server_with_policy, agent=agent,
        http_url="http://x", max_turns=2,
    )
    summary = runner.run()
    assert summary.termination_reason == "max_turns"
    assert summary.final_result is not None
    # Pendulum reference PD will score normally on held-out even with
    # zero successful submits; final_submit_index is None but final_score
    # comes from the staged policy.
    assert summary.final_result["status"] == "completed"
    assert summary.final_result["final_submit_index"] is None
