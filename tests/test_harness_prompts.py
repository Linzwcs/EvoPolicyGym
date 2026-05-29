"""Prompt composition tests.

Pure functions; no I/O. The point is to pin the exact wording so any
drift surfaces in the diff (these prompts are the contract between the
harness and the inner Claude session)."""

from __future__ import annotations

from hlbench_harness.prompts import (
    AGENTS_MD_EXCERPT,
    compose_continuation_prompt,
    compose_initial_prompt,
)
from hlbench_harness.state import TurnObservation

# --------------------------- fixtures ------------------------------------


def _info_stub(remaining: int = 32, n_submits: int = 0, n_ok: int = 0) -> dict:
    return {
        "schema_version": "0.1",
        "env": "pendulum",
        "env_version": "0.1",
        "harness_version": "0.1.0a1",
        "agents_md_hash": "sha256:deadbeef",
        "episode_budget": 32,
        "min_episodes_per_submit": 1,
        "max_episodes_per_submit": 32,
        "resource_limits": {},
        "allowed_imports": [],
        "denied_imports": [],
        "env_meta": {"obs_space": {}, "action_space": {}, "max_episode_steps": 200,
                     "n_env_instances": 256, "obs_storage": "inline"},
        "state": {
            "remaining_budget": remaining,
            "n_submits": n_submits,
            "n_successful_submits": n_ok,
            "last_submit_index": (n_submits - 1) if n_submits else None,
            "last_submit_status": "ok" if n_ok else None,
            "submit_in_progress": False,
            "in_progress_submit_id": None,
            "is_finalized": False,
            "started_at": "2026-05-30T00:00:00.000Z",
        },
    }


# --------------------------- initial prompt ------------------------------


def test_initial_prompt_embeds_workspace_url_info_task_excerpt() -> None:
    prompt = compose_initial_prompt(
        workspace="/tmp/ws",
        http_url="http://127.0.0.1:8765",
        info=_info_stub(),
        task_md="# Pendulum\n\nSwing the pendulum upright.",
        agents_md_excerpt=AGENTS_MD_EXCERPT,
        max_turns=12,
    )
    # Workspace + URL must appear so the agent knows where to operate.
    assert "/tmp/ws" in prompt
    assert "http://127.0.0.1:8765" in prompt
    # Task description text from the env's TASK.md.
    assert "Swing the pendulum upright" in prompt
    # /info embedded as JSON (so first turn doesn't waste a fetch).
    assert '"episode_budget": 32' in prompt
    # Rules excerpt (transformers/anthropic block at a minimum).
    assert "transformers" in prompt
    # Operating instructions reference the three core tools.
    assert "/info" in prompt
    assert "/submit" in prompt
    assert "/finalize" in prompt
    # Max turns is surfaced so the agent can pace itself.
    assert "12" in prompt


def test_initial_prompt_stable_for_identical_inputs() -> None:
    """Composition is a pure function; same input → same string."""
    info = _info_stub()
    p1 = compose_initial_prompt(
        workspace="/w", http_url="http://h", info=info,
        task_md="task", agents_md_excerpt="rules", max_turns=8,
    )
    p2 = compose_initial_prompt(
        workspace="/w", http_url="http://h", info=info,
        task_md="task", agents_md_excerpt="rules", max_turns=8,
    )
    assert p1 == p2


# --------------------------- continuation prompt -------------------------


def _obs_with_last(last_submit: dict | None, remaining: int = 24, turn: int = 1) -> TurnObservation:
    info = _info_stub(remaining=remaining, n_submits=1 if last_submit else 0,
                     n_ok=1 if last_submit and last_submit.get("status") == "ok" else 0)
    return TurnObservation(
        turn_index=turn,
        info=info,
        submit_summaries=[last_submit] if last_submit else [],
    )


def test_continuation_no_submit_yet_says_so() -> None:
    obs = _obs_with_last(None, remaining=32, turn=1)
    prompt = compose_continuation_prompt(obs, max_turns=12)
    assert "Turn 1/12" in prompt
    assert "remaining_budget: 32" in prompt
    assert "No submits recorded yet" in prompt
    assert "Continue iterating" in prompt


def test_continuation_with_ok_submit_shows_mean() -> None:
    obs = _obs_with_last({
        "submit_index": 0,
        "status": "ok",
        "n_episodes": 4,
        "mean_return": -156.78,
        "std_return": 23.45,
        "timeouts": [],
        "errors": [],
    }, remaining=28, turn=2)
    prompt = compose_continuation_prompt(obs, max_turns=10)
    assert "Turn 2/10" in prompt
    assert "remaining_budget: 28" in prompt
    assert "#0 status=ok" in prompt
    assert "mean_return=-156.78" in prompt
    assert "std=23.45" in prompt
    assert "Continue iterating" in prompt


def test_continuation_with_failed_submit_points_to_errors_txt() -> None:
    obs = _obs_with_last({
        "submit_index": 1,
        "status": "init_error",
        "n_episodes": 4,
        "mean_return": None,
        "std_return": None,
    }, remaining=20, turn=3)
    prompt = compose_continuation_prompt(obs, max_turns=12)
    assert "status=init_error" in prompt
    assert "errors.txt" in prompt
    assert "no returns" in prompt


def test_continuation_budget_exhausted_nudges_finalize() -> None:
    obs = _obs_with_last({
        "submit_index": 7,
        "status": "ok",
        "n_episodes": 4,
        "mean_return": -120.0,
        "std_return": 10.0,
    }, remaining=0, turn=8)
    prompt = compose_continuation_prompt(obs, max_turns=12)
    assert "Budget exhausted" in prompt
    assert "POST /finalize" in prompt
    # Must NOT also tell agent to keep iterating.
    assert "Continue iterating" not in prompt


def test_continuation_when_already_finalized_says_nothing_to_do() -> None:
    # Manually flip is_finalized true on the info stub.
    info = _info_stub(remaining=0)
    info["state"]["is_finalized"] = True
    obs = TurnObservation(turn_index=9, info=info, submit_summaries=[{
        "submit_index": 0, "status": "ok", "n_episodes": 4,
        "mean_return": -130.0, "std_return": 5.0,
    }])
    prompt = compose_continuation_prompt(obs, max_turns=12)
    assert "already finalized" in prompt
