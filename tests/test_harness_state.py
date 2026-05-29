"""State-observer tests.

``observe()`` is the bridge between the live Server (in-memory) and
the on-disk per-submit ``summary.json`` files. The runner uses both
because the in-memory ``submit_history`` only keeps aggregates; the
detailed summaries (returns array, errors list) live in the feedback
dir."""

from __future__ import annotations

import json
from pathlib import Path

from hlbench_harness.state import TurnObservation, observe


def _write_summary(feedback_dir: Path, *, idx: int, status: str = "ok",
                   mean_return: float | None = -150.0, width: int = 3) -> None:
    """Drop a minimal summary.json on disk matching SPEC §4.1."""
    sd = feedback_dir / f"submit_{idx:0{width}d}"
    sd.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.1",
        "submit_index": idx,
        "env": "pendulum",
        "status": status,
        "n_episodes": 4,
        "first_global_episode": idx * 4 if status == "ok" else None,
        "env_instances": [0, 1, 2, 3],
        "remaining_budget": max(0, 32 - (idx + 1) * 4),
        "submit_started_at": "2026-05-30T00:00:00.000Z",
        "submit_completed_at": "2026-05-30T00:00:01.000Z",
        "wall_time_seconds": 1.0,
        "returns": [mean_return] * 4 if mean_return is not None else None,
        "mean_return": mean_return,
        "std_return": 0.0 if mean_return is not None else None,
        "min_return": mean_return,
        "max_return": mean_return,
        "episode_lengths": [200] * 4 if mean_return is not None else None,
        "mean_episode_length": 200.0 if mean_return is not None else None,
        "timeouts": [] if mean_return is not None else None,
        "errors": [] if mean_return is not None else None,
        "reward_components_mean": None,
        "reward_components_per_episode": None,
    }
    (sd / "summary.json").write_text(json.dumps(payload))


def test_observe_empty_workspace(tmp_path: Path) -> None:
    """No submits yet: feedback dir doesn't exist."""
    info = {"state": {"remaining_budget": 32, "n_submits": 0,
                      "n_successful_submits": 0, "is_finalized": False}}
    obs = observe(turn_index=0, info=info, workspace=tmp_path)
    assert obs.turn_index == 0
    assert obs.remaining_budget == 32
    assert obs.is_finalized is False
    assert obs.submit_summaries == []
    assert obs.last_submit is None


def test_observe_reads_summaries_in_order(tmp_path: Path) -> None:
    """Multiple submits: summaries are loaded in submit_index order."""
    fb = tmp_path / "feedback"
    fb.mkdir()
    _write_summary(fb, idx=0, mean_return=-200.0)
    _write_summary(fb, idx=1, mean_return=-150.0)
    _write_summary(fb, idx=2, mean_return=-120.0)
    info = {"state": {"remaining_budget": 20, "n_submits": 3,
                      "n_successful_submits": 3, "is_finalized": False}}
    obs = observe(turn_index=3, info=info, workspace=tmp_path)
    assert len(obs.submit_summaries) == 3
    assert [s["submit_index"] for s in obs.submit_summaries] == [0, 1, 2]
    assert obs.last_submit is not None
    assert obs.last_submit["submit_index"] == 2
    assert obs.last_submit["mean_return"] == -120.0


def test_observe_tolerates_partial_submit_dir(tmp_path: Path) -> None:
    """A submit_NNN/ without summary.json (impossible in practice but
    we shouldn't crash) is silently skipped."""
    fb = tmp_path / "feedback"
    fb.mkdir()
    _write_summary(fb, idx=0)
    (fb / "submit_001").mkdir()  # no summary.json
    _write_summary(fb, idx=2)
    info = {"state": {"remaining_budget": 24, "n_submits": 3,
                      "n_successful_submits": 2, "is_finalized": False}}
    obs = observe(turn_index=3, info=info, workspace=tmp_path)
    # Only the two with summary.json survive.
    assert [s["submit_index"] for s in obs.submit_summaries] == [0, 2]


def test_progress_line_no_submits() -> None:
    info = {"state": {"remaining_budget": 32, "n_submits": 0,
                      "n_successful_submits": 0, "is_finalized": False}}
    obs = TurnObservation(turn_index=0, info=info, submit_summaries=[])
    assert "remaining_budget=32" in obs.progress_line()
    assert "submits=0" in obs.progress_line()
    assert "last=none" in obs.progress_line()


def test_progress_line_with_ok_submit() -> None:
    info = {"state": {"remaining_budget": 28, "n_submits": 1,
                      "n_successful_submits": 1, "is_finalized": False}}
    obs = TurnObservation(turn_index=1, info=info, submit_summaries=[{
        "submit_index": 0, "status": "ok", "mean_return": -156.7,
    }])
    line = obs.progress_line()
    assert "remaining_budget=28" in line
    assert "submits=1 (1 ok)" in line
    assert "last=#0:ok" in line
    assert "mean_return=-156.70" in line


def test_progress_line_with_failed_submit_says_na() -> None:
    info = {"state": {"remaining_budget": 28, "n_submits": 1,
                      "n_successful_submits": 0, "is_finalized": False}}
    obs = TurnObservation(turn_index=1, info=info, submit_summaries=[{
        "submit_index": 0, "status": "init_error", "mean_return": None,
    }])
    line = obs.progress_line()
    assert "last=#0:init_error" in line
    assert "mean_return=n/a" in line
