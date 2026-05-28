"""Unit tests for hlbench.core.scoring (Day 11 metrics).

These are pure-function tests — no env, no sandbox, no I/O. They exist
to pin the math down so refactors of Server / SubmitHandler can't drift
the auxiliary numbers.
"""

from __future__ import annotations

import math

from hlbench.core.scoring import (
    auc_in_loop,
    build_auxiliary,
    episodes_to_threshold,
    final_score,
    normalized_score,
)
from hlbench.core.submit_handler import SubmitHistoryEntry

# Pendulum-style baselines: negative rewards, expert > random.
_EXPERT = -150.0
_RANDOM = -1200.0


# --------------------------- normalized_score ----------------------------


def test_normalized_score_basic_anchors() -> None:
    """Random maps to 0, expert to 1.0, beyond-expert > 1.0."""
    assert normalized_score(_RANDOM, expert=_EXPERT, random=_RANDOM) == 0.0
    assert normalized_score(_EXPERT, expert=_EXPERT, random=_RANDOM) == 1.0
    # Beat expert by half the (expert - random) gap → 1.5.
    super_expert = _EXPERT + 0.5 * (_EXPERT - _RANDOM)
    assert math.isclose(
        normalized_score(super_expert, expert=_EXPERT, random=_RANDOM), 1.5
    )


def test_normalized_score_below_random_is_negative() -> None:
    """No clipping in the raw function — callers decide."""
    assert normalized_score(-1500, expert=_EXPERT, random=_RANDOM) < 0


def test_normalized_score_degenerate_baselines_returns_zero() -> None:
    assert normalized_score(0, expert=10, random=10) == 0.0


# --------------------------- final_score ---------------------------------


def test_final_score_clips_at_zero_and_120() -> None:
    """SPEC §5.2: clamp to [0, 1.2] then × 100."""
    # Way below random → 0.
    assert final_score(-1e9, expert=_EXPERT, random=_RANDOM) == 0.0
    # Way above expert → 120.
    assert final_score(1e9, expert=_EXPERT, random=_RANDOM) == 120.0
    # Exactly expert → 100.
    assert final_score(_EXPERT, expert=_EXPERT, random=_RANDOM) == 100.0
    # Halfway → 50.
    halfway = (_EXPERT + _RANDOM) / 2
    assert final_score(halfway, expert=_EXPERT, random=_RANDOM) == 50.0


# --------------------------- episodes_to_threshold -----------------------


def _ok(idx: int, mean: float, cum: int) -> SubmitHistoryEntry:
    return SubmitHistoryEntry(
        submit_index=idx, status="ok", n_episodes=4,
        mean_return=mean, cumulative_episodes=cum,
    )


def _fail(idx: int, cum: int) -> SubmitHistoryEntry:
    return SubmitHistoryEntry(
        submit_index=idx, status="init_error", n_episodes=4,
        mean_return=None, cumulative_episodes=cum,
    )


def test_episodes_to_threshold_returns_first_crossing() -> None:
    history = [
        _ok(0, -1000, 4),   # normalized ~ 0.19  — below 50%
        _ok(1, -600,  8),   # normalized ~ 0.57  — first to cross 50%
        _ok(2, -300,  12),  # normalized ~ 0.86  — first to cross 80%
    ]
    assert episodes_to_threshold(
        history, 0.5, expert=_EXPERT, random=_RANDOM
    ) == 8
    assert episodes_to_threshold(
        history, 0.8, expert=_EXPERT, random=_RANDOM
    ) == 12


def test_episodes_to_threshold_none_when_never_reached() -> None:
    history = [_ok(0, -1100, 4), _ok(1, -1050, 8)]
    assert episodes_to_threshold(
        history, 0.8, expert=_EXPERT, random=_RANDOM
    ) is None


def test_episodes_to_threshold_skips_failed_submits() -> None:
    """Failed submits don't count even if they share cumulative_episodes."""
    history = [
        _fail(0, 0),       # init_error: no progress
        _ok(1, -150, 4),   # normalized = 1.0
    ]
    # The 50pct threshold fires at the first ok submit (index 1).
    assert episodes_to_threshold(
        history, 0.5, expert=_EXPERT, random=_RANDOM
    ) == 4


# --------------------------- auc_in_loop ---------------------------------


def test_auc_in_loop_constant_perfect_run_is_100() -> None:
    """If every submit hits expert, AUC averages to 100."""
    history = [
        _ok(0, _EXPERT, 10),
        _ok(1, _EXPERT, 20),
        _ok(2, _EXPERT, 30),
    ]
    # Trapezoid from (0,0) to (10,1) then flat to (30,1).
    # Area = 0.5*10*1 + 20*1 = 25 ; /30 episodes = 0.833 → 83.3.
    # The (0, 0) anchor is what costs us — perfect from-scratch is not 100.
    auc = auc_in_loop(history, expert=_EXPERT, random=_RANDOM)
    assert auc is not None
    assert 80 < auc < 90


def test_auc_in_loop_zero_progress_is_zero() -> None:
    """All submits at random baseline → curve is flat at 0 → AUC = 0."""
    history = [_ok(0, _RANDOM, 10), _ok(1, _RANDOM, 20)]
    assert auc_in_loop(history, expert=_EXPERT, random=_RANDOM) == 0.0


def test_auc_in_loop_no_successful_submits_is_none() -> None:
    history = [_fail(0, 0), _fail(1, 0)]
    assert auc_in_loop(history, expert=_EXPERT, random=_RANDOM) is None


def test_auc_in_loop_clips_negative_scores() -> None:
    """One disastrous submit at -3000 shouldn't make AUC negative."""
    history = [_ok(0, -3000, 5), _ok(1, _EXPERT, 10)]
    auc = auc_in_loop(history, expert=_EXPERT, random=_RANDOM)
    assert auc is not None and auc >= 0


# --------------------------- build_auxiliary -----------------------------


def test_build_auxiliary_complete_bundle() -> None:
    history = [
        _ok(0, -1000, 4),   # below 50%
        _ok(1, -500,  8),   # above 50%, below 80%
        _ok(2, -200,  12),  # above 80%
    ]
    aux = build_auxiliary(
        history,
        expert=_EXPERT, random=_RANDOM,
        held_out_mean=-180.0,
        n_submits=3, n_successful_submits=3, episodes_used=12,
    )
    assert aux["n_submits"] == 3
    assert aux["n_successful_submits"] == 3
    assert aux["episodes_used"] == 12
    assert aux["mean_episodes_per_submit"] == 4.0
    assert aux["episodes_to_50pct"] == 8
    assert aux["episodes_to_80pct"] == 12
    assert aux["held_out_gap"] == -20.0  # last in-loop -200 minus held-out -180
    assert aux["auc_in_loop"] is not None
    assert aux["mean_submit_wall_time"] is None  # not tracked yet


def test_build_auxiliary_no_held_out_no_gap() -> None:
    history = [_ok(0, -300, 4)]
    aux = build_auxiliary(
        history,
        expert=_EXPERT, random=_RANDOM,
        held_out_mean=None,
        n_submits=1, n_successful_submits=1, episodes_used=4,
    )
    assert aux["held_out_gap"] is None
    assert aux["n_submits"] == 1


def test_build_auxiliary_empty_run() -> None:
    aux = build_auxiliary(
        history=(),
        expert=_EXPERT, random=_RANDOM,
        held_out_mean=None,
        n_submits=0, n_successful_submits=0, episodes_used=0,
    )
    assert aux["mean_episodes_per_submit"] is None
    assert aux["auc_in_loop"] is None
    assert aux["episodes_to_50pct"] is None
    assert aux["held_out_gap"] is None
