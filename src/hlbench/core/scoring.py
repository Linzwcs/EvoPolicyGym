"""Score + auxiliary-metric computations for ``run.json:outcome``.

All functions are pure: state in, numbers out. Server pulls them
together in ``finalize()``. Keep them here so unit tests don't need
to spin up a Server.

SPEC §5.2 (final score) and §5.3 (auxiliary metrics) are the
references; some interpretation calls (notably the negative-reward
case for ``episodes_to_Npct``) are documented inline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hlbench.core.submit_handler import SubmitHistoryEntry


def normalized_score(
    mean_return: float, *, expert: float, random: float,
) -> float:
    """Map raw mean_return → [0, 1+] band where 0=random, 1=expert.

    Negative when policy is worse than random. Clipping is the caller's
    job; this returns the unclipped value so curve metrics that need
    raw progression don't lose information."""
    denom = expert - random
    if denom == 0:
        return 0.0
    return (mean_return - random) / denom


def final_score(mean_return: float, *, expert: float, random: float) -> float:
    """SPEC §5.2: ``clip(normalized, 0.0, 1.2) * 100``.

    Negative scores clamp to 0; super-expert clamps at 120."""
    n = normalized_score(mean_return, expert=expert, random=random)
    return round(max(0.0, min(1.2, n)) * 100, 3)


def auc_in_loop(
    history: Sequence[SubmitHistoryEntry], *, expert: float, random: float,
) -> float | None:
    """Trapezoidal AUC of the (cumulative_episodes, normalized_score) curve,
    averaged over the run and scaled to [0, 100].

    Concretely: the (0, 0) baseline anchors the start; each successful
    submit contributes a trapezoid. Skipped (failed) submits don't
    advance the curve but still consume their share of the x-axis.

    Returns ``None`` if no successful submit recorded a mean_return —
    nothing to integrate.
    """
    ok_points = [
        (e.cumulative_episodes, normalized_score(
            e.mean_return, expert=expert, random=random  # type: ignore[arg-type]
        ))
        for e in history
        if e.status == "ok" and e.mean_return is not None
    ]
    if not ok_points:
        return None

    total_episodes = ok_points[-1][0]
    if total_episodes == 0:
        return None

    # Trapezoidal rule with (0, 0) start.
    prev_x, prev_y = 0, 0.0
    area = 0.0
    for x, y in ok_points:
        # Clamp normalized score to [0, 1.2] so a single disastrous
        # submit doesn't drag the AUC into negative territory.
        y_clipped = max(0.0, min(1.2, y))
        area += (x - prev_x) * (prev_y + y_clipped) / 2.0
        prev_x, prev_y = x, y_clipped

    return round((area / total_episodes) * 100, 3)


def episodes_to_threshold(
    history: Sequence[SubmitHistoryEntry],
    threshold_fraction: float,
    *,
    expert: float,
    random: float,
) -> int | None:
    """Cumulative episodes when the in-loop normalized score first crosses
    ``threshold_fraction`` (e.g. 0.5 for "50% of the way from random to
    expert", 0.8 for 80%).

    Returns ``None`` if the threshold is never reached. The SPEC §5.3
    wording is "exceeds 0.5 × expert", which is a clean rule for
    positive-reward envs but breaks for negative-reward envs like
    Pendulum (0.5 × -150 = -75 is *better* than expert -150). We
    interpret it as the normalized fraction, which Day 14 will
    propagate back into the spec.
    """
    for entry in history:
        if entry.status != "ok" or entry.mean_return is None:
            continue
        n = normalized_score(entry.mean_return, expert=expert, random=random)
        if n >= threshold_fraction:
            return entry.cumulative_episodes
    return None


def build_auxiliary(
    history: Sequence[SubmitHistoryEntry],
    *,
    expert: float,
    random: float,
    held_out_mean: float | None,
    n_submits: int,
    n_successful_submits: int,
    episodes_used: int,
) -> dict[str, Any]:
    """Bundle all auxiliary metrics for ``run.json:outcome.auxiliary``.

    ``held_out_gap`` uses the most recent successful submit as the
    "final in-loop" mean; if none exists, it's ``None``."""
    last_ok_mean: float | None = None
    for e in reversed(history):
        if e.status == "ok" and e.mean_return is not None:
            last_ok_mean = e.mean_return
            break

    held_out_gap: float | None
    if held_out_mean is None or last_ok_mean is None:
        held_out_gap = None
    else:
        held_out_gap = round(last_ok_mean - held_out_mean, 3)

    return {
        "auc_in_loop": auc_in_loop(history, expert=expert, random=random),
        "episodes_to_50pct": episodes_to_threshold(
            history, 0.5, expert=expert, random=random,
        ),
        "episodes_to_80pct": episodes_to_threshold(
            history, 0.8, expert=expert, random=random,
        ),
        "mean_submit_wall_time": None,  # not tracked in SubmitHistoryEntry; post-MVP
        "held_out_gap": held_out_gap,
        "n_submits": n_submits,
        "n_successful_submits": n_successful_submits,
        "episodes_used": episodes_used,
        "mean_episodes_per_submit": (
            round(episodes_used / n_submits, 3) if n_submits else None
        ),
    }
