"""Reward calculation for policy-improvement epochs."""

from __future__ import annotations

from typing import Any


def score_delta(reference: dict[str, Any], evaluation: dict[str, Any]) -> float:
    return float(evaluation.get("mean_score", 0.0)) - float(reference.get("mean_score", 0.0))


def success_delta(reference: dict[str, Any], evaluation: dict[str, Any]) -> float:
    return float(evaluation.get("success_rate", 0.0)) - float(reference.get("success_rate", 0.0))


def compute_reward(
    *,
    reference: dict[str, Any],
    evaluation: dict[str, Any],
    compile_ok: bool,
    agent_ok: bool,
    evaluation_ok: bool = True,
    minimum_score: float = 0.0,
    protected_changed: bool = False,
) -> dict[str, Any]:
    invalid = (not compile_ok) or (not agent_ok) or (not evaluation_ok) or protected_changed
    validation_delta = score_delta(reference["validation"], evaluation["validation"])
    heldout_delta = score_delta(reference["heldout"], evaluation["heldout"])
    reward = minimum_score if invalid else validation_delta
    return {
        "reward": reward,
        "invalid": invalid,
        "minimum_score": minimum_score,
        "minimum_score_applied": invalid,
        "validation_score_delta": validation_delta,
        "heldout_score_delta": heldout_delta,
        "validation_success_delta": success_delta(reference["validation"], evaluation["validation"]),
        "heldout_success_delta": success_delta(reference["heldout"], evaluation["heldout"]),
    }
