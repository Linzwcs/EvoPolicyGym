"""Read explicit Balatro outcomes without inferring wins from reward."""

from __future__ import annotations

from typing import Any


def episode_outcome(
    episode: dict[str, Any],
    document: dict[str, Any],
    position: int,
) -> tuple[float, float, bool, bool]:
    """Return scored progress, reward, win, and failure for one batch position."""
    if episode.get("failure") is not None:
        return 0.0, 0.0, False, True
    reward = episode.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("completed Episode reward must be numeric")

    outcome = episode
    if "won" not in episode or "rounds_cleared" not in episode:
        content = document.get("content")
        diagnostics = content.get("episode_diagnostics") if isinstance(content, dict) else None
        if not isinstance(diagnostics, list) or position >= len(diagnostics):
            raise ValueError(
                "explicit won and rounds_cleared diagnostics are required; "
                "reward alone cannot identify a win"
            )
        outcome = diagnostics[position]
        if not isinstance(outcome, dict) or outcome.get("episode_index") != position:
            raise ValueError("Episode diagnostics must match the batch position")

    won = outcome.get("won")
    rounds = outcome.get("rounds_cleared")
    if type(won) is not bool or type(rounds) is not int or rounds < 0:
        raise ValueError("Episode requires boolean won and non-negative integer rounds_cleared")
    expected_reward = rounds * (2 if won else 1)
    if reward != expected_reward:
        raise ValueError("Episode reward disagrees with run-score-v3 diagnostics")
    return float(rounds), float(reward), won, False
