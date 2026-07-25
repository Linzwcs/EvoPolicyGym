"""High-level game runner — simulates complete Balatro runs.

Provides :func:`simulate_run` which loops the game step function with
a player agent, and :func:`random_agent` for testing / baseline.

Usage::

    from jackdaw.engine.runner import simulate_run, random_agent

    result = simulate_run("b_red", 1, "SEED42", random_agent)
    print(result["won"], result["round"])
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable
from typing import Any

from jackdaw.engine.actions import (
    Action,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    get_legal_actions,
)
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run


def simulate_run(
    back_key: str,
    stake: int,
    seed: str,
    agent: Callable[[dict[str, Any], list[Action]], Action],
    *,
    challenge: dict[str, Any] | None = None,
    max_actions: int = 10_000,
) -> dict[str, Any]:
    """Simulate a complete Balatro run from initialization to game over / win.

    Parameters
    ----------
    back_key:
        Deck back key (e.g. ``"b_red"``).
    stake:
        Stake level (1-8).
    seed:
        RNG seed string.
    agent:
        Callable ``(game_state, legal_actions) -> action``.  Called every
        decision point.  The agent receives the full game_state dict and
        a list of legal actions, and must return one of them (or a
        compatible action with filled-in indices).
    challenge:
        Optional challenge definition dict.
    max_actions:
        Safety limit to prevent infinite loops.

    Returns
    -------
    dict
        The final game_state.  Key fields:

        * ``won`` (bool) — True if the run was won
        * ``round`` (int) — rounds completed
        * ``phase`` (GamePhase) — terminal phase
        * ``dollars`` (int) — final bank balance
        * ``actions_taken`` (int) — total actions executed
    """
    gs = initialize_run(back_key, stake, seed, challenge=challenge)

    actions_taken = 0
    while actions_taken < max_actions:
        phase = gs.get("phase")
        if phase == GamePhase.GAME_OVER:
            break
        if gs.get("won") and phase == GamePhase.ROUND_EVAL:
            # Won the game — still need to cash out
            pass
        if gs.get("won") and phase == GamePhase.SHOP:
            # Game won, in shop — stop
            break

        legal = get_legal_actions(gs)
        if not legal:
            break

        action = agent(gs, legal)
        try:
            step(gs, action)
        except IllegalActionError:
            # Agent returned an invalid action — skip it
            break
        actions_taken += 1

    gs["actions_taken"] = actions_taken
    return gs


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------


def random_agent(
    game_state: dict[str, Any],
    legal_actions: list[Action],
) -> Action:
    """Pick a random legal action.

    For marker actions (PlayHand/Discard with empty indices), fills in
    random card subsets from the hand.  Prefers play/discard over
    utility actions (sort, reorder) to make progress.
    """
    hand: list = game_state.get("hand", [])

    # Separate progress-making actions from utility
    progress: list[Action] = []
    utility: list[Action] = []
    for a in legal_actions:
        if isinstance(a, (SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)):
            utility.append(a)
        else:
            progress.append(a)

    pool = progress if progress else utility
    if not pool:
        return legal_actions[0]

    action = _random.choice(pool)

    # Resolve marker actions with random card indices
    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = _random.randint(1, n)
        indices = tuple(sorted(_random.sample(range(len(hand)), count)))
        return PlayHand(card_indices=indices)

    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = _random.randint(1, n)
        indices = tuple(sorted(_random.sample(range(len(hand)), count)))
        return Discard(card_indices=indices)

    return action


# ---------------------------------------------------------------------------
# Scripted agent — always plays first 5 cards, never discards
# ---------------------------------------------------------------------------


def greedy_play_agent(
    game_state: dict[str, Any],
    legal_actions: list[Action],
) -> Action:
    """Deterministic agent: always plays first 5 cards, selects blinds,
    cashes out, and advances.  Never discards, never buys, never rerolls.
    """
    hand: list = game_state.get("hand", [])

    for a in legal_actions:
        if isinstance(a, SelectBlind):
            return a
        if isinstance(a, CashOut):
            return a
        if isinstance(a, NextRound):
            return a
        if isinstance(a, SkipPack):
            return a

    # Play hand if possible
    for a in legal_actions:
        if isinstance(a, PlayHand) and hand:
            n = min(5, len(hand))
            return PlayHand(card_indices=tuple(range(n)))

    # Fallback
    return legal_actions[0]
