"""State validation — compare simulator state against a live Balatro game.

Provides :func:`validate_step` for point-in-time comparison and
:func:`validate_run_log` for post-hoc analysis of a recorded run.

This module is used by ``scripts/validate.py`` to bridge the
simulator with balatrobot (https://coder.github.io/balatrobot/).
"""

from __future__ import annotations

from typing import Any


def validate_step(
    sim_state: dict[str, Any],
    live_state: dict[str, Any],
) -> list[str]:
    """Compare simulator state to live game state.

    Parameters
    ----------
    sim_state:
        The simulator's ``game_state`` dict from :func:`step`.
    live_state:
        State dict read from balatrobot's API.  Expected shape::

            {
                "money": int,
                "ante": int,
                "round": {"chips": int, "hands_left": int, ...},
                "hand": [{"suit": str, "rank": str}, ...],
                "jokers": [{"key": str, "ability": dict}, ...],
                "deck_size": int,
                "blind": {"key": str, "chips": int},
                "phase": str,
            }

    Returns
    -------
    list[str]
        Human-readable discrepancy descriptions.  Empty if states match.
    """
    diffs: list[str] = []

    # --- Economy ---
    sim_dollars = sim_state.get("dollars", 0)
    live_dollars = live_state.get("money", 0)
    if sim_dollars != live_dollars:
        diffs.append(f"dollars: sim={sim_dollars} live={live_dollars}")

    # --- Ante ---
    sim_ante = sim_state.get("round_resets", {}).get("ante", 1)
    live_ante = live_state.get("ante", 1)
    if sim_ante != live_ante:
        diffs.append(f"ante: sim={sim_ante} live={live_ante}")

    # --- Chips (current round) ---
    sim_chips = sim_state.get("chips", 0)
    live_round = live_state.get("round", {})
    live_chips = live_round.get("chips", 0)
    if sim_chips != live_chips:
        diffs.append(f"chips: sim={sim_chips} live={live_chips}")

    # --- Hands left ---
    sim_hands = sim_state.get("current_round", {}).get("hands_left", 0)
    live_hands = live_round.get("hands_left", 0)
    if sim_hands != live_hands:
        diffs.append(f"hands_left: sim={sim_hands} live={live_hands}")

    # --- Discards left ---
    sim_discards = sim_state.get("current_round", {}).get("discards_left", 0)
    live_discards = live_round.get("discards_left", 0)
    if sim_discards != live_discards:
        diffs.append(f"discards_left: sim={sim_discards} live={live_discards}")

    # --- Hand cards ---
    sim_hand = sim_state.get("hand", [])
    live_hand = live_state.get("hand", [])
    if len(sim_hand) != len(live_hand):
        diffs.append(f"hand_size: sim={len(sim_hand)} live={len(live_hand)}")

    # --- Deck size ---
    sim_deck = len(sim_state.get("deck", []))
    live_deck = live_state.get("deck_size", 0)
    if sim_deck != live_deck:
        diffs.append(f"deck_size: sim={sim_deck} live={live_deck}")

    # --- Joker count ---
    sim_jokers = len(sim_state.get("jokers", []))
    live_jokers = len(live_state.get("jokers", []))
    if sim_jokers != live_jokers:
        diffs.append(f"joker_count: sim={sim_jokers} live={live_jokers}")

    # --- Blind target ---
    sim_blind = sim_state.get("blind")
    live_blind = live_state.get("blind", {})
    if sim_blind is not None:
        sim_blind_chips = getattr(sim_blind, "chips", 0)
        live_blind_chips = live_blind.get("chips", 0)
        if sim_blind_chips != live_blind_chips:
            diffs.append(f"blind_chips: sim={sim_blind_chips} live={live_blind_chips}")

    return diffs


def validate_hand_cards(
    sim_hand: list[Any],
    live_hand: list[dict[str, str]],
) -> list[str]:
    """Compare hand cards in detail (suit + rank).

    Parameters
    ----------
    sim_hand:
        List of Card objects from the simulator.
    live_hand:
        List of ``{"suit": str, "rank": str}`` from balatrobot.

    Returns
    -------
    list[str]
        Discrepancies per card position.
    """
    diffs: list[str] = []
    for i in range(max(len(sim_hand), len(live_hand))):
        if i >= len(sim_hand):
            diffs.append(f"hand[{i}]: missing in sim, live={live_hand[i]}")
            continue
        if i >= len(live_hand):
            diffs.append(f"hand[{i}]: sim has extra card")
            continue
        sc = sim_hand[i]
        lc = live_hand[i]
        if sc.base is None:
            continue
        sim_suit = sc.base.suit.value
        sim_rank = sc.base.rank.value
        live_suit = lc.get("suit", "")
        live_rank = lc.get("rank", "")
        if sim_suit != live_suit or sim_rank != live_rank:
            diffs.append(
                f"hand[{i}]: sim={sim_rank} of {sim_suit}, live={live_rank} of {live_suit}"
            )
    return diffs


def validate_jokers(
    sim_jokers: list[Any],
    live_jokers: list[dict[str, Any]],
) -> list[str]:
    """Compare joker keys and basic ability state.

    Parameters
    ----------
    sim_jokers:
        List of Card objects.
    live_jokers:
        List of ``{"key": str, "ability": dict}`` from balatrobot.
    """
    diffs: list[str] = []
    if len(sim_jokers) != len(live_jokers):
        diffs.append(f"joker_count: sim={len(sim_jokers)} live={len(live_jokers)}")
    for i in range(min(len(sim_jokers), len(live_jokers))):
        sk = sim_jokers[i].center_key
        lk = live_jokers[i].get("key", "")
        if sk != lk:
            diffs.append(f"joker[{i}].key: sim={sk} live={lk}")
    return diffs


def format_report(
    step_diffs: list[list[str]],
    seed: str,
) -> str:
    """Format a validation report from a sequence of step comparisons.

    Parameters
    ----------
    step_diffs:
        List of diff lists, one per step.
    seed:
        The run seed.

    Returns
    -------
    str
        Human-readable report.
    """
    total = sum(len(d) for d in step_diffs)
    clean = sum(1 for d in step_diffs if not d)
    lines = [
        f"Validation report for seed {seed!r}",
        f"Steps compared: {len(step_diffs)}",
        f"Clean steps: {clean}/{len(step_diffs)}",
        f"Total discrepancies: {total}",
        "",
    ]
    for i, d in enumerate(step_diffs):
        if d:
            lines.append(f"Step {i}:")
            for msg in d:
                lines.append(f"  - {msg}")
    return "\n".join(lines)
