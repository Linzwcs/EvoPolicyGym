"""Verify the jackdaw.engine public API contract.

All tests import exclusively from ``jackdaw.engine`` — no reaching into
submodules.  This guards the public surface against accidental breakage.
"""

from __future__ import annotations

import jackdaw.engine
from jackdaw.engine import (
    CashOut,
    GamePhase,
    PlayHand,
    SelectBlind,
    get_legal_actions,
    initialize_run,
    random_agent,
    simulate_run,
    step,
)

# ------------------------------------------------------------------
# Test 1: simulate_run completes with random_agent
# ------------------------------------------------------------------


def test_simulate_run_completes():
    result = simulate_run("b_red", 1, "API_TEST", random_agent)
    assert result["phase"] == GamePhase.GAME_OVER or result.get("won")


# ------------------------------------------------------------------
# Test 2: manual step-through via public interface
# ------------------------------------------------------------------


def test_manual_step_through():
    gs = initialize_run("b_red", 1, "API_TEST")
    assert gs["phase"] == GamePhase.BLIND_SELECT

    actions = get_legal_actions(gs)
    assert any(isinstance(a, SelectBlind) for a in actions)

    gs = step(gs, SelectBlind())
    assert gs["phase"] == GamePhase.SELECTING_HAND


# ------------------------------------------------------------------
# Test 3: ``from jackdaw.engine import *`` exports exactly __all__
# ------------------------------------------------------------------


def test_all_matches_star_import():
    exported = set(jackdaw.engine.__all__)

    # Every name in __all__ must be importable from the package
    for name in exported:
        assert hasattr(jackdaw.engine, name), f"{name!r} in __all__ but not importable"

    # No public names leak beyond __all__ (exclude submodules that Python
    # auto-adds when the package imports from them)
    public_attrs = {n for n in dir(jackdaw.engine) if not n.startswith("_")}
    import types

    submodules = {
        n for n in public_attrs if isinstance(getattr(jackdaw.engine, n), types.ModuleType)
    }
    extras = public_attrs - exported - submodules
    assert not extras, f"Public names not in __all__: {extras}"


# ------------------------------------------------------------------
# Test 4: required keys exist after initialize_run
# ------------------------------------------------------------------

# Keys that init_game_object + initialize_run guarantee.
# If any of these are missing, gs["key"] access would KeyError —
# catching that here instead of mid-game.
_REQUIRED_TOP_LEVEL = {
    # Phase & control
    "phase",
    "won",
    "round",
    "stake",
    "win_ante",
    "blind_on_deck",
    "blind",
    "skips",
    "selected_back_key",
    "seeded",
    # Economy
    "dollars",
    "chips",
    "interest_cap",
    "interest_amount",
    "discount_percent",
    "base_reroll_cost",
    "inflation",
    "rental_rate",
    "bankrupt_at",
    # Card areas
    "hand",
    "deck",
    "jokers",
    "consumables",
    "discard_pile",
    # Card area limits
    "hand_size",
    "joker_slots",
    "consumable_slots",
    # Sub-dicts
    "round_resets",
    "current_round",
    "starting_params",
    "round_bonus",
    "shop",
    "modifiers",
    "round_scores",
    "probabilities",
    "previous_round",
    # Pool / rates
    "edition_rate",
    "joker_rate",
    "tarot_rate",
    "planet_rate",
    "spectral_rate",
    "playing_card_rate",
    # Tracking
    "hands_played",
    "unused_discards",
    "bosses_used",
    "used_jokers",
    "used_vouchers",
    "banned_keys",
    # RNG & hand levels
    "rng",
    "hand_levels",
}

_REQUIRED_ROUND_RESETS = {
    "hands",
    "discards",
    "reroll_cost",
    "ante",
    "blind_ante",
    "blind_states",
    "blind_choices",
    "boss_rerolled",
}

_REQUIRED_CURRENT_ROUND = {
    "hands_left",
    "hands_played",
    "discards_left",
    "discards_used",
    "reroll_cost",
    "reroll_cost_increase",
    "free_rerolls",
    "jokers_purchased",
    "dollars",
    "idol_card",
    "mail_card",
    "ancient_card",
    "castle_card",
}


def test_required_keys_after_initialize_run():
    gs = initialize_run("b_red", 1, "KEY_TEST")

    missing = _REQUIRED_TOP_LEVEL - gs.keys()
    assert not missing, f"Missing top-level keys after initialize_run: {missing}"

    missing_rr = _REQUIRED_ROUND_RESETS - gs["round_resets"].keys()
    assert not missing_rr, f"Missing round_resets keys: {missing_rr}"

    missing_cr = _REQUIRED_CURRENT_ROUND - gs["current_round"].keys()
    assert not missing_cr, f"Missing current_round keys: {missing_cr}"


# ------------------------------------------------------------------
# Test 5: required keys survive select_blind → play_hand → cash_out
# ------------------------------------------------------------------


def test_required_keys_survive_step_sequence():
    gs = initialize_run("b_red", 1, "KEY_TEST")

    # select blind
    gs = step(gs, SelectBlind())
    assert gs["phase"] == GamePhase.SELECTING_HAND

    missing = _REQUIRED_TOP_LEVEL - gs.keys()
    assert not missing, f"Missing after SelectBlind: {missing}"

    # play first 5 cards
    hand = gs["hand"]
    indices = tuple(range(min(5, len(hand))))
    gs = step(gs, PlayHand(card_indices=indices))

    missing = _REQUIRED_TOP_LEVEL - gs.keys()
    assert not missing, f"Missing after PlayHand: {missing}"

    # if we won the round, cash out
    if gs["phase"] == GamePhase.ROUND_EVAL:
        gs = step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP

        missing = _REQUIRED_TOP_LEVEL - gs.keys()
        assert not missing, f"Missing after CashOut: {missing}"
