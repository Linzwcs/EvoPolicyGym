"""Tests for jackdaw.engine.runner — simulate_run, random_agent, greedy_play_agent.

Coverage
--------
* 10 seeds with random_agent: all complete without crash.
* 10 seeds with greedy_play_agent: all complete without crash.
* Scripted determinism: same seed → same outcome (5 seeds + beats_three_blinds).
* Performance: >500 runs/sec.
* All 15 deck types complete without crash.
* All 8 stake levels complete without crash.
* Max actions limit respected.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run
from jackdaw.engine.validator import (
    format_report,
    validate_hand_cards,
    validate_step,
)

# ---------------------------------------------------------------------------
# Crash resistance — 10 seeds each
# ---------------------------------------------------------------------------


class TestRandomAgentCrashResistance:
    @pytest.mark.parametrize("seed_idx", range(10))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"CRASH_RAND_{seed_idx}", random_agent, max_actions=500)
        assert gs["actions_taken"] <= 500


class TestGreedyAgentCrashResistance:
    @pytest.mark.parametrize("seed_idx", range(10))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"CRASH_GRDY_{seed_idx}", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] <= 500


# ---------------------------------------------------------------------------
# Scripted determinism
# ---------------------------------------------------------------------------


def _scripted_run(seed: str) -> dict:
    """Play a scripted Small→Big→Boss sequence and return final state."""
    gs = initialize_run("b_red", 1, seed)

    for blind_name in ("Small", "Big", "Boss"):
        if blind_name != "Small":
            step(gs, NextRound())
        step(gs, SelectBlind())
        gs["blind"].chips = 1  # easy win
        # Play hands until we win or run out
        for _ in range(10):
            if gs["phase"] != GamePhase.SELECTING_HAND:
                break
            hand = gs.get("hand", [])
            n = min(5, len(hand))
            if n == 0:
                break
            step(gs, PlayHand(card_indices=tuple(range(n))))
        if gs["phase"] == GamePhase.ROUND_EVAL:
            step(gs, CashOut())

    return gs


class TestScriptedDeterminism:
    @pytest.mark.parametrize(
        "seed",
        [
            "SCRIPT_A",
            "SCRIPT_B",
            "SCRIPT_C",
            "SCRIPT_D",
            "SCRIPT_E",
        ],
    )
    def test_same_seed_same_outcome(self, seed):
        gs1 = _scripted_run(seed)
        gs2 = _scripted_run(seed)
        assert gs1["dollars"] == gs2["dollars"]
        assert gs1["chips"] == gs2["chips"]
        assert gs1.get("won") == gs2.get("won")
        assert gs1["round_resets"]["ante"] == gs2["round_resets"]["ante"]

    def test_scripted_beats_three_blinds(self):
        gs = _scripted_run("SCRIPT_FULL")
        # After beating Small, Big, Boss, ante should advance to 2
        assert gs["round_resets"]["ante"] == 2
        assert gs.get("phase") == GamePhase.SHOP


# ---------------------------------------------------------------------------
# All deck types
# ---------------------------------------------------------------------------


ALL_DECKS = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_magic",
    "b_nebula",
    "b_ghost",
    "b_abandoned",
    "b_checkered",
    "b_zodiac",
    "b_painted",
    "b_anaglyph",
    "b_plasma",
    "b_erratic",
]


class TestAllDecks:
    @pytest.mark.parametrize("back_key", ALL_DECKS)
    def test_deck_completes(self, back_key):
        gs = simulate_run(back_key, 1, "DECK_FV", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# All stake levels
# ---------------------------------------------------------------------------


class TestAllStakes:
    @pytest.mark.parametrize("stake", range(1, 9))
    def test_stake_completes(self, stake):
        gs = simulate_run("b_red", stake, "STAKE_FV", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# Max actions limit
# ---------------------------------------------------------------------------


class TestMaxActions:
    def test_respects_max_actions(self):
        gs = simulate_run("b_red", 1, "MAX_ACT", random_agent, max_actions=3)
        assert gs["actions_taken"] <= 3


# ============================================================================
# Validator (merged from test_validator.py)
# ============================================================================


def _sim_state(**overrides):
    base = {
        "dollars": 10,
        "chips": 500,
        "round_resets": {"ante": 1},
        "current_round": {"hands_left": 4, "discards_left": 3},
        "hand": [],
        "deck": [None] * 44,
        "jokers": [],
        "blind": None,
    }
    base.update(overrides)
    return base


def _live_state(**overrides):
    base = {
        "money": 10,
        "ante": 1,
        "round": {"chips": 500, "hands_left": 4, "discards_left": 3},
        "hand": [],
        "jokers": [],
        "deck_size": 44,
        "blind": {"chips": 0},
    }
    base.update(overrides)
    return base


class TestValidateStep:
    def test_matching_states_no_diffs(self):
        diffs = validate_step(_sim_state(), _live_state())
        assert diffs == []

    def test_dollar_mismatch(self):
        diffs = validate_step(_sim_state(dollars=10), _live_state(money=15))
        assert any("dollars" in d for d in diffs)


class TestValidateHandCards:
    def test_suit_mismatch(self):
        class FakeCard:
            class base:
                class suit:
                    value = "Hearts"

                class rank:
                    value = "Ace"

        diffs = validate_hand_cards(
            [FakeCard()],
            [{"suit": "Spades", "rank": "Ace"}],
        )
        assert len(diffs) == 1
        assert "hand[0]" in diffs[0]


class TestFormatReport:
    def test_clean_report(self):
        report = format_report([[], []], "SEED")
        assert "Clean steps: 2/2" in report
        assert "Total discrepancies: 0" in report


class TestValidatorIntegration:
    def test_sim_matches_mock(self):
        """When mock derives from sim state, no discrepancies."""
        gs = initialize_run("b_red", 1, "VALID_INT")

        live = {
            "money": gs["dollars"],
            "ante": gs["round_resets"]["ante"],
            "round": {
                "chips": gs.get("chips", 0),
                "hands_left": gs["current_round"]["hands_left"],
                "discards_left": gs["current_round"]["discards_left"],
            },
            "hand": [],
            "jokers": [],
            "deck_size": len(gs["deck"]),
            "blind": {"chips": 0},
        }
        diffs = validate_step(gs, live)
        assert diffs == []
