"""Tests for deck building and back (deck type) effects.

Verifies standard 52-card deck, Abandoned/Checkered/Erratic deck mutations,
Back.apply_to_run, and Back.trigger_effect.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.back import Back
from jackdaw.engine.card import reset_sort_id_counter
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.deck_builder import build_deck
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ============================================================================
# Standard deck
# ============================================================================


class TestStandardDeck:
    def test_52_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        assert len(cards) == 52


# ============================================================================
# Abandoned Deck
# ============================================================================


class TestAbandonedDeck:
    def test_40_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_abandoned", rng)
        assert len(cards) == 40


# ============================================================================
# Checkered Deck
# ============================================================================


class TestCheckeredDeck:
    def test_only_two_suits(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_checkered", rng)
        suits = {c.base.suit for c in cards}
        assert suits == {Suit.SPADES, Suit.HEARTS}


# ============================================================================
# Erratic Deck
# ============================================================================


# ============================================================================
# Back construction
# ============================================================================


def back(key: str) -> Back:
    return Back(key)


class TestApplyToRun:
    def test_red_deck_discards_delta(self):
        m = back("b_red").apply_to_run({})
        assert m["discards_delta"] == 1
        assert "hands_delta" not in m

    def test_plasma_deck_ante_scaling(self):
        m = back("b_plasma").apply_to_run({})
        assert m["ante_scaling"] == 2


# ============================================================================
# trigger_effect — Plasma Deck
# ============================================================================


class TestTriggerEffectPlasma:
    def test_plasma_100_chips_20_mult_gives_60_each(self):
        result = back("b_plasma").trigger_effect("final_scoring_step", chips=100.0, mult=20.0)
        assert result == {"chips": 60, "mult": 60}


# ============================================================================
# trigger_effect — Anaglyph Deck
# ============================================================================


class TestTriggerEffectAnaglyph:
    def test_anaglyph_boss_defeated_creates_double_tag(self):
        result = back("b_anaglyph").trigger_effect("eval", boss_defeated=True)
        assert result == {"create_tag": "tag_double"}


# ============================================================================
# Scoring integration — Phase 10 uses Back.trigger_effect
# ============================================================================


class TestScoringPhase10Integration:
    def _make_minimal_scoring_fixtures(self):
        from jackdaw.engine.blind import Blind
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.hand_levels import HandLevels

        card = create_playing_card(Suit.SPADES, Rank.ACE)
        hand_levels = HandLevels()
        rng = PseudoRandom("PHASE10_TEST")
        blind = Blind.create("bl_small", ante=1)
        return card, hand_levels, rng, blind

    def test_plasma_deck_scoring_averages_chips_mult(self):
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.scoring import score_hand

        card, hand_levels, rng, blind = self._make_minimal_scoring_fixtures()
        aces = [create_playing_card(Suit.SPADES, Rank.ACE) for _ in range(5)]

        result = score_hand(
            played_cards=aces,
            held_cards=[],
            jokers=[],
            hand_levels=hand_levels,
            blind=blind,
            rng=rng,
            back_key="b_plasma",
        )
        assert result.chips == result.mult
