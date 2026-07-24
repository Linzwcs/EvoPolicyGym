"""Hand evaluation tests: detection, modifiers, and pipeline integration."""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.hand_eval import (
    HandEvalResult,
    evaluate_hand,
    evaluate_poker_hand,
    get_flush,
    get_straight,
)
from jackdaw.engine.hand_levels import HandLevels


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    """Helper: create a playing card."""
    suit_letter = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rank_letter = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "T",
        "Jack": "J",
        "Queen": "Q",
        "King": "K",
        "Ace": "A",
    }
    c = Card()
    key = f"{suit_letter[suit]}_{rank_letter[rank]}"
    c.set_base(key, suit, rank)
    c.set_ability(enhancement)
    return c


# ============================================================================
# Hand Type Detection (12 tests via evaluate_poker_hand)
# ============================================================================


class TestHandTypeDetection:
    """One test per hand type through evaluate_poker_hand."""

    def test_two_pair(self):
        hand = [
            _card("Hearts", "5"),
            _card("Spades", "5"),
            _card("Clubs", "Jack"),
            _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Two Pair"]
        assert results["Pair"]  # downward: Two Pair also has Pair

    def test_straight(self):
        hand = [
            _card("Hearts", "4"),
            _card("Spades", "5"),
            _card("Clubs", "6"),
            _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Straight"]

    def test_flush(self):
        hand = [_card("Hearts", r) for r in ["2", "5", "8", "Jack", "Ace"]]
        results = evaluate_poker_hand(hand)
        assert results["Flush"]

    def test_full_house(self):
        hand = [
            _card("Hearts", "King"),
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "5"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Full House"]
        assert results["Three of a Kind"]  # downward
        assert results["Pair"]  # downward

    def test_straight_flush(self):
        hand = [
            _card("Hearts", "4"),
            _card("Hearts", "5"),
            _card("Hearts", "6"),
            _card("Hearts", "7"),
            _card("Hearts", "8"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Straight Flush"]
        assert results["Flush"]
        assert results["Straight"]

    def test_five_of_a_kind(self):
        hand = [
            _card("Hearts", "Ace"),
            _card("Spades", "Ace"),
            _card("Clubs", "Ace"),
            _card("Diamonds", "Ace"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Five of a Kind"]
        assert results["Four of a Kind"]  # downward
        assert results["Three of a Kind"]  # downward
        assert results["Pair"]  # downward


# ============================================================================
# Downward Propagation (1 test)
# ============================================================================


class TestDownwardPropagation:
    """Downward propagation: higher hands populate lower entries."""

    def test_five_of_a_kind_populates_four_three_pair(self):
        hand = [
            _card("Hearts", "Ace"),
            _card("Spades", "Ace"),
            _card("Clubs", "Ace"),
            _card("Diamonds", "Ace"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Five of a Kind"]
        assert results["Four of a Kind"]
        assert results["Three of a Kind"]
        assert results["Pair"]


# ============================================================================
# Modifier Effects (Four Fingers, Shortcut, Smeared, Wild Card)
# ============================================================================


class TestModifierEffects:
    """Joker-based modifiers applied at the detection level."""

    def test_four_fingers_four_is_enough(self):
        hand = [
            _card("Clubs", "3"),
            _card("Clubs", "7"),
            _card("Clubs", "10"),
            _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        result = get_flush(hand, four_fingers=True)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_shortcut_one_gap(self):
        """Shortcut: one rank gap is allowed."""
        hand = [
            _card("Hearts", "3"),
            _card("Spades", "4"),
            # gap at 5
            _card("Clubs", "6"),
            _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        result = get_straight(hand, shortcut=True)
        assert len(result) == 1

    def test_smeared_joker_red_suits(self):
        """Smeared: Hearts and Diamonds count as same suit."""
        hand = [
            _card("Hearts", "2"),
            _card("Hearts", "5"),
            _card("Diamonds", "8"),
            _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        # Without smeared: no flush (3H + 2D)
        assert get_flush(hand) == []
        # With smeared: all red = flush
        result = get_flush(hand, smeared=True)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_wild_card_matches_all_suits(self):
        """Wild Card counts for every suit in flush detection."""
        hand = [
            _card("Spades", "2"),
            _card("Spades", "5"),
            _card("Spades", "8"),
            _card("Spades", "Jack"),
            _card("Hearts", "Ace", enhancement="m_wild"),  # Wild Card
        ]
        result = get_flush(hand)
        assert len(result) == 1
        assert len(result[0]) == 5  # 4 Spades + Wild = flush


# ============================================================================
# Straight Edge Cases (Ace-low, no-wrap)
# ============================================================================


class TestStraightEdgeCases:
    def test_ace_low_straight(self):
        """A-2-3-4-5 is valid (Ace wraps low)."""
        hand = [
            _card("Hearts", "Ace"),
            _card("Spades", "2"),
            _card("Clubs", "3"),
            _card("Diamonds", "4"),
            _card("Hearts", "5"),
        ]
        result = get_straight(hand)
        assert len(result) == 1

    def test_no_wrap_around(self):
        """Q-K-A-2-3 is NOT valid (no wrapping)."""
        hand = [
            _card("Hearts", "Queen"),
            _card("Spades", "King"),
            _card("Clubs", "Ace"),
            _card("Diamonds", "2"),
            _card("Hearts", "3"),
        ]
        result = get_straight(hand)
        assert result == []


# ============================================================================
# Full Pipeline (Splash, Stone, debuffed joker, evaluate_hand integration)
# ============================================================================


class TestFullPipeline:
    """Pipeline integration via evaluate_hand."""

    def test_splash_all_cards_score(self):
        """Splash: all 5 played cards become scoring cards, even High Card."""
        hand = [
            _card("Hearts", "2"),
            _card("Spades", "5"),
            _card("Clubs", "8"),
            _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        jokers = [create_joker("j_splash")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "High Card"
        assert len(result.scoring_cards) == 5
        # All 5 cards should be in scoring
        assert set(id(c) for c in result.scoring_cards) == set(id(c) for c in hand)

    def test_stone_card_added_to_scoring(self):
        """Stone Card played as 5th card with Full House: included in scoring."""
        hand = [
            _card("Hearts", "King"),
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "3", enhancement="m_stone"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Three of a Kind"
        # Stone card should be augmented into scoring
        stone = hand[4]
        assert stone in result.scoring_cards

    def test_debuffed_four_fingers_no_effect(self):
        """Debuffed Four Fingers: does NOT enable 4-card flush."""
        hand = [
            _card("Clubs", "3"),
            _card("Clubs", "7"),
            _card("Clubs", "10"),
            _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        ff = create_joker("j_four_fingers")
        ff.set_debuff(True)
        result = evaluate_hand(hand, jokers=[ff])
        assert result.detected_hand != "Flush"

    def test_evaluate_hand_returns_result(self):
        """evaluate_hand returns HandEvalResult with all expected fields."""
        hand = [_card("Hearts", r) for r in ["2", "5", "8", "Jack", "Ace"]]
        result = evaluate_hand(hand)
        assert isinstance(result, HandEvalResult)
        assert result.detected_hand == "Flush"
        assert len(result.scoring_cards) == 5
        assert result.all_played == hand


# ============================================================================
# HandLevels (merged from test_hand_levels.py)
# ============================================================================


class TestLevelUp:
    def test_pair_level_5(self):
        """Pair at level 5: chips=10+15*4=70, mult=2+1*4=6."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR, amount=4)  # 1->5
        assert levels.get(HandType.PAIR) == (70, 6)

    def test_pair_level_2(self):
        """Pair at level 2: chips=10+15=25, mult=2+1=3."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR)
        assert levels.get(HandType.PAIR) == (25, 3)
        assert levels[HandType.PAIR].level == 2

    def test_level_down(self):
        """The Arm boss blind: level down by -1."""
        levels = HandLevels()
        levels.level_up(HandType.PAIR, amount=3)  # level 4
        levels.level_up(HandType.PAIR, amount=-1)  # level 3
        assert levels[HandType.PAIR].level == 3
        # chips=10+15*2=40, mult=2+1*2=4
        assert levels.get(HandType.PAIR) == (40, 4)

    def test_level_up_makes_secret_visible(self):
        """Leveling a secret hand makes it visible."""
        levels = HandLevels()
        assert levels[HandType.FLUSH_FIVE].visible is False
        levels.level_up(HandType.FLUSH_FIVE)
        assert levels[HandType.FLUSH_FIVE].visible is True


class TestBlackHole:
    def test_secret_hands_become_visible(self):
        levels = HandLevels()
        levels.level_up_all()
        assert levels[HandType.FLUSH_FIVE].visible is True
        assert levels[HandType.FLUSH_HOUSE].visible is True
        assert levels[HandType.FIVE_OF_A_KIND].visible is True


class TestPlayRecording:
    def test_reset_round_counts(self):
        levels = HandLevels()
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.FLUSH)
        levels.reset_round_counts()
        assert levels[HandType.PAIR].played_this_round == 0
        assert levels[HandType.FLUSH].played_this_round == 0
        # Total played is NOT reset
        assert levels[HandType.PAIR].played == 2
        assert levels[HandType.FLUSH].played == 1


class TestMostPlayed:
    def test_after_plays(self):
        levels = HandLevels()
        levels.record_play(HandType.FLUSH)
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        levels.record_play(HandType.PAIR)
        assert levels.most_played() is HandType.PAIR
