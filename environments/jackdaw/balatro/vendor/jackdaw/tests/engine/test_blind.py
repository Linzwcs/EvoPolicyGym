"""Tests for the Blind class.

Verifies chip target calculation, blind type identification, boss state,
reward dollars, and scaling across antes and stake levels.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind, get_ante_blinds, get_new_boss
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.blind_scaling import get_blind_amount, get_blind_target
from jackdaw.engine.data.prototypes import BLINDS as ALL_BLINDS
from jackdaw.engine.rng import PseudoRandom

# ============================================================================
# Small Blind
# ============================================================================


class TestSmallBlind:
    def test_ante_1_scaling_1(self):
        b = Blind.create("bl_small", ante=1)
        assert b.chips == 300  # 300 × 1.0
        assert b.mult == 1.0
        assert b.boss is False
        assert b.name == "Small Blind"
        assert b.dollars == 3


# ============================================================================
# Big Blind
# ============================================================================


class TestBigBlind:
    def test_ante_1_scaling_1(self):
        b = Blind.create("bl_big", ante=1)
        assert b.chips == 450  # 300 × 1.5
        assert b.mult == 1.5
        assert b.dollars == 4


# ============================================================================
# Boss Blinds
# ============================================================================


class TestBossBlind:
    def test_the_hook(self):
        b = Blind.create("bl_hook", ante=1)
        assert b.chips == 600  # 300 × 2.0
        assert b.mult == 2.0
        assert b.boss is True
        assert b.name == "The Hook"
        assert b.dollars == 5

    def test_the_wall_high_mult(self):
        """The Wall has mult=4 (double the normal boss)."""
        b = Blind.create("bl_wall", ante=1)
        assert b.chips == 1200  # 300 × 4.0
        assert b.mult == 4.0

    def test_the_needle_mult_1(self):
        """The Needle has mult=1 (unusually low for a boss)."""
        b = Blind.create("bl_needle", ante=1)
        assert b.chips == 300  # 300 × 1.0
        assert b.mult == 1.0
        assert b.boss is True


# ============================================================================
# Chip target calculation across antes and scaling
# ============================================================================


class TestChipTargets:
    def test_small_blind_ante_1(self):
        b = Blind.create("bl_small", ante=1, scaling=1)
        assert b.chips == get_blind_amount(1, scaling=1)

    def test_small_blind_ante_8(self):
        b = Blind.create("bl_small", ante=8, scaling=1)
        assert b.chips == get_blind_amount(8, scaling=1)


# ============================================================================
# Plasma Deck (ante_scaling = 2.0)
# ============================================================================


class TestPlasmaScaling:
    def test_small_blind_plasma(self):
        b = Blind.create("bl_small", ante=1, ante_scaling=2.0)
        assert b.chips == 600  # 300 × 1.0 × 2.0


# ============================================================================
# No blind reward (stake modifier)
# ============================================================================


class TestNoBlindReward:
    def test_small_blind_no_reward(self):
        b = Blind.create("bl_small", ante=1, no_blind_reward=True)
        assert b.dollars == 0


# ============================================================================
# Debuff config
# ============================================================================


class TestDebuffConfig:
    def test_suit_debuff(self):
        """The Club debuffs all Clubs."""
        b = Blind.create("bl_club", ante=1)
        assert b.debuff_config.get("suit") == "Clubs"

    def test_the_plant_debuffs_face(self):
        b = Blind.create("bl_plant", ante=1)
        assert b.debuff_config.get("is_face") == "face"

    def test_the_psychic_hand_size(self):
        b = Blind.create("bl_psychic", ante=1)
        assert b.debuff_config.get("h_size_ge") == 5


# ============================================================================
# debuff_card
# ============================================================================


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
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
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


class TestDebuffCardSuitBlinds:
    """The Goad/Head/Club/Window debuff specific suits."""

    def test_the_goad_debuffs_spades(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Spades", "5")
        b.debuff_card(c)
        assert c.debuff is True


class TestDebuffCardPlant:
    """The Plant debuffs face cards (J/Q/K)."""

    def test_debuffs_king(self):
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Hearts", "King")
        b.debuff_card(c)
        assert c.debuff is True

    def test_with_pareidolia(self):
        """With Pareidolia, ALL cards are face → all debuffed by The Plant."""
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c, pareidolia=True)
        assert c.debuff is True


class TestDebuffCardPillar:
    """The Pillar debuffs cards played earlier this ante."""

    def test_debuffs_played_card(self):
        reset_sort_id_counter()
        b = Blind.create("bl_pillar", ante=1)
        c = _card("Hearts", "5")
        c.ability["played_this_ante"] = True
        b.debuff_card(c)
        assert c.debuff is True


class TestDebuffCardVerdantLeaf:
    """Verdant Leaf debuffs ALL non-joker cards unconditionally."""

    def test_debuffs_all(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_leaf", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c)
        assert c.debuff is True


class TestDebuffCardDisabled:
    """Disabled boss blinds debuff nothing."""

    def test_disabled_suit_blind(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        b.disabled = True
        c = _card("Spades", "5")
        b.debuff_card(c)
        assert c.debuff is False


class TestDebuffCardClearsDebuff:
    """debuff_card should CLEAR debuff if the card doesn't match."""

    def test_previously_debuffed_cleared(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Hearts", "5")
        c.debuff = True  # previously debuffed
        b.debuff_card(c)
        assert c.debuff is False  # Hearts not debuffed by The Goad


# ============================================================================
# debuff_hand
# ============================================================================


class TestDebuffHandEye:
    """The Eye: each hand type can only be used once."""

    def test_second_use_blocked(self):
        b = Blind.create("bl_eye", ante=1)
        b.debuff_hand([], {}, "Pair")
        assert b.debuff_hand([], {}, "Pair") is True


class TestDebuffHandMouth:
    """The Mouth: only one hand type allowed per round."""

    def test_different_type_blocked(self):
        b = Blind.create("bl_mouth", ante=1)
        b.debuff_hand([], {}, "Flush")
        assert b.debuff_hand([], {}, "Pair") is True


class TestDebuffHandPsychic:
    """The Psychic: must play at least 5 cards (h_size_ge=5)."""

    def test_4_cards_blocked(self):
        b = Blind.create("bl_psychic", ante=1)
        cards = [_card("Hearts", str(i)) for i in range(2, 6)]
        assert b.debuff_hand(cards, {}, "Pair") is True


# ============================================================================
# modify_hand
# ============================================================================


class TestModifyHand:
    """Blind.modify_hand: The Flint halves chips and mult."""

    def test_the_flint(self):
        b = Blind.create("bl_flint", ante=1)
        mult, chips, modified = b.modify_hand(20.0, 100)
        assert modified is True
        # floor(20 * 0.5 + 0.5) = 10, floor(100 * 0.5 + 0.5) = 50
        assert mult == 10.0
        assert chips == 50

    def test_the_flint_minimum_mult_1(self):
        b = Blind.create("bl_flint", ante=1)
        mult, chips, _ = b.modify_hand(1.0, 2)
        assert mult >= 1.0

    def test_the_flint_disabled(self):
        b = Blind.create("bl_flint", ante=1)
        b.disabled = True
        mult, chips, modified = b.modify_hand(20.0, 100)
        assert modified is False
        assert mult == 20.0
        assert chips == 100


# ============================================================================
# press_play
# ============================================================================


class TestPressPlayHook:
    """The Hook: discard 2 random cards from hand."""

    def test_returns_discard_indices(self):
        reset_sort_id_counter()
        b = Blind.create("bl_hook", ante=1)
        hand = [_card("Hearts", str(i)) for i in range(2, 10)]
        rng = PseudoRandom("TESTSEED")
        result = b.press_play(hand, [], rng=rng)
        assert "discard_indices" in result
        assert len(result["discard_indices"]) == 2
        assert b.triggered is True


class TestPressPlayTooth:
    """The Tooth: lose $1 per card played."""

    def test_money_cost(self):
        reset_sort_id_counter()
        b = Blind.create("bl_tooth", ante=1)
        played = [_card("Hearts", "5"), _card("Spades", "King")]
        result = b.press_play([], played)
        assert result["money_cost"] == 2
        assert b.triggered is True


# ============================================================================
# drawn_to_hand
# ============================================================================


class TestDrawnToHandBell:
    """Cerulean Bell: force-select a random card."""

    def test_returns_forced_index(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_bell", ante=1)
        hand = [_card("Hearts", "5"), _card("Spades", "King"), _card("Clubs", "Ace")]
        rng = PseudoRandom("TESTSEED")
        result = b.drawn_to_hand(hand, rng=rng)
        assert "forced_card_index" in result
        assert 0 <= result["forced_card_index"] < len(hand)


class TestDrawnToHandCrimsonHeart:
    """Crimson Heart: debuff a random joker."""

    def test_debuffs_one_joker(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_heart", ante=1)
        from jackdaw.engine.card_factory import create_joker

        jokers = [create_joker("j_joker"), create_joker("j_greedy_joker")]
        rng = PseudoRandom("TESTSEED")
        result = b.drawn_to_hand([], joker_cards=jokers, rng=rng)
        assert "debuffed_joker_index" in result
        # Exactly one joker should be debuffed
        debuffed = [j for j in jokers if j.debuff]
        assert len(debuffed) == 1


# ============================================================================
# stay_flipped
# ============================================================================


class TestStayFlipped:
    def test_the_house_first_hand(self):
        reset_sort_id_counter()
        b = Blind.create("bl_house", ante=1)
        c = _card("Hearts", "5")
        assert b.stay_flipped(c, hands_played=0, discards_used=0) is True


# ============================================================================
# disable
# ============================================================================


class TestDisable:
    def test_sets_disabled(self):
        b = Blind.create("bl_hook", ante=1)
        b.disable()
        assert b.disabled is True

    def test_clears_card_debuffs(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Spades", "5")
        b.debuff_card(c)
        assert c.debuff is True
        b.disable(playing_cards=[c])
        assert c.debuff is False


# ============================================================================
# get_new_boss / get_ante_blinds
# ============================================================================


class TestGetNewBoss:
    """Boss blind selection matching common_events.lua:2338."""

    def test_returns_valid_boss(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        boss = get_new_boss(1, bosses_used, rng)
        assert boss in ALL_BLINDS
        assert ALL_BLINDS[boss].boss is not None

    def test_deterministic(self):
        bosses_used1 = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        bosses_used2 = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        b1 = get_new_boss(1, bosses_used1, PseudoRandom("TESTSEED"))
        b2 = get_new_boss(1, bosses_used2, PseudoRandom("TESTSEED"))
        assert b1 == b2

    def test_showdown_at_ante_8(self):
        """At ante 8 (win_ante), only showdown blinds are eligible."""
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")

        boss = get_new_boss(8, bosses_used, rng, win_ante=8)
        proto = ALL_BLINDS[boss]
        assert proto.boss.get("showdown") is True, (
            f"{boss} is not a showdown blind but was selected at ante 8"
        )


class TestGetAnteBlinds:
    def test_structure(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        result = get_ante_blinds(1, bosses_used, rng)
        assert result["Small"] == "bl_small"
        assert result["Big"] == "bl_big"
        assert result["Boss"] in ALL_BLINDS


# ============================================================================
# Blind chip requirement scaling
# ============================================================================

# Ground truth from misc_functions.lua:922-944
SCALING_1 = [300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000]
SCALING_2 = [300, 900, 2_600, 8_000, 20_000, 36_000, 60_000, 100_000]
SCALING_3 = [300, 1_000, 3_200, 9_000, 25_000, 60_000, 110_000, 200_000]


class TestScaling1:
    """White/Red stake (scaling=1)."""

    def test_ante_1(self):
        assert get_blind_amount(1, scaling=1) == SCALING_1[0]

    def test_ante_8(self):
        assert get_blind_amount(8, scaling=1) == SCALING_1[7]


class TestScaling2:
    """Green-Blue stake (scaling=2)."""

    def test_ante_1(self):
        assert get_blind_amount(1, scaling=2) == SCALING_2[0]

    def test_ante_8(self):
        assert get_blind_amount(8, scaling=2) == SCALING_2[7]


class TestScaling3:
    """Purple-Gold stake (scaling=3)."""

    def test_ante_1(self):
        assert get_blind_amount(1, scaling=3) == SCALING_3[0]

    def test_ante_8(self):
        assert get_blind_amount(8, scaling=3) == SCALING_3[7]


# ============================================================================
# Exponential formula (antes 9+) — ground truth from LuaJIT
# ============================================================================

EXPONENTIAL_TRUTH = {
    1: {9: 110_000, 10: 560_000, 11: 7_200_000, 12: 300_000_000},
    2: {9: 230_000, 10: 1_100_000, 11: 14_000_000, 12: 600_000_000},
    3: {9: 460_000, 10: 2_200_000, 11: 29_000_000, 12: 1_200_000_000},
}


class TestExponentialFormula:
    """Antes 9-12 use exponential growth with sig-fig rounding."""

    @pytest.mark.parametrize("scaling", [1, 2, 3])
    @pytest.mark.parametrize("ante", [9, 10, 11, 12])
    def test_matches_luajit(self, scaling: int, ante: int):
        expected = EXPONENTIAL_TRUTH[scaling][ante]
        actual = get_blind_amount(ante, scaling)
        assert actual == expected, f"scaling={scaling} ante={ante}: {actual} != {expected}"


# ============================================================================
# get_blind_target (Plasma Deck)
# ============================================================================


class TestGetBlindTarget:
    def test_plasma_deck_doubles(self):
        """Plasma Deck: ante_scaling=2.0 doubles all targets."""
        normal = get_blind_target(1, "Boss", scaling=1, ante_scaling=1.0)
        plasma = get_blind_target(1, "Boss", scaling=1, ante_scaling=2.0)
        assert plasma == normal * 2
