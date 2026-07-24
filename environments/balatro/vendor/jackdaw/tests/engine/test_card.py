"""Tests for the Card class.

Verifies card creation from prototypes (both string-key and dict APIs),
base field population, ability initialization with post-init fields,
scoring methods, deep-copy isolation, and sort_id auto-increment.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, CardBase, reset_sort_id_counter
from jackdaw.engine.card_area import CardArea, draw_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import JOKERS
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset_sort_ids():
    """Reset sort_id counter before each test for determinism."""
    reset_sort_id_counter()


def _playing_card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
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


# ============================================================================
# CardBase
# ============================================================================


class TestCardBase:
    """CardBase: playing card identity with computed numeric values."""

    def test_ace_of_spades(self):
        b = CardBase.from_card_key("S_A", "Spades", "Ace")
        assert b.suit is Suit.SPADES
        assert b.rank is Rank.ACE
        assert b.id == 14
        assert b.nominal == 11
        assert b.suit_nominal == 0.04
        assert b.suit_nominal_original == 0.004
        assert b.face_nominal == 0.4
        assert b.original_value is Rank.ACE
        assert b.times_played == 0


# ============================================================================
# Card creation and sort_id
# ============================================================================


class TestCardCreation:
    def test_defaults(self):
        c = Card()
        assert c.base is None
        assert c.center_key == "c_base"
        assert c.ability == {}
        assert c.edition is None
        assert c.seal is None
        assert c.debuff is False
        assert c.eternal is False
        assert c.perish_tally == 5


# ============================================================================
# set_ability — string key API (primary interface)
# ============================================================================


class TestSetAbilityByKey:
    """set_ability(center_key: str) looks up P_CENTERS and populates ability."""

    def test_duo_xmult(self):
        """j_duo: config.Xmult=2 (top-level) → ability.x_mult=2."""
        c = Card()
        c.set_ability("j_duo")
        assert c.ability["x_mult"] == 2
        assert c.ability["name"] == "The Duo"
        assert c.ability["type"] == "Pair"

    def test_ice_cream_extra_chips(self):
        """j_ice_cream: config.extra.chips=100."""
        c = Card()
        c.set_ability("j_ice_cream")
        assert c.ability["extra"]["chips"] == 100
        assert c.ability["extra"]["chip_mod"] == 5

    def test_bonus_card_enhancement(self):
        """m_bonus: config.bonus=30."""
        c = Card()
        c.set_base("D_5", "Diamonds", "5")
        c.set_ability("m_bonus")
        assert c.ability["bonus"] == 30
        assert c.get_chip_bonus() == 5 + 30

    def test_invalid_key_raises(self):
        c = Card()
        with pytest.raises(KeyError, match="Unknown center key"):
            c.set_ability("j_nonexistent_joker")


# ============================================================================
# set_ability — special post-init fields (card.lua:308-337)
# ============================================================================


class TestPostInitFields:
    """Joker-specific fields set after the main ability assignment."""

    def test_caino(self):
        c = Card()
        c.set_ability("j_caino")
        assert c.ability["caino_xmult"] == 1

    def test_yorick(self):
        c = Card()
        c.set_ability("j_yorick")
        assert c.ability["yorick_discards"] == c.ability["extra"]["discards"]
        assert c.ability["yorick_discards"] > 0  # should be 23

    def test_loyalty_card(self):
        c = Card()
        c.set_ability("j_loyalty_card")
        assert c.ability["loyalty_remaining"] == c.ability["extra"]["every"]
        assert c.ability["loyalty_remaining"] == 5
        assert c.ability["burnt_hand"] == 0


# ============================================================================
# Deep copy isolation
# ============================================================================


class TestDeepCopyIsolation:
    """Mutating one card's extra must not affect another card or the prototype."""

    def test_two_cards_from_same_proto(self):
        c1 = Card()
        c1.set_ability("j_greedy_joker")
        c2 = Card()
        c2.set_ability("j_greedy_joker")

        c1.ability["extra"]["s_mult"] = 999
        assert c2.ability["extra"]["s_mult"] == 3  # unaffected

    def test_prototype_unaffected(self):
        c = Card()
        c.set_ability("j_greedy_joker")
        c.ability["extra"]["s_mult"] = 999
        assert JOKERS["j_greedy_joker"].config["extra"]["s_mult"] == 3


# ============================================================================
# set_ability — dict API (backward compat)
# ============================================================================


class TestSetAbilityByDict:
    """set_ability(dict) still works for custom/test centers."""

    def test_raw_dict(self):
        c = Card()
        c.set_ability(
            {
                "key": "test_joker",
                "name": "Test",
                "set": "Joker",
                "effect": "Mult",
                "config": {"mult": 7},
                "cost": 3,
            }
        )
        assert c.ability["name"] == "Test"
        assert c.ability["mult"] == 7
        assert c.center_key == "test_joker"
        assert c.base_cost == 3


# ============================================================================
# perma_bonus preservation
# ============================================================================


class TestPermaBonus:
    def test_preserved_across_key_change(self):
        c = Card()
        c.set_base("H_3", "Hearts", "3")
        c.set_ability("c_base")
        c.ability["perma_bonus"] = 10
        c.set_ability("m_bonus")  # change to Bonus Card
        assert c.ability["perma_bonus"] == 10
        assert c.get_chip_bonus() == 3 + 30 + 10  # nominal + bonus + perma


# ============================================================================
# Scoring methods
# ============================================================================


class TestIsFace:
    """Card.is_face() matching card.lua:964."""

    def test_jack_is_face(self):
        c = Card()
        c.set_base("H_J", "Hearts", "Jack")
        assert c.is_face() is True

    def test_ace_is_not_face(self):
        c = Card()
        c.set_base("D_A", "Diamonds", "Ace")
        assert c.is_face() is False

    def test_debuffed_is_not_face(self):
        c = Card()
        c.set_base("H_K", "Hearts", "King")
        c.debuff = True
        assert c.is_face() is False

    def test_pareidolia_makes_all_face(self):
        """Pareidolia joker: ALL cards are face cards."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        assert c.is_face(pareidolia=True) is True


class TestIsSuit:
    """Card.is_suit() matching card.lua:4064."""

    def test_basic_match(self):
        c = Card()
        c.set_base("S_A", "Spades", "Ace")
        assert c.is_suit("Spades") is True
        assert c.is_suit("Hearts") is False

    def test_wild_card_matches_all(self):
        """Wild Card matches every suit."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_wild")
        assert c.is_suit("Spades") is True
        assert c.is_suit("Hearts") is True
        assert c.is_suit("Diamonds") is True
        assert c.is_suit("Clubs") is True

    def test_wild_card_debuffed(self):
        """Debuffed Wild Card: debuff check returns False before Wild check."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_wild")
        c.debuff = True
        assert c.is_suit("Spades") is False

    def test_stone_card_never_matches(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_stone")
        assert c.is_suit("Hearts") is False

    def test_smeared_red_suits(self):
        """Smeared: Heart matches Diamond (both red)."""
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("c_base")
        assert c.is_suit("Diamonds", smeared=True) is True
        assert c.is_suit("Hearts", smeared=True) is True


class TestScoringMethods:
    def test_get_id_stone_card(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_stone")
        assert c.get_id() == -1

    # -- Scoring Methods (from card_scoring) --

    # get_chip_bonus

    def test_stone_card_override(self):
        """Stone Card: ignores nominal, returns bonus only (50)."""
        c = _playing_card("Hearts", "Ace", enhancement="m_stone")
        assert c.get_chip_bonus() == 50

    def test_perma_bonus(self):
        c = _playing_card("Hearts", "5")
        c.ability["perma_bonus"] = 15
        assert c.get_chip_bonus() == 5 + 15

    # get_chip_mult

    def test_mult_card(self):
        """Mult Card: ability.mult = 4."""
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        assert c.get_chip_mult() == 4

    # get_chip_x_mult

    def test_glass_card(self):
        """Glass Card: x_mult = 2.0."""
        c = _playing_card("Hearts", "5", enhancement="m_glass")
        assert c.get_chip_x_mult() == 2

    # get_chip_h_x_mult

    def test_steel_card(self):
        """Steel Card: h_x_mult = 1.5."""
        c = _playing_card("Hearts", "5", enhancement="m_steel")
        assert c.get_chip_h_x_mult() == pytest.approx(1.5)

    # get_edition

    def test_foil_edition(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"foil": True})
        ed = c.get_edition()
        assert ed is not None
        assert ed["chip_mod"] == 50
        assert "mult_mod" not in ed
        assert "x_mult_mod" not in ed

    def test_holographic_edition(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"holo": True})
        ed = c.get_edition()
        assert ed is not None
        assert ed["mult_mod"] == 10
        assert "chip_mod" not in ed

    def test_polychrome_edition(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"polychrome": True})
        ed = c.get_edition()
        assert ed is not None
        assert ed["x_mult_mod"] == pytest.approx(1.5)
        assert "chip_mod" not in ed
        assert "mult_mod" not in ed

    def test_edition_has_card_ref(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"holo": True})
        ed = c.get_edition()
        assert ed["card"] is c

    # get_p_dollars

    def test_gold_seal(self):
        """Gold Seal: +$3 on score."""
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        assert c.get_p_dollars() == 3

    def test_gold_card_enhancement(self):
        """Gold Card enhancement: h_dollars=3, NOT p_dollars.

        Gold Card earns $3 when HELD at end of round, not when scored.
        get_p_dollars should return 0 for Gold Card (p_dollars=0 in config).
        """
        c = _playing_card("Hearts", "5", enhancement="m_gold")
        assert c.get_p_dollars() == 0  # h_dollars, not p_dollars

    # calculate_seal

    def test_red_seal_repetition(self):
        c = _playing_card("Hearts", "5")
        c.set_seal("Red")
        result = c.calculate_seal(repetition=True)
        assert result is not None
        assert result["repetitions"] == 1
        assert result["card"] is c


# ============================================================================
# Stickers
# ============================================================================


class TestStickers:
    def test_set_perishable_resets_tally(self):
        c = Card()
        c.perish_tally = 0
        c.set_perishable(True)
        assert c.perish_tally == 5


# ============================================================================
# set_cost
# ============================================================================


class TestSetCost:
    """Card.set_cost() matching card.lua:369."""

    def _make_joker(self, key: str = "j_joker") -> Card:
        """Helper: create a joker with set_ability."""
        c = Card()
        c.set_ability(key)
        return c

    # -- Base formula --

    def test_base_cost_no_modifiers(self):
        """j_joker: base_cost=2, no inflation/discount/edition."""
        c = self._make_joker("j_joker")
        c.set_cost()
        # floor((2 + 0 + 0.5) * 100/100) = floor(2.5) = 2
        assert c.cost == 2

    # -- Discount --

    def test_25_percent_discount(self):
        """Clearance Sale: 25% discount on cost=5."""
        c = self._make_joker("j_greedy_joker")
        c.set_cost(discount_percent=25)
        # floor((5 + 0 + 0.5) * 75/100) = floor(4.125) = 4
        assert c.cost == 4

    # -- Edition surcharges --

    def test_foil_surcharge(self):
        """Foil edition adds +2 to cost."""
        c = self._make_joker("j_greedy_joker")
        c.set_edition({"foil": True})
        c.set_cost()
        # floor((5 + 2 + 0.5) * 100/100) = floor(7.5) = 7
        assert c.cost == 7

    # -- Rental override --

    def test_rental_override(self):
        """Rental cards always cost 1."""
        c = self._make_joker("j_greedy_joker")
        c.set_rental(True)
        c.set_cost()
        assert c.cost == 1
        assert c.sell_cost == 1  # floor(1/2) = 0, clamped to 1


# ============================================================================
# CardArea and draw_card (merged from test_card_area.py)
# ============================================================================


_RANK_LETTER = {
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


def _make_playing_card(suit: str, rank: str) -> Card:
    """Helper: create a Card with base set."""
    c = Card()
    key = f"{suit[0]}_{_RANK_LETTER[rank]}"
    c.set_base(key, suit, rank)
    c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
    return c


def _make_deck(n: int = 52) -> list[Card]:
    """Helper: create n cards with sequential sort_ids and bases."""
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
    rank_letters = {
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
    cards = []
    for suit in suits:
        for rank in ranks:
            c = Card()
            key = f"{suit[0]}_{rank_letters[rank]}"
            c.set_base(key, suit, rank)
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            cards.append(c)
            if len(cards) >= n:
                return cards
    return cards


class TestCardAreaBasics:
    def test_add_and_len(self):
        area = CardArea()
        c = Card()
        area.add(c)
        assert len(area) == 1
        assert area.cards[0] is c

    def test_remove(self):
        area = CardArea()
        c1, c2, c3 = Card(), Card(), Card()
        area.add(c1)
        area.add(c2)
        area.add(c3)
        removed = area.remove(c2)
        assert removed is c2
        assert len(area) == 2
        assert c2 not in area.cards

    def test_has_space(self):
        area = CardArea(card_limit=3)
        assert area.has_space() is True
        area.add(Card())
        area.add(Card())
        assert area.has_space() is True
        area.add(Card())
        assert area.has_space() is False


class TestHighlighting:
    def test_remove_also_unhighlights(self):
        area = CardArea()
        c = Card()
        area.add(c)
        area.add_to_highlighted(c)
        area.remove(c)
        assert c not in area.highlighted


class TestDrawCard:
    def test_draw_from_deck_to_hand(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=8, area_type="hand")
        cards = _make_deck(52)
        for c in cards:
            deck.add(c)
        assert len(deck) == 52

        drawn = draw_card(deck, hand, count=8)
        assert len(drawn) == 8
        assert len(deck) == 44
        assert len(hand) == 8

    def test_draw_respects_target_limit(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=3, area_type="hand")
        for c in _make_deck(10):
            deck.add(c)

        drawn = draw_card(deck, hand, count=5)
        assert len(drawn) == 3  # limited by hand capacity
        assert len(hand) == 3
        assert len(deck) == 7


class TestShuffle:
    def test_deterministic_shuffle(self):
        """Same seed produces same card order."""

        def make_area():
            reset_sort_id_counter()
            area = CardArea(card_limit=52, area_type="deck")
            for c in _make_deck(10):
                area.add(c)
            return area

        a1 = make_area()
        rng1 = PseudoRandom("TESTSEED")
        a1.shuffle(rng1, "shuffle")
        ids1 = [c.sort_id for c in a1.cards]

        a2 = make_area()
        rng2 = PseudoRandom("TESTSEED")
        a2.shuffle(rng2, "shuffle")
        ids2 = [c.sort_id for c in a2.cards]

        assert ids1 == ids2


class TestSort:
    def test_sort_by_value_descending(self):
        area = CardArea(card_limit=10, area_type="hand")
        # Add cards in arbitrary order
        for rank in ["3", "Ace", "7", "King", "2"]:
            c = Card()
            c.set_base(f"H_{rank[0]}", "Hearts", rank)
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            area.add(c)

        area.sort_by_value(descending=True)
        ranks = [c.base.rank for c in area.cards]
        # Ace(11) > King(10) > 7 > 3 > 2
        assert ranks[0] is Rank.ACE
        assert ranks[1] is Rank.KING
        assert ranks[-1] is Rank.TWO
