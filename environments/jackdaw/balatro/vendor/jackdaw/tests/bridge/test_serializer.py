"""Tests for jackdaw.bridge.serializer — Card → balatrobot JSON."""

from __future__ import annotations

import pytest

from jackdaw.bridge.serializer import (
    game_state_to_bot_response,
    serialize_area,
    serialize_card,
    serialize_hands,
)
from jackdaw.engine.actions import CashOut, GamePhase, PlayHand, SelectBlind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.run_init import initialize_run


@pytest.fixture(autouse=True)
def _reset_sort_ids():
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


def _joker(center_key: str) -> Card:
    c = Card()
    c.set_ability(center_key)
    return c


def _consumable(center_key: str) -> Card:
    c = Card()
    c.set_ability(center_key)
    return c


# ============================================================================
# Playing cards
# ============================================================================


class TestPlayingCard:
    def test_ace_of_hearts_plain(self):
        card = _playing_card("Hearts", "Ace")
        result = serialize_card(card)

        assert result["id"] == card.sort_id
        assert result["key"] == "H_A"
        assert result["set"] == "DEFAULT"
        assert result["label"] == "Ace of Hearts"
        assert result["value"]["suit"] == "H"
        assert result["value"]["rank"] == "A"
        assert result["modifier"]["seal"] is None
        assert result["modifier"]["edition"] is None
        assert result["modifier"]["enhancement"] is None
        assert result["modifier"]["eternal"] is False
        assert result["modifier"]["perishable"] is None
        assert result["modifier"]["rental"] is False
        assert result["state"]["debuff"] is False
        assert result["state"]["hidden"] is False
        assert result["state"]["highlight"] is False
        assert result["cost"]["sell"] == card.sell_cost
        assert result["cost"]["buy"] == card.cost

    def test_ten_of_spades(self):
        card = _playing_card("Spades", "10")
        result = serialize_card(card)
        assert result["key"] == "S_T"
        assert result["value"]["rank"] == "T"
        assert result["label"] == "10 of Spades"


class TestEnhancedPlayingCard:
    def test_glass_card_with_gold_seal(self):
        card = _playing_card("Diamonds", "Queen", enhancement="m_glass")
        card.set_seal("Gold")
        result = serialize_card(card)

        assert result["set"] == "ENHANCED"
        assert result["label"] == "Queen of Diamonds"
        assert result["modifier"]["enhancement"] == "GLASS"
        assert result["modifier"]["seal"] == "GOLD"
        assert result["value"]["suit"] == "D"
        assert result["value"]["rank"] == "Q"

    def test_bonus_card(self):
        card = _playing_card("Clubs", "5", enhancement="m_bonus")
        result = serialize_card(card)
        assert result["modifier"]["enhancement"] == "BONUS"
        assert result["set"] == "ENHANCED"


# ============================================================================
# Jokers
# ============================================================================


class TestJoker:
    def test_basic_joker_with_foil(self):
        card = _joker("j_joker")
        card.set_edition({"foil": True})
        result = serialize_card(card)

        assert result["key"] == "j_joker"
        assert result["set"] == "JOKER"
        assert result["label"] == "Joker"
        assert result["value"]["suit"] is None
        assert result["value"]["rank"] is None
        assert result["value"]["effect"] == "Joker"
        assert result["modifier"]["edition"] == "FOIL"
        assert result["modifier"]["enhancement"] is None

    def test_holo_joker(self):
        card = _joker("j_joker")
        card.set_edition({"holo": True})
        assert serialize_card(card)["modifier"]["edition"] == "HOLO"

    def test_polychrome_joker(self):
        card = _joker("j_joker")
        card.set_edition({"polychrome": True})
        assert serialize_card(card)["modifier"]["edition"] == "POLYCHROME"

    def test_negative_joker(self):
        card = _joker("j_joker")
        card.set_edition({"negative": True})
        assert serialize_card(card)["modifier"]["edition"] == "NEGATIVE"


# ============================================================================
# Consumables
# ============================================================================


class TestConsumable:
    def test_tarot_fool(self):
        card = _consumable("c_fool")
        result = serialize_card(card)

        assert result["key"] == "c_fool"
        assert result["set"] == "TAROT"
        assert result["value"]["suit"] is None
        assert result["value"]["rank"] is None

    def test_planet_mercury(self):
        card = _consumable("c_mercury")
        result = serialize_card(card)

        assert result["key"] == "c_mercury"
        assert result["set"] == "PLANET"


# ============================================================================
# Vouchers
# ============================================================================


class TestVoucher:
    def test_voucher_overstock(self):
        card = _consumable("v_overstock_norm")
        result = serialize_card(card)

        assert result["key"] == "v_overstock_norm"
        assert result["set"] == "VOUCHER"
        assert result["value"]["suit"] is None


# ============================================================================
# Booster packs
# ============================================================================


class TestBooster:
    def test_arcana_pack(self):
        card = _consumable("p_arcana_normal_1")
        result = serialize_card(card)

        assert result["key"] == "p_arcana_normal_1"
        assert result["set"] == "BOOSTER"


# ============================================================================
# All modifiers combined
# ============================================================================


class TestAllModifiers:
    def test_fully_modified_card(self):
        card = _playing_card("Hearts", "King", enhancement="m_glass")
        card.set_edition({"polychrome": True})
        card.set_seal("Purple")
        card.set_eternal(True)
        card.set_perishable(True)
        card.set_rental(True)
        card.set_debuff(True)

        result = serialize_card(card)

        assert result["modifier"]["seal"] == "PURPLE"
        assert result["modifier"]["edition"] == "POLYCHROME"
        assert result["modifier"]["enhancement"] == "GLASS"
        assert result["modifier"]["eternal"] is True
        assert result["modifier"]["perishable"] == 5  # default perish_tally
        assert result["modifier"]["rental"] is True
        assert result["state"]["debuff"] is True


# ============================================================================
# Hidden (facing back)
# ============================================================================


class TestHiddenCard:
    def test_card_facing_back(self):
        card = _playing_card("Spades", "Ace")
        card.facing = "back"
        result = serialize_card(card)

        assert result["state"]["hidden"] is True

    def test_card_facing_front(self):
        card = _playing_card("Spades", "Ace")
        result = serialize_card(card)

        assert result["state"]["hidden"] is False


# ============================================================================
# Area serialization
# ============================================================================


class TestSerializeArea:
    def test_empty(self):
        result = serialize_area([], limit=5)

        assert result["count"] == 0
        assert result["limit"] == 5
        assert result["highlighted_limit"] == 0
        assert result["cards"] == []

    def test_with_cards(self):
        jokers = [_joker("j_joker"), _joker("j_greedy_joker"), _joker("j_lusty_joker")]
        result = serialize_area(jokers, limit=5)

        assert result["count"] == 3
        assert result["limit"] == 5
        assert result["highlighted_limit"] == 0
        assert len(result["cards"]) == 3
        assert result["cards"][0]["key"] == "j_joker"
        assert result["cards"][1]["key"] == "j_greedy_joker"
        assert result["cards"][2]["key"] == "j_lusty_joker"

    def test_highlighted_limit(self):
        cards = [_playing_card("Hearts", "Ace")]
        result = serialize_area(cards, limit=8, highlighted_limit=5)

        assert result["highlighted_limit"] == 5


# ============================================================================
# Poker hand serialization
# ============================================================================


_VISIBLE_HANDS = {
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
}

_SECRET_HANDS = {"Flush Five", "Flush House", "Five of a Kind"}


class TestSerializeHands:
    def test_default_levels(self):
        hl = HandLevels()
        result = serialize_hands(hl)

        # 9 visible, 3 secret absent
        assert set(result.keys()) == _VISIBLE_HANDS
        for name in _SECRET_HANDS:
            assert name not in result

        # Spot-check a few entries
        hc = result["High Card"]
        assert hc["order"] == 12
        assert hc["level"] == 1
        assert hc["chips"] == 5
        assert hc["mult"] == 1
        assert hc["played"] == 0
        assert hc["played_this_round"] == 0
        assert hc["example"] == []

        pair = result["Pair"]
        assert pair["order"] == 11
        assert pair["chips"] == 10
        assert pair["mult"] == 2

        sf = result["Straight Flush"]
        assert sf["order"] == 4
        assert sf["chips"] == 100
        assert sf["mult"] == 8

    def test_leveled_pair(self):
        hl = HandLevels()
        hl.level_up("Pair", 2)  # level 1 → 3
        result = serialize_hands(hl)

        pair = result["Pair"]
        assert pair["level"] == 3
        # chips = 10 + 15 * (3-1) = 40
        assert pair["chips"] == 40
        # mult = 2 + 1 * (3-1) = 4
        assert pair["mult"] == 4

    def test_secret_hand_visible_after_level_up(self):
        hl = HandLevels()
        assert "Flush Five" not in serialize_hands(hl)

        hl.level_up("Flush Five")
        result = serialize_hands(hl)
        assert "Flush Five" in result
        assert result["Flush Five"]["level"] == 2


# ============================================================================
# Blind serialization
# ============================================================================


class TestBlindSerialization:
    def test_all_three_blinds_present(self):
        gs = initialize_run("b_red", 1, "BLIND_TEST")
        result = game_state_to_bot_response(gs)
        blinds = result["blinds"]

        assert set(blinds.keys()) == {"small", "big", "boss"}
        assert blinds["small"]["type"] == "SMALL"
        assert blinds["big"]["type"] == "BIG"
        assert blinds["boss"]["type"] == "BOSS"

    def test_blind_names(self):
        gs = initialize_run("b_red", 1, "BLIND_TEST")
        result = game_state_to_bot_response(gs)
        blinds = result["blinds"]

        assert blinds["small"]["name"] == "Small Blind"
        assert blinds["big"]["name"] == "Big Blind"
        # Boss name depends on seed, but should be non-empty
        assert blinds["boss"]["name"] != ""

    def test_blind_status_at_init(self):
        gs = initialize_run("b_red", 1, "BLIND_TEST")
        result = game_state_to_bot_response(gs)
        blinds = result["blinds"]

        assert blinds["small"]["status"] == "SELECT"
        assert blinds["big"]["status"] == "UPCOMING"
        assert blinds["boss"]["status"] == "UPCOMING"

    def test_blind_scores_positive(self):
        gs = initialize_run("b_red", 1, "BLIND_TEST")
        result = game_state_to_bot_response(gs)
        blinds = result["blinds"]

        assert blinds["small"]["score"] > 0
        assert blinds["big"]["score"] > 0
        assert blinds["boss"]["score"] > 0
        # Big > Small due to mult
        assert blinds["big"]["score"] > blinds["small"]["score"]

    def test_small_big_no_special_effect(self):
        gs = initialize_run("b_red", 1, "BLIND_TEST")
        result = game_state_to_bot_response(gs)

        assert result["blinds"]["small"]["effect"] == "No special effect"
        assert result["blinds"]["big"]["effect"] == "No special effect"

    def test_blind_status_after_select(self):
        gs = initialize_run("b_red", 1, "BLIND_TEST")
        step(gs, SelectBlind())
        result = game_state_to_bot_response(gs)
        blinds = result["blinds"]

        assert blinds["small"]["status"] == "CURRENT"


# ============================================================================
# game_state_to_bot_response — top-level
# ============================================================================


class TestGameStateToBotResponseBlindSelect:
    def test_top_level_fields(self):
        gs = initialize_run("b_red", 1, "TEST")
        result = game_state_to_bot_response(gs)

        assert result["state"] == "BLIND_SELECT"
        assert result["deck"] == "RED"
        assert result["stake"] == "WHITE"
        assert result["seed"] == "TEST"
        assert result["won"] is False
        assert result["ante_num"] == 1
        assert result["money"] == 4
        assert isinstance(result["used_vouchers"], dict)
        assert isinstance(result["hands"], dict)
        assert isinstance(result["blinds"], dict)

    def test_areas_present(self):
        gs = initialize_run("b_red", 1, "TEST")
        result = game_state_to_bot_response(gs)

        # Deck should have cards
        assert result["cards"]["count"] > 0
        assert len(result["cards"]["cards"]) == result["cards"]["count"]

        # Jokers/consumables empty at start
        assert result["jokers"]["count"] == 0
        assert result["consumables"]["count"] == 0

        # Hand empty before selecting blind
        assert result["hand"]["count"] == 0
        assert result["hand"]["highlighted_limit"] == 5


class TestGameStateToBotResponseSelectingHand:
    def test_hand_and_round_populated(self):
        gs = initialize_run("b_red", 1, "TEST")
        step(gs, SelectBlind())
        result = game_state_to_bot_response(gs)

        assert result["state"] == "SELECTING_HAND"
        assert result["hand"]["count"] > 0
        assert result["round"]["hands_left"] > 0
        assert result["round"]["hands_played"] == 0
        assert result["round"]["discards_left"] >= 0


class TestGameStateToBotResponseShop:
    def test_shop_areas_populated(self):
        gs = initialize_run("b_red", 1, "SHOP_TEST")
        step(gs, SelectBlind())

        # Set blind to 1 chip so first hand wins
        gs["blind"].chips = 1
        hand = gs.get("hand", [])
        n = min(5, len(hand))
        step(gs, PlayHand(card_indices=tuple(range(n))))

        assert gs["phase"] == GamePhase.ROUND_EVAL
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP

        result = game_state_to_bot_response(gs)

        assert result["state"] == "SHOP"
        assert result["shop"]["count"] >= 0
        assert result["vouchers"]["count"] >= 0
        assert result["packs"]["count"] >= 0


class TestStateEnumRoundtrip:
    @pytest.mark.parametrize(
        "phase,expected",
        [
            (GamePhase.BLIND_SELECT, "BLIND_SELECT"),
            (GamePhase.SELECTING_HAND, "SELECTING_HAND"),
            (GamePhase.ROUND_EVAL, "ROUND_EVAL"),
            (GamePhase.SHOP, "SHOP"),
            (GamePhase.PACK_OPENING, "SMODS_BOOSTER_OPENED"),
            (GamePhase.GAME_OVER, "GAME_OVER"),
        ],
    )
    def test_phase_to_state_string(self, phase, expected):
        gs = initialize_run("b_red", 1, "PHASE_TEST")
        gs["phase"] = phase
        result = game_state_to_bot_response(gs)
        assert result["state"] == expected
