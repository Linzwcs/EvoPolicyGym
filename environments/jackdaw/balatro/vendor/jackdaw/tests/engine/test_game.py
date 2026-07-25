"""Tests for jackdaw.engine.game — step function (trimmed).

Coverage: one test per action handler + key edge cases + integration.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.bridge.balatrobot_adapter import action_to_rpc
from jackdaw.engine.actions import (
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapJokersLeft,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.card import Card
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_gs(seed: str = "GAME_TEST") -> dict[str, Any]:
    """Create a fully initialised game_state ready for blind selection."""
    return initialize_run("b_red", 1, seed)


def _joker_card(key: str = "j_joker", **kw) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    c.sell_cost = kw.pop("sell_cost", 3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _setup_shop(seed="SHOP_TEST"):
    """Set up a game state in the SHOP phase after beating Small Blind."""
    gs = _init_gs(seed)
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    return gs


def _make_consumable(key: str, set_name: str = "Tarot", **kw) -> Card:
    c = Card(center_key=key, cost=0)
    ability = {"set": set_name, "effect": ""}
    ability.update(kw.pop("extra_ability", {}))
    c.ability = ability
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# SelectBlind
# ---------------------------------------------------------------------------


class TestSelectBlind:
    def test_phase_transitions_to_selecting_hand(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_blind_created(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["blind"] is not None
        assert gs["blind"].chips > 0


# ---------------------------------------------------------------------------
# SkipBlind
# ---------------------------------------------------------------------------


class TestSkipBlind:
    def test_skip_small_advances_to_big(self):
        gs = _init_gs()
        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Big"
        assert gs["round_resets"]["blind_states"]["Small"] == "Skipped"
        assert gs["round_resets"]["blind_states"]["Big"] == "Select"
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_skip_boss_raises(self):
        gs = _init_gs()
        gs["blind_on_deck"] = "Boss"
        with pytest.raises(IllegalActionError, match="Cannot skip Boss"):
            step(gs, SkipBlind())


# ---------------------------------------------------------------------------
# PlayHand
# ---------------------------------------------------------------------------


class TestPlayHand:
    def _setup_playing(self, seed="PLAY_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_chips_accumulate(self):
        gs = self._setup_playing()
        hand = gs["hand"]
        step(gs, PlayHand(card_indices=tuple(range(min(5, len(hand))))))
        assert gs["chips"] > 0

    def test_hands_left_decremented(self):
        gs = self._setup_playing()
        initial = gs["current_round"]["hands_left"]
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["current_round"]["hands_left"] == initial - 1

    def test_round_won_transitions_to_round_eval(self):
        gs = self._setup_playing()
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_game_over_when_no_hands_and_not_beaten(self):
        gs = self._setup_playing()
        gs["current_round"]["hands_left"] = 1
        gs["blind"].chips = 999_999_999
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["phase"] == GamePhase.GAME_OVER


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


class TestDiscard:
    def _setup_playing(self, seed="DISC_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_discards_left_decremented(self):
        gs = self._setup_playing()
        initial = gs["current_round"]["discards_left"]
        step(gs, Discard(card_indices=(0,)))
        assert gs["current_round"]["discards_left"] == initial - 1


# ---------------------------------------------------------------------------
# CashOut
# ---------------------------------------------------------------------------


class TestCashOut:
    def _setup_round_eval(self):
        gs = _init_gs("CASHOUT_TEST")
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        return gs

    def test_dollars_increase(self):
        gs = self._setup_round_eval()
        before = gs["dollars"]
        step(gs, CashOut())
        assert gs["dollars"] >= before

    def test_phase_transitions_to_shop(self):
        gs = self._setup_round_eval()
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP


# ---------------------------------------------------------------------------
# SellCard
# ---------------------------------------------------------------------------


class TestSellCard:
    def test_sell_joker(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(sell_cost=5)]
        before = gs["dollars"]
        step(gs, SellCard(area="jokers", card_index=0))
        assert gs["dollars"] == before + 5
        assert len(gs["jokers"]) == 0

    def test_eternal_blocks_sale(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(eternal=True)]
        with pytest.raises(IllegalActionError, match="eternal"):
            step(gs, SellCard(area="jokers", card_index=0))


# ---------------------------------------------------------------------------
# NextRound
# ---------------------------------------------------------------------------


class TestNextRound:
    def test_phase_transitions_to_blind_select(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT


# ---------------------------------------------------------------------------
# SortHand
# ---------------------------------------------------------------------------


class TestSortHand:
    def test_hand_reordered(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        step(gs, SortHand(mode="rank"))
        ids = [c.base.id for c in gs["hand"] if c.base]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# SwapJokers
# ---------------------------------------------------------------------------


class TestSwapJokers:
    def test_swap_joker_left(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        j0 = _joker_card("j_a")
        j1 = _joker_card("j_b")
        j2 = _joker_card("j_c")
        gs["jokers"] = [j0, j1, j2]
        step(gs, SwapJokersLeft(idx=2))
        assert gs["jokers"] == [j0, j2, j1]


# ---------------------------------------------------------------------------
# Reroll
# ---------------------------------------------------------------------------


class TestReroll:
    def test_dollars_deducted(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 10
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["dollars"] == 5


# ---------------------------------------------------------------------------
# Full mini-game (integration)
# ---------------------------------------------------------------------------


class TestMiniGame:
    def test_select_blind_play_hand_cash_out_next_round(self):
        """Full loop: blind select -> play hand -> cash out -> next round."""
        gs = _init_gs("MINI_GAME")

        assert gs["phase"] == GamePhase.BLIND_SELECT
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND

        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["dollars"] > 0

        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT


# ---------------------------------------------------------------------------
# Extra edge cases: pack opening, buy_and_use, full ante
# ---------------------------------------------------------------------------


class TestPackOpening:
    def test_open_booster_transitions_to_pack(self):
        gs = _setup_shop("OPEN_PACK")
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        step(gs, OpenBooster(card_index=0))
        assert gs["phase"] == GamePhase.PACK_OPENING

    def test_skip_pack_returns_to_shop(self):
        gs = _setup_shop("SKIP_PACK_TEST")
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        step(gs, OpenBooster(card_index=0))
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["pack_cards"] == []


class TestUseConsumable:
    def test_use_planet_in_shop(self):
        gs = _setup_shop("MERCURY_TEST")
        from jackdaw.engine.data.hands import HandType

        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        hl = gs["hand_levels"]
        level_before = hl[HandType.PAIR].level
        step(gs, UseConsumable(card_index=0))
        assert hl[HandType.PAIR].level == level_before + 1


class TestFullAnteProgression:
    def test_full_ante_cycle(self):
        gs = _init_gs("FULL_ANTE")

        # Small Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["blind_on_deck"] == "Big"

        # Big Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["blind_on_deck"] == "Boss"

        # Boss Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["round_resets"]["ante"] == 2
        assert gs["blind_on_deck"] == "Small"


# ============================================================================
# Legal actions (merged from test_actions.py)
# ============================================================================


def _action_card(key: str = "c_base", cost: int = 0, **kw) -> Card:
    c = Card(center_key=key, cost=cost)
    c.ability = kw.pop("ability", {"set": "", "effect": ""})
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _action_joker_card(key: str = "j_joker", cost: int = 5, **kw) -> Card:
    c = Card(center_key=key, cost=cost)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestLegalBlindSelect:
    def test_boss_blind_no_skip(self):
        gs = {"phase": GamePhase.BLIND_SELECT, "blind_on_deck": "Boss"}
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert SelectBlind in types
        assert SkipBlind not in types


class TestLegalSelectingHand:
    def test_play_and_discard_available(self):
        hand = [_action_card() for _ in range(8)]
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 3, "discards_left": 2},
            "hand": hand,
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand in types
        assert Discard in types

    def test_no_hands_left_no_play(self):
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "current_round": {"hands_left": 0, "discards_left": 2},
            "hand": [_action_card()],
            "jokers": [],
            "consumables": [],
        }
        actions = get_legal_actions(gs)
        types = {type(a) for a in actions}
        assert PlayHand not in types
        assert Discard in types


class TestLegalShop:
    def test_buy_affordable_joker(self):
        joker = _action_joker_card(cost=5)
        gs = {
            "phase": GamePhase.SHOP,
            "dollars": 10,
            "jokers": [],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "shop_cards": [joker],
            "shop_vouchers": [],
            "shop_boosters": [],
            "current_round": {"reroll_cost": 5, "free_rerolls": 0},
        }
        actions = get_legal_actions(gs)
        buys = [a for a in actions if isinstance(a, BuyCard)]
        assert len(buys) == 1
        assert buys[0].shop_index == 0


class TestLegalPackOpening:
    def test_pick_and_skip(self):
        gs = {
            "phase": GamePhase.PACK_OPENING,
            "pack_cards": [_action_card(), _action_card(), _action_card()],
            "pack_choices_remaining": 1,
        }
        actions = get_legal_actions(gs)
        picks = [a for a in actions if isinstance(a, PickPackCard)]
        assert len(picks) == 3
        assert any(isinstance(a, SkipPack) for a in actions)


class TestLegalRoundEval:
    def test_cashout(self):
        gs = {"phase": GamePhase.ROUND_EVAL, "consumables": []}
        actions = get_legal_actions(gs)
        assert any(isinstance(a, CashOut) for a in actions)


class TestLegalGameOver:
    def test_empty(self):
        gs = {"phase": GamePhase.GAME_OVER}
        assert get_legal_actions(gs) == []


# ============================================================================
# Balatrobot adapter (merged from test_balatrobot_adapter.py)
# ============================================================================


class TestActionToRpc:
    def test_play_hand(self):
        rpc = action_to_rpc(PlayHand(card_indices=(0, 2, 4)))
        assert rpc["method"] == "play"
        assert rpc["params"]["cards"] == [0, 2, 4]

    def test_buy_card(self):
        rpc = action_to_rpc(BuyCard(shop_index=1))
        assert rpc["method"] == "buy"
        assert rpc["params"]["card"] == 1

    def test_sell_joker(self):
        rpc = action_to_rpc(SellCard(area="jokers", card_index=2))
        assert rpc["method"] == "sell"
        assert rpc["params"]["joker"] == 2

    def test_reroll(self):
        rpc = action_to_rpc(Reroll())
        assert rpc["method"] == "reroll"

    def test_pick_pack_card(self):
        rpc = action_to_rpc(PickPackCard(card_index=2))
        assert rpc["method"] == "pack"
        assert rpc["params"]["card"] == 2


# ============================================================================
# Mechanics checklist (merged from test_mechanics_checklist.py)
# ============================================================================


def _mech_init(seed: str = "MECH") -> dict[str, Any]:
    return initialize_run("b_red", 1, seed)


def _mech_joker(key: str = "j_joker", **kw) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    c.sell_cost = 3
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestCardFlipping:
    def test_card_has_facing_attribute(self):
        c = Card()
        assert hasattr(c, "facing")
        assert c.facing == "front"

    def test_facing_can_be_set_to_back(self):
        c = Card()
        c.facing = "back"
        assert c.facing == "back"

    def test_the_fish_flips_cards(self):
        """The Fish boss flips hand cards after play."""
        gs = _mech_init("FISH_FLIP")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_fish"
        step(gs, SelectBlind())
        # Play a hand
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # After play, remaining hand cards should be flipped
        if gs["phase"] == GamePhase.SELECTING_HAND:
            for card in gs["hand"]:
                assert card.facing == "back", f"{card.card_key} not flipped"


class TestBossPressPlay:
    def test_the_hook_discards(self):
        """The Hook discards 2 random hand cards on play."""
        gs = _mech_init("HOOK_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_hook"
        step(gs, SelectBlind())
        len(gs["hand"])
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0,)))
        # Hand should have fewer cards (1 played + 2 hooked + replacements drawn)
        # The net effect depends on deck size, but discard_pile should have entries
        assert len(gs.get("discard_pile", [])) >= 2

    def test_the_tooth_costs_dollars(self):
        """The Tooth costs $1 per card played."""
        gs = _mech_init("TOOTH_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_tooth"
        step(gs, SelectBlind())
        dollars_before = gs["dollars"]
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        # Should lose $3 (ignoring scoring dollars)
        result = gs["last_score_result"]
        expected = dollars_before - 3 + result.dollars_earned
        assert gs["dollars"] == expected


class TestSealEffects:
    def test_purple_seal_creates_tarot_on_discard(self):
        """Discarding a Purple Seal card creates a Tarot consumable."""
        gs = _mech_init("PURPLE_SEAL")
        step(gs, SelectBlind())
        gs["hand"][0].seal = "Purple"
        initial_cons = len(gs.get("consumables", []))
        step(gs, Discard(card_indices=(0,)))
        assert len(gs.get("consumables", [])) == initial_cons + 1


class TestDoubleTag:
    def test_double_tag_duplicates(self):
        """Double Tag duplicates a newly awarded tag."""
        gs = _mech_init("DOUBLE_TAG")
        # Put a Double Tag in the active tags
        gs["tags"] = ["tag_double"]
        # Force Small tag to be tag_economy
        gs["round_resets"]["blind_tags"]["Small"] = "tag_economy"
        gs["dollars"] = 10
        step(gs, SkipBlind())
        # Should have received tag_economy + a duplicate
        awarded = gs.get("awarded_tags", [])
        economy_awards = [a for a in awarded if a["key"] == "tag_economy"]
        assert len(economy_awards) >= 2, f"Expected 2 economy tags, got {len(economy_awards)}"
