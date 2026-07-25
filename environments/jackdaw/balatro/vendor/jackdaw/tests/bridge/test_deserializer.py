"""Tests for jackdaw.bridge.deserializer — balatrobot RPC → engine Action."""

from __future__ import annotations

import pytest

from jackdaw.bridge.balatrobot_adapter import action_to_rpc
from jackdaw.bridge.deserializer import rpc_to_action
from jackdaw.engine.actions import (
    BuyCard,
    CashOut,
    Discard,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SwapHandRight,
    SwapJokersRight,
    UseConsumable,
)

# ============================================================================
# Game action methods
# ============================================================================


class TestPlay:
    def test_play(self):
        action = rpc_to_action("play", {"cards": [0, 2, 4]})
        assert action == PlayHand(card_indices=(0, 2, 4))

    def test_play_empty(self):
        action = rpc_to_action("play", {"cards": []})
        assert action == PlayHand(card_indices=())

    def test_play_no_cards_key(self):
        action = rpc_to_action("play", {})
        assert action == PlayHand(card_indices=())


class TestDiscard:
    def test_discard(self):
        action = rpc_to_action("discard", {"cards": [1, 3]})
        assert action == Discard(card_indices=(1, 3))


class TestBlindActions:
    def test_select(self):
        assert rpc_to_action("select", {}) == SelectBlind()

    def test_skip(self):
        assert rpc_to_action("skip", {}) == SkipBlind()


class TestBuy:
    def test_buy_card(self):
        assert rpc_to_action("buy", {"card": 0}) == BuyCard(shop_index=0)

    def test_buy_voucher(self):
        assert rpc_to_action("buy", {"voucher": 1}) == RedeemVoucher(card_index=1)

    def test_buy_pack(self):
        assert rpc_to_action("buy", {"pack": 2}) == OpenBooster(card_index=2)

    def test_buy_no_params(self):
        with pytest.raises(ValueError, match="buy"):
            rpc_to_action("buy", {})


class TestSell:
    def test_sell_joker(self):
        assert rpc_to_action("sell", {"joker": 0}) == SellCard(area="jokers", card_index=0)

    def test_sell_consumable(self):
        result = rpc_to_action("sell", {"consumable": 1})
        assert result == SellCard(area="consumables", card_index=1)

    def test_sell_no_params(self):
        with pytest.raises(ValueError, match="sell"):
            rpc_to_action("sell", {})


class TestUse:
    def test_use_without_targets(self):
        result = rpc_to_action("use", {"consumable": 0})
        assert result == UseConsumable(card_index=0, target_indices=None)

    def test_use_with_targets(self):
        result = rpc_to_action("use", {"consumable": 0, "cards": [1, 3]})
        assert result == UseConsumable(card_index=0, target_indices=(1, 3))

    def test_use_missing_consumable(self):
        with pytest.raises(ValueError, match="use"):
            rpc_to_action("use", {"cards": [0]})


class TestShopActions:
    def test_reroll(self):
        assert rpc_to_action("reroll", {}) == Reroll()

    def test_next_round(self):
        assert rpc_to_action("next_round", {}) == NextRound()

    def test_cash_out(self):
        assert rpc_to_action("cash_out", {}) == CashOut()


class TestPack:
    def test_pick_card(self):
        assert rpc_to_action("pack", {"card": 2}) == PickPackCard(card_index=2)

    def test_pick_card_with_targets(self):
        result = rpc_to_action("pack", {"card": 0, "targets": [1, 2]})
        assert result == PickPackCard(card_index=0, target_indices=(1, 2))

    def test_skip_pack(self):
        assert rpc_to_action("pack", {"skip": True}) == SkipPack()

    def test_pack_no_params(self):
        with pytest.raises(ValueError, match="pack"):
            rpc_to_action("pack", {})


class TestRearrange:
    def test_rearrange_hand_adjacent(self):
        result = rpc_to_action("rearrange", {"hand": [0, 2, 1, 3]})
        assert result == SwapHandRight(idx=1)

    def test_rearrange_jokers_adjacent(self):
        result = rpc_to_action("rearrange", {"jokers": [1, 0, 2]})
        assert result == SwapJokersRight(idx=0)

    def test_rearrange_no_params(self):
        with pytest.raises(ValueError, match="rearrange"):
            rpc_to_action("rearrange", {})


# ============================================================================
# Query-only methods return None
# ============================================================================


class TestQueryMethods:
    @pytest.mark.parametrize(
        "method",
        [
            "gamestate",
            "health",
            "start",
            "menu",
            "save",
            "load",
            "screenshot",
            "set",
            "add",
            "rpc.discover",
        ],
    )
    def test_returns_none(self, method):
        assert rpc_to_action(method) is None


# ============================================================================
# Unknown method
# ============================================================================


class TestUnknownMethod:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown RPC method"):
            rpc_to_action("nonexistent_method", {})


# ============================================================================
# None params treated as empty dict
# ============================================================================


class TestNoneParams:
    def test_select_none_params(self):
        assert rpc_to_action("select", None) == SelectBlind()

    def test_select_no_params_arg(self):
        assert rpc_to_action("select") == SelectBlind()


# ============================================================================
# Roundtrip: action_to_rpc → rpc_to_action
# ============================================================================


class TestRoundtrip:
    """Verify that action_to_rpc followed by rpc_to_action recovers the
    original Action for all types that have a clean 1:1 mapping.

    SortHand is excluded because it doesn't have a direct balatrobot
    equivalent (SortHand maps to rearrange with empty hand list).
    """

    @pytest.mark.parametrize(
        "action",
        [
            PlayHand(card_indices=(0, 1, 2)),
            Discard(card_indices=(3, 4)),
            SelectBlind(),
            SkipBlind(),
            BuyCard(shop_index=2),
            RedeemVoucher(card_index=0),
            OpenBooster(card_index=1),
            SellCard(area="jokers", card_index=0),
            SellCard(area="consumables", card_index=1),
            UseConsumable(card_index=0, target_indices=None),
            UseConsumable(card_index=1, target_indices=(0, 2)),
            Reroll(),
            NextRound(),
            CashOut(),
            PickPackCard(card_index=0),
            PickPackCard(card_index=1, target_indices=(0, 3)),
            SkipPack(),
        ],
        ids=lambda a: type(a).__name__,
    )
    def test_roundtrip(self, action):
        rpc = action_to_rpc(action)
        recovered = rpc_to_action(rpc["method"], rpc["params"])
        assert recovered == action

    def test_roundtrip_swap_hand(self):
        gs = {"hand": [None] * 5}
        action = SwapHandRight(idx=2)
        rpc = action_to_rpc(action, gs)
        recovered = rpc_to_action(rpc["method"], rpc["params"])
        assert recovered == action

    def test_roundtrip_swap_jokers(self):
        gs = {"jokers": [None] * 3}
        action = SwapJokersRight(idx=1)
        rpc = action_to_rpc(action, gs)
        recovered = rpc_to_action(rpc["method"], rpc["params"])
        assert recovered == action
