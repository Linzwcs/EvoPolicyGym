"""Roundtrip validation — serializer ↔ deserializer ↔ backend end-to-end."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import NOT_ALLOWED, RPCError, SimBackend
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
# Helpers
# ============================================================================

_VALID_STATES = {
    "BLIND_SELECT",
    "SELECTING_HAND",
    "ROUND_EVAL",
    "SHOP",
    "SMODS_BOOSTER_OPENED",
    "GAME_OVER",
}

_VALID_DECKS = {
    "RED",
    "BLUE",
    "YELLOW",
    "GREEN",
    "BLACK",
    "MAGIC",
    "NEBULA",
    "GHOST",
    "ABANDONED",
    "CHECKERED",
    "ZODIAC",
    "PAINTED",
    "ANAGLYPH",
    "PLASMA",
    "ERRATIC",
}

_VALID_STAKES = {"WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD"}

_TOP_LEVEL_KEYS = {
    "state",
    "round_num",
    "ante_num",
    "money",
    "deck",
    "stake",
    "seed",
    "won",
    "used_vouchers",
    "hands",
    "round",
    "blinds",
    "jokers",
    "consumables",
    "cards",
    "hand",
    "shop",
    "vouchers",
    "packs",
    "pack",
}

_AREA_KEYS = {"count", "limit", "highlighted_limit", "cards"}

_CARD_KEYS = {"id", "key", "set", "label", "value", "modifier", "state", "cost"}


def _start(sim: SimBackend, seed: str = "M15_ROUNDTRIP") -> dict:
    return sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})


def _assert_valid_gamestate(gs: dict) -> None:
    """Assert structural validity of a serialized gamestate."""
    assert set(gs.keys()) >= _TOP_LEVEL_KEYS
    assert gs["state"] in _VALID_STATES
    assert isinstance(gs["round_num"], int)
    assert isinstance(gs["ante_num"], int)
    assert isinstance(gs["money"], int)
    assert gs["deck"] in _VALID_DECKS
    assert gs["stake"] in _VALID_STAKES
    assert isinstance(gs["seed"], str) and gs["seed"]
    assert isinstance(gs["won"], bool)

    # Round
    r = gs["round"]
    _round_keys = (
        "hands_left",
        "hands_played",
        "discards_left",
        "discards_used",
        "reroll_cost",
        "chips",
    )
    for k in _round_keys:
        assert isinstance(r[k], int), f"round.{k} should be int"

    # Blinds
    blinds = gs["blinds"]
    for btype in ("small", "big", "boss"):
        b = blinds[btype]
        assert b["type"] == btype.upper()
        assert isinstance(b["status"], str)
        assert isinstance(b["name"], str)
        assert isinstance(b["score"], int)

    # Areas
    for area_key in ("jokers", "consumables", "cards", "hand", "shop", "vouchers", "packs", "pack"):
        area = gs[area_key]
        assert set(area.keys()) >= _AREA_KEYS, f"area '{area_key}' missing keys"
        assert isinstance(area["count"], int)
        assert isinstance(area["limit"], int)
        assert isinstance(area["cards"], list)
        assert area["count"] == len(area["cards"])

    # Hands
    assert isinstance(gs["hands"], dict)
    assert len(gs["hands"]) >= 9  # at least the 9 visible hands


def _assert_valid_card(card: dict) -> None:
    """Assert structural validity of a serialized card."""
    assert set(card.keys()) >= _CARD_KEYS
    assert isinstance(card["id"], int)
    assert isinstance(card["key"], str)
    assert isinstance(card["set"], str)
    assert isinstance(card["label"], str)

    v = card["value"]
    assert "suit" in v and "rank" in v and "effect" in v

    m = card["modifier"]
    for k in ("seal", "edition", "enhancement", "eternal", "perishable", "rental"):
        assert k in m, f"modifier missing '{k}'"
    assert isinstance(m["eternal"], bool)
    assert isinstance(m["rental"], bool)

    s = card["state"]
    for k in ("debuff", "hidden", "highlight"):
        assert k in s and isinstance(s[k], bool)

    c = card["cost"]
    assert isinstance(c["sell"], int)
    assert isinstance(c["buy"], int)


# ============================================================================
# 1. Serializer ↔ Deserializer roundtrip
# ============================================================================


class TestSerializerDeserializerRoundtrip:
    @pytest.mark.parametrize(
        "action",
        [
            PlayHand(card_indices=(0, 1, 2, 3, 4)),
            PlayHand(card_indices=(0,)),
            Discard(card_indices=(2, 3)),
            SelectBlind(),
            SkipBlind(),
            BuyCard(shop_index=0),
            BuyCard(shop_index=3),
            RedeemVoucher(card_index=0),
            OpenBooster(card_index=1),
            SellCard(area="jokers", card_index=0),
            SellCard(area="jokers", card_index=4),
            SellCard(area="consumables", card_index=0),
            SellCard(area="consumables", card_index=1),
            UseConsumable(card_index=0, target_indices=None),
            UseConsumable(card_index=0, target_indices=(1,)),
            UseConsumable(card_index=1, target_indices=(0, 2, 4)),
            Reroll(),
            NextRound(),
            CashOut(),
            PickPackCard(card_index=0),
            PickPackCard(card_index=2, target_indices=(0, 1)),
            SkipPack(),
        ],
        ids=lambda a: f"{type(a).__name__}_{hash(a)}",
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


# ============================================================================
# 2. Full run through backend
# ============================================================================


class TestFullRunThroughBackend:
    def test_three_antes_or_game_over(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_FULL_RUN")
        _assert_valid_gamestate(gs)
        assert gs["state"] == "BLIND_SELECT"

        max_steps = 200
        steps = 0

        while steps < max_steps:
            state = gs["state"]

            if state == "GAME_OVER":
                break

            if state == "BLIND_SELECT":
                gs = sim.handle("select", {})
                steps += 1

            elif state == "SELECTING_HAND":
                hand_count = gs["hand"]["count"]
                n = min(5, hand_count)
                if n == 0 or gs["round"]["hands_left"] <= 0:
                    break
                gs = sim.handle("play", {"cards": list(range(n))})
                steps += 1

            elif state == "ROUND_EVAL":
                gs = sim.handle("cash_out", {})
                steps += 1

            elif state == "SHOP":
                gs = sim.handle("next_round", {})
                steps += 1

            else:
                pytest.fail(f"Unexpected state: {state}")

            _assert_valid_gamestate(gs)

            # Ante 4 reached = 3 antes beaten, good enough
            if gs["ante_num"] >= 4:
                break

        # Should have progressed past ante 1
        assert gs["ante_num"] >= 2 or gs["state"] == "GAME_OVER"
        assert steps > 0

    def test_state_transitions_are_valid(self):
        """Verify each step produces a valid next state."""
        sim = SimBackend()
        gs = _start(sim, seed="M15_TRANSITIONS")

        valid_next = {
            "BLIND_SELECT": {"SELECTING_HAND"},
            "SELECTING_HAND": {"SELECTING_HAND", "ROUND_EVAL", "GAME_OVER"},
            "ROUND_EVAL": {"SHOP"},
            "SHOP": {"BLIND_SELECT", "SMODS_BOOSTER_OPENED"},
        }

        prev_state = gs["state"]
        for _ in range(50):
            state = gs["state"]
            if state == "GAME_OVER":
                break

            if state == "BLIND_SELECT":
                gs = sim.handle("select", {})
            elif state == "SELECTING_HAND":
                n = min(5, gs["hand"]["count"])
                if n == 0:
                    break
                gs = sim.handle("play", {"cards": list(range(n))})
            elif state == "ROUND_EVAL":
                gs = sim.handle("cash_out", {})
            elif state == "SHOP":
                gs = sim.handle("next_round", {})
            else:
                break

            new_state = gs["state"]
            allowed = valid_next.get(prev_state, set())
            assert new_state in allowed, (
                f"Invalid transition {prev_state} → {new_state} (expected one of {allowed})"
            )
            prev_state = new_state


# ============================================================================
# 3. Gamestate schema completeness at each phase
# ============================================================================


class TestGamestateSchemaCompleteness:
    def test_blind_select_phase(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_SCHEMA_1")
        assert gs["state"] == "BLIND_SELECT"
        _assert_valid_gamestate(gs)

    def test_selecting_hand_phase(self):
        sim = SimBackend()
        _start(sim, seed="M15_SCHEMA_2")
        gs = sim.handle("select", {})
        assert gs["state"] == "SELECTING_HAND"
        _assert_valid_gamestate(gs)
        assert gs["hand"]["count"] > 0
        assert gs["round"]["hands_left"] > 0

    def test_round_eval_phase(self):
        sim = SimBackend()
        _start(sim, seed="M15_SCHEMA_3")
        sim.handle("select", {})
        # Make blind trivially beatable
        sim._gs["blind"].chips = 1
        n = min(5, len(sim._gs["hand"]))
        gs = sim.handle("play", {"cards": list(range(n))})
        assert gs["state"] == "ROUND_EVAL"
        _assert_valid_gamestate(gs)

    def test_shop_phase(self):
        sim = SimBackend()
        _start(sim, seed="M15_SCHEMA_4")
        sim.handle("select", {})
        sim._gs["blind"].chips = 1
        n = min(5, len(sim._gs["hand"]))
        sim.handle("play", {"cards": list(range(n))})
        gs = sim.handle("cash_out", {})
        assert gs["state"] == "SHOP"
        _assert_valid_gamestate(gs)

    def test_hands_dict_structure(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_SCHEMA_5")
        hands = gs["hands"]
        assert len(hands) >= 9
        for name, h in hands.items():
            assert isinstance(name, str)
            assert isinstance(h["order"], int)
            assert isinstance(h["level"], int) and h["level"] >= 1
            assert isinstance(h["chips"], int) and h["chips"] > 0
            assert isinstance(h["mult"], int) and h["mult"] >= 1
            assert isinstance(h["played"], int)
            assert isinstance(h["played_this_round"], int)


# ============================================================================
# 4. Card schema in context
# ============================================================================


class TestCardSchemaInContext:
    def test_hand_card_has_full_schema(self):
        sim = SimBackend()
        _start(sim, seed="M15_CARD_SCHEMA")
        gs = sim.handle("select", {})
        assert gs["state"] == "SELECTING_HAND"
        assert gs["hand"]["count"] > 0

        card = gs["hand"]["cards"][0]
        _assert_valid_card(card)

        # Playing card should have suit/rank
        assert card["value"]["suit"] in ("H", "D", "C", "S")
        assert card["value"]["rank"] in (
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "T",
            "J",
            "Q",
            "K",
            "A",
        )

    def test_deck_cards_have_full_schema(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_CARD_SCHEMA_2")
        assert gs["cards"]["count"] > 0
        for card in gs["cards"]["cards"][:5]:
            _assert_valid_card(card)


# ============================================================================
# 5. Blind scores increase with ante
# ============================================================================


class TestBlindScoresIncreaseWithAnte:
    def test_boss_score_increases(self):
        sim = SimBackend()
        gs = _start(sim, seed="M15_SCALING")
        ante1_boss_score = gs["blinds"]["boss"]["score"]
        assert ante1_boss_score > 0

        # Play through 3 blinds of ante 1
        for _ in range(3):
            gs = sim.handle("select", {})
            sim._gs["blind"].chips = 1
            n = min(5, gs["hand"]["count"])
            gs = sim.handle("play", {"cards": list(range(n))})
            # May need multiple hands if somehow didn't beat
            for _ in range(5):
                if gs["state"] != "SELECTING_HAND":
                    break
                n = min(5, gs["hand"]["count"])
                if n == 0:
                    break
                gs = sim.handle("play", {"cards": list(range(n))})
            if gs["state"] == "GAME_OVER":
                pytest.skip("Game over before reaching ante 2")
            gs = sim.handle("cash_out", {})
            if gs["state"] == "SHOP":
                gs = sim.handle("next_round", {})

        assert gs["ante_num"] >= 2
        ante2_boss_score = gs["blinds"]["boss"]["score"]
        assert ante2_boss_score > ante1_boss_score


# ============================================================================
# 6. Money changes on actions
# ============================================================================


class TestMoneyChangesOnActions:
    def test_reroll_costs_money(self):
        sim = SimBackend()
        _start(sim, seed="M15_MONEY")

        # Get to shop
        sim.handle("select", {})
        sim._gs["blind"].chips = 1
        n = min(5, len(sim._gs["hand"]))
        sim.handle("play", {"cards": list(range(n))})
        gs = sim.handle("cash_out", {})
        assert gs["state"] == "SHOP"

        money_before = gs["money"]
        reroll_cost = gs["round"]["reroll_cost"]

        if money_before >= reroll_cost:
            gs = sim.handle("reroll", {})
            assert gs["money"] == money_before - reroll_cost
        else:
            # Can't afford reroll — verify it fails
            with pytest.raises(RPCError) as exc_info:
                sim.handle("reroll", {})
            assert exc_info.value.code == NOT_ALLOWED

    def test_money_after_cashout_increased(self):
        sim = SimBackend()
        _start(sim, seed="M15_MONEY_2")

        money_at_start = sim.handle("gamestate", None)["money"]

        sim.handle("select", {})
        sim._gs["blind"].chips = 1
        n = min(5, len(sim._gs["hand"]))
        sim.handle("play", {"cards": list(range(n))})
        gs = sim.handle("cash_out", {})

        # After beating a blind and cashing out, money should increase
        assert gs["money"] > money_at_start
