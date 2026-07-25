"""Tests for jackdaw.engine.economy and jackdaw.engine.round_lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.prototypes import VOUCHERS
from jackdaw.engine.economy import (
    calculate_discard_cost,
    calculate_round_earnings,
)
from jackdaw.engine.round_lifecycle import process_round_end_cards
from jackdaw.engine.vouchers import (
    apply_voucher,
    check_voucher_prerequisites,
    get_available_voucher_pool,
    get_next_voucher_key,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_blind() -> Blind:
    """Small Blind at ante 1: $3 reward."""
    return Blind.create("bl_small", ante=1)


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    c.sell_cost = 1
    return c


def _rental_joker() -> Card:
    """A rental joker (costs $3/round)."""
    j = _joker("j_spare_trousers")
    j.ability["rental"] = True
    return j


# ---------------------------------------------------------------------------
# calculate_discard_cost
# ---------------------------------------------------------------------------


class TestCalculateDiscardCost:
    def test_golden_needle_returns_one(self):
        gs = {"modifiers": {"discard_cost": 1}}
        assert calculate_discard_cost(gs) == 1


# ---------------------------------------------------------------------------
# Basic round — blind + hands + interest
# ---------------------------------------------------------------------------


class TestBasicRound:
    def test_comprehensive_earnings(self):
        """Beat Small Blind with 2 unused hands, $12 in bank.
        Expected: blind $3 + hands $2 (2x$1) + interest $2 (12//5=2) = $7.
        """
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=2,
            discards_left=0,
            money=12,
            jokers=[],
            game_state={},
        )
        assert result.blind_reward == 3
        assert result.unused_hands_bonus == 2
        assert result.interest == 2
        assert result.joker_dollars == 0
        assert result.rental_cost == 0
        assert result.total == 7


# ---------------------------------------------------------------------------
# Interest calculation
# ---------------------------------------------------------------------------


class TestInterest:
    def test_twelve_dollars_two_brackets(self):
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=12,
            jokers=[],
            game_state={},
        )
        assert result.interest == 2  # 12 // 5 = 2


# ---------------------------------------------------------------------------
# Rental joker costs
# ---------------------------------------------------------------------------


class TestRentalJokers:
    def test_one_rental_costs_three(self):
        j = _rental_joker()
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=0,
            discards_left=0,
            money=0,
            jokers=[j],
            game_state={},
        )
        assert result.rental_cost == 3
        assert result.total == 0  # blind(3) - rental(3)


# ---------------------------------------------------------------------------
# Green Deck — money per discard modifier
# ---------------------------------------------------------------------------


class TestGreenDeck:
    def test_green_deck_hand_and_discard_bonus(self):
        gs = {"modifiers": {"money_per_discard": 1}}
        result = calculate_round_earnings(
            blind=_small_blind(),
            hands_left=3,
            discards_left=2,
            money=0,
            jokers=[],
            game_state=gs,
        )
        assert result.unused_hands_bonus == 3
        assert result.unused_discards_bonus == 2


# ---------------------------------------------------------------------------
# To the Moon — increased interest_amount
# ---------------------------------------------------------------------------


# ===========================================================================
# Round lifecycle — perishable, rental, reset_round_targets
# ===========================================================================


def _lifecycle_joker(
    key: str = "j_joker",
    *,
    perishable: bool = False,
    perish_tally: int = 5,
    rental: bool = False,
    eternal: bool = False,
    debuff: bool = False,
) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker"}
    c.perishable = perishable
    c.perish_tally = perish_tally
    c.rental = rental
    c.eternal = eternal
    c.debuff = debuff
    if perishable:
        c.ability["perishable"] = True
        c.ability["perish_tally"] = perish_tally
    if rental:
        c.ability["rental"] = True
    return c


def _lifecycle_gs(**kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {"dollars": 20, "rental_rate": 3}
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Perishable countdown
# ---------------------------------------------------------------------------


class TestPerishableCountdown:
    def test_1_to_0_debuffed(self):
        j = _lifecycle_joker(perishable=True, perish_tally=1)
        result = process_round_end_cards([j], _lifecycle_gs())
        assert j.perish_tally == 0
        assert j.ability["perish_tally"] == 0
        assert j.debuff is True
        assert result.perished == [j]


# ---------------------------------------------------------------------------
# Rental charges
# ---------------------------------------------------------------------------


class TestRentalCharges:
    def test_rental_costs_3_dollars(self):
        j = _lifecycle_joker(rental=True)
        gs = _lifecycle_gs(dollars=20)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == 17
        assert result.rental_cost == 3
        assert result.rental_cards == [j]


# ---------------------------------------------------------------------------
# reset_round_targets
# ---------------------------------------------------------------------------


def _make_target_gs(seed: str = "TARGET_KNOWN") -> dict:
    from jackdaw.engine.run_init import initialize_run

    return initialize_run("b_red", 1, seed)


class TestResetRoundTargets:
    def test_idol_card_ante1(self):
        gs = _make_target_gs()
        idol = gs["current_round"]["idol_card"]
        assert idol == {"suit": "Clubs", "rank": "6"}


# ============================================================================
# Voucher effects (merged from test_vouchers.py)
# ============================================================================


def _voucher_gs(**kw) -> dict:
    """Minimal game_state with sensible defaults."""
    base = {
        "shop": {"joker_max": 2},
        "discount_percent": 0,
        "tarot_rate": 4,
        "planet_rate": 4,
        "playing_card_rate": 0,
        "edition_rate": 1,
        "interest_cap": 25,
        "consumable_slots": 2,
        "joker_slots": 5,
        "hand_size": 8,
        "round_resets": {
            "hands": 4,
            "discards": 3,
            "reroll_cost": 5,
            "ante": 1,
        },
        "current_round": {"reroll_cost": 5},
    }
    base.update(kw)
    return base


class _ControlledRng:
    """RNG stub that always picks the first element."""

    def seed(self, key: str) -> float:
        return 0.0

    def element(self, table: list, seed_val: float) -> tuple:
        return (table[0], 0)


class TestCheckVoucherPrerequisites:
    def test_voucher_with_no_requires_always_passes(self):
        assert check_voucher_prerequisites("v_overstock_norm", {}) is True

    def test_requires_satisfied_returns_true(self):
        assert check_voucher_prerequisites("v_overstock_plus", {"v_overstock_norm": True}) is True


class TestGetAvailableVoucherPool:
    def test_no_used_vouchers_returns_all_unlocked(self):
        pool = get_available_voucher_pool({})
        for key in pool:
            proto = VOUCHERS[key]
            for req in proto.requires:
                assert req not in pool or key == req, (
                    f"{key} in pool but its prereq {req} is not used"
                )

    def test_includes_upgraded_when_base_used(self):
        pool = get_available_voucher_pool({"v_overstock_norm": True})
        assert "v_overstock_plus" in pool


class TestGetNextVoucherKey:
    def test_returns_valid_voucher_key(self):
        rng = _ControlledRng()
        key = get_next_voucher_key(rng, {})
        assert key in VOUCHERS


class TestApplyVoucherShopModifiers:
    def test_overstock_increases_joker_slots(self):
        gs = _voucher_gs()
        apply_voucher("v_overstock_norm", gs)
        assert gs["shop"]["joker_max"] == 3


class TestApplyVoucherEconomy:
    def test_seed_money(self):
        gs = _voucher_gs()
        apply_voucher("v_seed_money", gs)
        assert gs["interest_cap"] == 50


class TestApplyVoucherHandsDiscards:
    def test_grabber_adds_hand(self):
        gs = _voucher_gs()
        apply_voucher("v_grabber", gs)
        assert gs["round_resets"]["hands"] == 5

    def test_wasteful_adds_discard(self):
        gs = _voucher_gs()
        apply_voucher("v_wasteful", gs)
        assert gs["round_resets"]["discards"] == 4


class TestApplyVoucherSlots:
    def test_antimatter_adds_joker_slot(self):
        gs = _voucher_gs()
        apply_voucher("v_antimatter", gs)
        assert gs["joker_slots"] == 6

    def test_crystal_ball_adds_consumable_slot(self):
        gs = _voucher_gs()
        apply_voucher("v_crystal_ball", gs)
        assert gs["consumable_slots"] == 3


class TestApplyVoucherAnteModifiers:
    def test_directors_cut(self):
        gs = _voucher_gs()
        apply_voucher("v_directors_cut", gs)
        assert gs["boss_blind_rerolls"] == 1
        assert gs["boss_blind_reroll_cost"] == 10


class TestApplyVoucherBossBlindRerolls:
    def test_retcon_sets_unlimited_free_rerolls(self):
        gs = _voucher_gs()
        apply_voucher("v_retcon", gs)
        assert gs["boss_blind_rerolls"] == -1
        assert gs["boss_blind_reroll_cost"] == 0


class TestApplyVoucherPassive:
    def test_hone_doubles_edition_rate(self):
        gs = _voucher_gs()
        apply_voucher("v_hone", gs)
        assert gs["edition_rate"] == 2
