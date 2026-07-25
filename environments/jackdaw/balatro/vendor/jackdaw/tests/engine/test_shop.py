"""Tests for jackdaw.engine.shop — type selection, rates, populate, buy/sell/reroll,
and integration scenarios."""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_area import CardArea
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.data.prototypes import BOOSTERS, CENTER_POOLS, PLANETS
from jackdaw.engine.packs import generate_pack_cards
from jackdaw.engine.pools import (
    UNAVAILABLE,
    check_soul_chance,
    get_current_pool,
    select_from_pool,
)
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.shop import (
    TYPE_JOKER,
    TYPE_PLAYING_CARD,
    buy_card,
    calculate_reroll_cost,
    get_pack,
    populate_shop,
    reroll_shop,
    roll_illusion_modifiers,
    select_shop_card_type,
    sell_card,
)
from jackdaw.engine.vouchers import get_next_voucher_key

_BOOSTER_KEYS: set[str] = set(CENTER_POOLS.get("Booster", []))


# ===========================================================================
# Helpers & fixtures
# ===========================================================================


def _joker(key: str = "j_joker", **kwargs) -> Card:
    card = create_joker(key, **kwargs)
    card.set_cost()
    return card


def _shop_area() -> CardArea:
    return CardArea(card_limit=10, area_type="shop")


def _joker_area(limit: int = 5) -> CardArea:
    return CardArea(card_limit=limit, area_type="joker")


def _base_gs(dollars: int = 10) -> dict:
    return {
        "current_round": {"voucher": "v_overstock_norm"},
        "used_jokers": {},
        "used_vouchers": {},
        "dollars": dollars,
        "jokers": [],
        "cards_purchased": 0,
    }


def _fresh_gs(
    *,
    seed: str = "TESTSEED",
    ante: int = 1,
    first_shop: bool = True,
    dollars: int = 100,
    has_showman: bool = False,
    tarot_rate: float = 4.0,
    planet_rate: float = 4.0,
    spectral_rate: float = 0.0,
    joker_rate: float = 20.0,
) -> tuple[PseudoRandom, dict[str, Any]]:
    rng = PseudoRandom(seed)
    voucher_key = get_next_voucher_key(rng, used_vouchers={}, in_shop=[])
    gs: dict[str, Any] = {
        "joker_rate": joker_rate,
        "tarot_rate": tarot_rate,
        "planet_rate": planet_rate,
        "spectral_rate": spectral_rate,
        "playing_card_rate": 0.0,
        "edition_rate": 1.0,
        "enable_eternals_in_shop": True,
        "enable_perishables_in_shop": True,
        "enable_rentals_in_shop": True,
        "banned_keys": set(),
        "used_jokers": {},
        "used_vouchers": {},
        "pool_flags": {},
        "has_showman": has_showman,
        "deck_enhancements": set(),
        "playing_card_count": 52,
        "played_hand_types": set(),
        "shop_vouchers": set(),
        "inflation": 0,
        "discount_percent": 0,
        "booster_ante_scaling": False,
        "has_astronomer": False,
        "shop": {"joker_max": 2},
        "current_round": {"voucher": voucher_key},
        "first_shop_buffoon": not first_shop,
        "dollars": dollars,
        "jokers": [],
    }
    return rng, gs


# ===========================================================================
# Section 1: Type selection & rates
# ===========================================================================


class TestSelectShopCardTypeKnownSeeds:
    def test_joker_known_seed(self):
        assert select_shop_card_type(PseudoRandom("SHOP_TYPE_TEST"), 1) == TYPE_JOKER

    def test_deterministic_same_seed(self):
        rng1 = PseudoRandom("DET_SHOP")
        rng2 = PseudoRandom("DET_SHOP")
        assert select_shop_card_type(rng1, 3) == select_shop_card_type(rng2, 3)


class TestModifiedRates:
    def test_magic_trick_enables_playing_card(self):
        assert (
            select_shop_card_type(PseudoRandom("SHOP_B"), 1, playing_card_rate=4)
            == TYPE_PLAYING_CARD
        )


class TestRollIllusionModifiers:
    def test_known_seed_with_enhancement(self):
        result = roll_illusion_modifiers(PseudoRandom("ILLUS_A"), 1)
        assert "enhancement" in result
        assert result["enhancement"] == "m_stone"

    def test_deterministic_same_seed(self):
        r1 = roll_illusion_modifiers(PseudoRandom("DET_ILLUS"), 1)
        r2 = roll_illusion_modifiers(PseudoRandom("DET_ILLUS"), 1)
        assert r1 == r2


class TestGetPack:
    def test_first_shop_returns_buffoon_normal_1(self):
        assert get_pack(PseudoRandom("GP_FIRST"), 1, first_shop=True) == "p_buffoon_normal_1"

    def test_known_seed_arcana_normal_2(self):
        assert get_pack(PseudoRandom("TESTTEST"), 1) == "p_arcana_normal_2"

    def test_deterministic(self):
        assert get_pack(PseudoRandom("GP_DET"), 1) == get_pack(PseudoRandom("GP_DET"), 1)

    def test_banned_pack_never_selected(self):
        banned = {"p_arcana_normal_1"}
        for i in range(200):
            assert (
                get_pack(PseudoRandom(f"GP_BAN_{i}"), 1, banned_keys=banned) != "p_arcana_normal_1"
            )


# ===========================================================================
# Section 2: Populate shop
# ===========================================================================


class TestPopulateShopStructure:
    def test_returns_dict_with_three_keys(self):
        result = populate_shop(PseudoRandom("PS_STRUCT"), 1, _base_gs())
        assert set(result.keys()) == {"jokers", "voucher", "boosters"}

    def test_default_two_joker_slots(self):
        result = populate_shop(PseudoRandom("PS_DEF"), 1, _base_gs())
        assert len(result["jokers"]) == 2

    def test_always_exactly_two_boosters(self):
        result = populate_shop(PseudoRandom("PS_BOOST"), 1, _base_gs())
        assert len(result["boosters"]) == 2

    def test_first_booster_is_buffoon_on_first_shop(self):
        gs = _base_gs()
        result = populate_shop(PseudoRandom("PS_FBUFF"), 1, gs)
        first_pack = result["boosters"][0]
        assert BOOSTERS[first_pack.center_key].kind == "Buffoon"


class TestPopulateShopDeterminism:
    def test_same_seed_same_result(self):
        gs1 = _base_gs()
        gs2 = _base_gs()
        r1 = populate_shop(PseudoRandom("PS_DET"), 1, gs1)
        r2 = populate_shop(PseudoRandom("PS_DET"), 1, gs2)
        assert [c.center_key for c in r1["jokers"]] == [c.center_key for c in r2["jokers"]]
        assert [c.center_key for c in r1["boosters"]] == [c.center_key for c in r2["boosters"]]


# ===========================================================================
# Section 3: Buy / Sell / Reroll actions
# ===========================================================================


class TestCalculateRerollCost:
    def test_default_cost_is_five(self):
        gs = {"current_round": {}, "round_resets": {"reroll_cost": 5}}
        assert calculate_reroll_cost(gs) == 5

    def test_free_rerolls_gives_zero_cost(self):
        gs = {
            "current_round": {"free_rerolls": 1},
            "round_resets": {"reroll_cost": 5},
        }
        assert calculate_reroll_cost(gs) == 0

    def test_increase_adds_to_base(self):
        gs = {
            "current_round": {"reroll_cost_increase": 2},
            "round_resets": {"reroll_cost": 5},
        }
        assert calculate_reroll_cost(gs) == 7


class TestBuyCard:
    def test_successful_purchase_deducts_dollars(self):
        card = _joker()
        card.cost = 5
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is True
        assert gs["dollars"] == 5

    def test_insufficient_funds_rejected(self):
        card = _joker()
        card.cost = 15
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area()
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is False
        assert result["reason"] == "insufficient_funds"

    def test_negative_edition_grants_bonus_space(self):
        card = _joker()
        card.cost = 3
        card.set_edition({"negative": True})
        shop = _shop_area()
        shop.add(card)
        dest = _joker_area(limit=1)
        filler = _joker()
        dest.add(filler)
        gs = _base_gs(dollars=10)
        result = buy_card(card, shop, dest, gs)
        assert result["ok"] is True


class TestSellCard:
    def test_successful_sale_awards_dollars(self):
        card = _joker()
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["ok"] is True
        assert gs["dollars"] == card.sell_cost

    def test_eternal_joker_rejected(self):
        card = _joker(eternal=True)
        card.set_cost()
        area = _joker_area()
        area.add(card)
        gs = _base_gs(dollars=0)
        result = sell_card(card, area, gs)
        assert result["ok"] is False
        assert result["reason"] == "eternal"


class TestRerollShop:
    def _make_shop(self, n_cards: int = 2) -> CardArea:
        area = CardArea(card_limit=10, area_type="shop")
        for key in list(("j_joker", "j_greedy_joker", "j_lusty_joker"))[:n_cards]:
            c = _joker(key)
            area.add(c)
        return area

    def test_cost_deducted(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_COST"), 1, gs)
        assert result["ok"] is True
        assert result["cost"] == 5
        assert gs["dollars"] == 5

    def test_new_cards_generated(self):
        shop = self._make_shop(2)
        gs = _base_gs(dollars=10)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_NEW"), 1, gs)
        assert len(result["new_cards"]) == 2
        assert len(shop.cards) == 2

    def test_insufficient_funds_rejected(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=3)
        gs.update(
            {"current_round": {"reroll_cost_increase": 0}, "round_resets": {"reroll_cost": 5}}
        )
        result = reroll_shop(shop, PseudoRandom("RR_POOR"), 1, gs)
        assert result["ok"] is False
        assert result["reason"] == "insufficient_funds"

    def test_free_reroll_zero_cost(self):
        shop = self._make_shop()
        gs = _base_gs(dollars=0)
        gs.update(
            {
                "current_round": {"free_rerolls": 1, "reroll_cost_increase": 0},
                "round_resets": {"reroll_cost": 5},
            }
        )
        result = reroll_shop(shop, PseudoRandom("RR_FREE"), 1, gs)
        assert result["ok"] is True
        assert result["cost"] == 0
        assert result["was_free"] is True


# ===========================================================================
# Section 4: Integration scenarios
# ===========================================================================


class TestFullShopTESTSEED:
    def test_shop_returns_two_joker_slots(self):
        rng, gs = _fresh_gs(seed="TESTSEED", ante=1)
        result = populate_shop(rng, 1, gs)
        assert len(result["jokers"]) == 2


class TestShowmanAllowsDuplicates:
    def test_showman_restores_full_pool(self):
        rng_base = PseudoRandom("SHOWMAN_TEST2")
        pool_with_showman, _ = get_current_pool(
            "Joker",
            rng_base,
            1,
            rarity=1,
            append="sho",
            used_jokers={"j_joker"},
            has_showman=True,
        )
        assert "j_joker" in pool_with_showman
        idx = pool_with_showman.index("j_joker")
        assert pool_with_showman[idx] != UNAVAILABLE


class TestFullCycle:
    def test_full_shop_cycle(self):
        rng, gs = _fresh_gs(seed="FULL_CYCLE", ante=1, dollars=200)
        gs["jokers"] = []

        # Step 1: populate shop
        result = populate_shop(rng, 1, gs)
        assert len(result["jokers"]) == 2

        # Step 2: buy the first joker
        joker_cards = [c for c in result["jokers"] if c.ability.get("set") == "Joker"]
        if not joker_cards:
            pytest.skip("No Joker-type card in first shop slot")

        shop_area = CardArea(card_limit=2, area_type="shop")
        for c in result["jokers"]:
            shop_area.add(c)
        joker_area = CardArea(card_limit=5, area_type="joker")

        first_joker = joker_cards[0]
        initial_dollars = gs["dollars"]
        buy_result = buy_card(first_joker, shop_area, joker_area, gs)
        assert buy_result["ok"] is True
        assert gs["dollars"] == initial_dollars - first_joker.cost
        assert first_joker.center_key in gs["used_jokers"]

        # Step 3: reroll
        dollars_before_reroll = gs["dollars"]
        reroll_result = reroll_shop(shop_area, rng, 1, gs)
        assert reroll_result["ok"] is True
        assert reroll_result["cost"] == 5
        assert gs["dollars"] == dollars_before_reroll - 5

        # Step 4: second reroll costs more
        r2 = reroll_shop(shop_area, rng, 1, gs)
        assert r2["ok"] is True
        assert r2["cost"] == 6


# ============================================================================
# Pack generation (merged from test_packs.py)
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_packs():
    reset_sort_id_counter()


def _pack_rng(seed: str = "TESTPACK") -> PseudoRandom:
    return PseudoRandom(seed)


class TestPackTypes:
    def test_arcana_generates_tarots(self):
        cards, _ = generate_pack_cards("p_arcana_normal_1", _pack_rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Tarot"

    def test_celestial_generates_planets(self):
        cards, _ = generate_pack_cards("p_celestial_normal_1", _pack_rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Planet"

    def test_spectral_generates_spectrals(self):
        cards, _ = generate_pack_cards("p_spectral_normal_1", _pack_rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") in ("Spectral", "Joker", "Planet")
            assert card.center_key not in ("", None)

    def test_standard_generates_playing_cards(self):
        cards, _ = generate_pack_cards("p_standard_normal_1", _pack_rng(), 1, {})
        for card in cards:
            assert card.base is not None

    def test_buffoon_generates_jokers(self):
        cards, _ = generate_pack_cards("p_buffoon_normal_1", _pack_rng(), 1, {})
        for card in cards:
            assert card.ability.get("set") == "Joker"


class TestTelescope:
    def test_telescope_forces_first_card_to_most_played_hand_planet(self):
        hand_type = "Flush"
        planet_key = None
        for k, v in PLANETS.items():
            if v.config.get("hand_type") == hand_type:
                planet_key = k
                break
        assert planet_key is not None

        cards, _ = generate_pack_cards(
            "p_celestial_normal_1",
            _pack_rng(),
            1,
            {"has_telescope": True, "most_played_hand": hand_type},
        )
        assert cards[0].center_key == planet_key


class TestOmenGlobe:
    def test_omen_globe_can_produce_spectral(self):
        found_spectral = False
        for i in range(30):
            cards, _ = generate_pack_cards(
                "p_arcana_normal_1",
                PseudoRandom(f"OG{i}"),
                1,
                {"has_omen_globe": True},
            )
            if any(c.ability.get("set") == "Spectral" for c in cards):
                found_spectral = True
                break
        assert found_spectral, "Omen Globe should occasionally produce Spectral cards"


class TestChoose:
    def test_choose_matches_proto(self):
        for pack_key in [
            "p_arcana_normal_1",
            "p_celestial_mega_1",
            "p_buffoon_jumbo_1",
            "p_standard_normal_1",
        ]:
            _, choose = generate_pack_cards(pack_key, _pack_rng(), 1, {})
            assert choose == BOOSTERS[pack_key].config["choose"]

    def test_mega_pack_choose_is_greater_than_1(self):
        _, choose = generate_pack_cards("p_arcana_mega_1", _pack_rng(), 1, {})
        assert choose > 1


# ============================================================================
# Pool mechanics (merged from test_pools.py)
# ============================================================================


class _PoolRNG:
    """Returns pre-scripted float values from random(), ignores key."""

    def __init__(self, values: list[float]) -> None:
        self._it = iter(values)

    def random(self, key: str) -> float:  # noqa: ARG002
        return next(self._it)


class TestJokerPool:
    """Tests for pool_type='Joker'."""

    def test_total_jokers_all_rarities(self):
        """Unfiltered pool across all rarities sums to 150."""
        total = 0
        rng = _PoolRNG([])
        for rar in (1, 2, 3, 4):
            pool, _ = get_current_pool("Joker", rng, ante=1, rarity=rar)
            total += len(pool)
        assert total == 150


class TestJokerBannedKeys:
    def test_banned_key_becomes_unavailable(self):
        pool, _ = get_current_pool("Joker", _PoolRNG([]), ante=1, rarity=1, banned_keys={"j_joker"})
        assert "j_joker" not in pool
        assert UNAVAILABLE in pool
        assert len(pool) == 61  # length preserved


class TestJokerEnhancementGate:
    """Enhancement gate jokers require their enhancement to be in deck_enhancements."""

    def test_lucky_cat_included_with_lucky(self):
        pool, _ = get_current_pool(
            "Joker",
            _PoolRNG([]),
            ante=1,
            rarity=2,
            deck_enhancements={"m_lucky"},
        )
        assert "j_lucky_cat" in pool


class TestVoucherPool:
    def test_prerequisite_unlocks_dependent_voucher(self):
        # v_liquidation requires v_clearance_sale
        pool_without, _ = get_current_pool("Voucher", _PoolRNG([]), ante=1)
        pool_with, _ = get_current_pool(
            "Voucher", _PoolRNG([]), ante=1, used_vouchers={"v_clearance_sale"}
        )
        assert "v_liquidation" not in pool_without
        assert "v_liquidation" in pool_with


class TestSelectFromPool:
    """Tests for the deterministic pool-selection step."""

    def test_known_seed_returns_specific_joker(self):
        """Fixed seed 'TEST_SELECT' + rarity=1 picks a deterministic joker."""
        rng = PseudoRandom("TEST_SELECT")
        pool, pool_key = get_current_pool("Joker", rng, ante=1, rarity=1)
        result = select_from_pool(pool, rng, pool_key, ante=1)
        assert result == "j_swashbuckler"


class _ScriptedRNG:
    """Returns a fixed float for every random() call."""

    def __init__(self, *values: float) -> None:
        self._it = iter(values)

    def random(self, _: str) -> float:
        return next(self._it)


class TestCheckSoulChanceThreshold:
    """Boundary tests using a scripted RNG stub."""

    def test_joker_hit_at_boundary(self):
        assert check_soul_chance("Joker", _ScriptedRNG(0.9971), 1) == "c_soul"
