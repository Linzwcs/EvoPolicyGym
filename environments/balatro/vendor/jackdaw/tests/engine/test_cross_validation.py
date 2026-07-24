"""Cross-validation test suite — verify simulator against oracle fixtures.

Each fixture captures exact values from the verified simulator output
(proven bit-exact against live Balatro via balatrobot for run init,
deck order, dealt hands, money, and scores).

Fixture files in tests/fixtures/cross_validation/:
  - run_init.json: 5 deck/stake combos
  - scoring.json: 10 hand types
  - shop.json: 9 shop snapshots (3 seeds × 3 antes)
  - economy.json: 5 interest/earnings calculations
  - hand_levels.json: 6 level-up scenarios
  - targeting.json: 9 targeting card values
  - consumables.json: 3 consumable effects

Target: 100+ assertions, all passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.consumables import ConsumableContext, use_consumable
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.economy import calculate_round_earnings
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand
from jackdaw.engine.shop import populate_shop
from jackdaw.engine.vouchers import get_next_voucher_key

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cross_validation"


def _load(name: str) -> list[dict]:
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)["cases"]


# ---------------------------------------------------------------------------
# Run initialization
# ---------------------------------------------------------------------------


class TestRunInitCrossValidation:
    CASES = _load("run_init.json")

    @pytest.mark.parametrize("case", CASES, ids=[c["back"] + "_s" + str(c["stake"]) for c in CASES])
    def test_dollars(self, case):
        gs = initialize_run(case["back"], case["stake"], case["seed"])
        assert gs["dollars"] == case["dollars"]

    @pytest.mark.parametrize("case", CASES, ids=[c["back"] + "_s" + str(c["stake"]) for c in CASES])
    def test_starting_params(self, case):
        gs = initialize_run(case["back"], case["stake"], case["seed"])
        sp = gs["starting_params"]
        assert sp["hands"] == case["hands"]
        assert sp["discards"] == case["discards"]

    @pytest.mark.parametrize("case", CASES, ids=[c["back"] + "_s" + str(c["stake"]) for c in CASES])
    def test_boss(self, case):
        gs = initialize_run(case["back"], case["stake"], case["seed"])
        assert gs["round_resets"]["blind_choices"]["Boss"] == case["boss"]

    @pytest.mark.parametrize("case", CASES, ids=[c["back"] + "_s" + str(c["stake"]) for c in CASES])
    def test_tags(self, case):
        gs = initialize_run(case["back"], case["stake"], case["seed"])
        tags = gs["round_resets"]["blind_tags"]
        assert tags["Small"] == case["small_tag"]
        assert tags["Big"] == case["big_tag"]

    @pytest.mark.parametrize("case", CASES, ids=[c["back"] + "_s" + str(c["stake"]) for c in CASES])
    def test_deck_order(self, case):
        gs = initialize_run(case["back"], case["stake"], case["seed"])
        assert len(gs["deck"]) == case["deck_size"]
        first5 = [c.card_key for c in gs["deck"][:5]]
        last5 = [c.card_key for c in gs["deck"][-5:]]
        assert first5 == case["deck_first5"]
        assert last5 == case["deck_last5"]

    @pytest.mark.parametrize("case", CASES, ids=[c["back"] + "_s" + str(c["stake"]) for c in CASES])
    def test_voucher(self, case):
        gs = initialize_run(case["back"], case["stake"], case["seed"])
        assert gs["current_round"].get("voucher") == case["voucher"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoringCrossValidation:
    CASES = _load("scoring.json")

    @staticmethod
    def _make_cards(*specs):
        return [create_playing_card(Suit(s), Rank(r)) for s, r in specs]

    # Hand definitions matching the fixture
    _HANDS = {
        "pair_aces": [
            ("Spades", "Ace"),
            ("Hearts", "Ace"),
            ("Clubs", "King"),
            ("Diamonds", "5"),
            ("Hearts", "3"),
        ],
        "flush_hearts": [
            ("Hearts", "Ace"),
            ("Hearts", "King"),
            ("Hearts", "10"),
            ("Hearts", "5"),
            ("Hearts", "3"),
        ],
        "straight_5_9": [
            ("Spades", "9"),
            ("Hearts", "8"),
            ("Clubs", "7"),
            ("Diamonds", "6"),
            ("Hearts", "5"),
        ],
        "full_house_kings_sevens": [
            ("Spades", "King"),
            ("Hearts", "King"),
            ("Clubs", "King"),
            ("Diamonds", "7"),
            ("Hearts", "7"),
        ],
        "high_card_ace": [
            ("Spades", "Ace"),
            ("Hearts", "Queen"),
            ("Clubs", "9"),
            ("Diamonds", "4"),
            ("Hearts", "2"),
        ],
        "two_pair_jacks_fives": [
            ("Spades", "Jack"),
            ("Hearts", "Jack"),
            ("Clubs", "5"),
            ("Diamonds", "5"),
            ("Hearts", "2"),
        ],
        "three_tens": [
            ("Spades", "10"),
            ("Hearts", "10"),
            ("Clubs", "10"),
            ("Diamonds", "6"),
            ("Hearts", "3"),
        ],
        "four_eights": [
            ("Spades", "8"),
            ("Hearts", "8"),
            ("Clubs", "8"),
            ("Diamonds", "8"),
            ("Hearts", "2"),
        ],
        "straight_flush_spades": [
            ("Spades", "Queen"),
            ("Spades", "Jack"),
            ("Spades", "10"),
            ("Spades", "9"),
            ("Spades", "8"),
        ],
        "single_ace": [("Diamonds", "Ace")],
    }

    @pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
    def test_hand_type(self, case):
        specs = self._HANDS[case["name"]]
        played = self._make_cards(*specs)
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("SCORING")
        r = score_hand(played, [], [], hl, blind, rng)
        assert r.hand_type == case["hand_type"]

    @pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
    def test_total(self, case):
        specs = self._HANDS[case["name"]]
        played = self._make_cards(*specs)
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("SCORING")
        r = score_hand(played, [], [], hl, blind, rng)
        assert r.total == case["total"]


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------


class TestShopCrossValidation:
    CASES = _load("shop.json")

    @pytest.mark.parametrize("case", CASES, ids=[f"{c['seed']}_a{c['ante']}" for c in CASES])
    def test_voucher(self, case):
        rng = PseudoRandom(case["seed"])
        vk = get_next_voucher_key(rng, {}, [], ante=case["ante"])
        assert vk == case["voucher"]

    @pytest.mark.parametrize("case", CASES, ids=[f"{c['seed']}_a{c['ante']}" for c in CASES])
    def test_joker_count(self, case):
        rng = PseudoRandom(case["seed"])
        vk = get_next_voucher_key(rng, {}, [], ante=case["ante"])
        gs = {
            "joker_rate": 20.0,
            "tarot_rate": 4.0,
            "planet_rate": 4.0,
            "spectral_rate": 0.0,
            "playing_card_rate": 0.0,
            "edition_rate": 1.0,
            "enable_eternals_in_shop": True,
            "enable_perishables_in_shop": True,
            "enable_rentals_in_shop": True,
            "banned_keys": set(),
            "used_jokers": {},
            "used_vouchers": {},
            "pool_flags": {},
            "has_showman": False,
            "deck_enhancements": set(),
            "playing_card_count": 52,
            "played_hand_types": set(),
            "shop_vouchers": set(),
            "inflation": 0,
            "discount_percent": 0,
            "booster_ante_scaling": False,
            "has_astronomer": False,
            "shop": {"joker_max": 2},
            "current_round": {"voucher": vk},
            "first_shop_buffoon": case["ante"] > 1,
        }
        result = populate_shop(rng, case["ante"], gs)
        assert len(result["jokers"]) == len(case["jokers"])

    @pytest.mark.parametrize("case", CASES, ids=[f"{c['seed']}_a{c['ante']}" for c in CASES])
    def test_joker_keys(self, case):
        rng = PseudoRandom(case["seed"])
        vk = get_next_voucher_key(rng, {}, [], ante=case["ante"])
        gs = {
            "joker_rate": 20.0,
            "tarot_rate": 4.0,
            "planet_rate": 4.0,
            "spectral_rate": 0.0,
            "playing_card_rate": 0.0,
            "edition_rate": 1.0,
            "enable_eternals_in_shop": True,
            "enable_perishables_in_shop": True,
            "enable_rentals_in_shop": True,
            "banned_keys": set(),
            "used_jokers": {},
            "used_vouchers": {},
            "pool_flags": {},
            "has_showman": False,
            "deck_enhancements": set(),
            "playing_card_count": 52,
            "played_hand_types": set(),
            "shop_vouchers": set(),
            "inflation": 0,
            "discount_percent": 0,
            "booster_ante_scaling": False,
            "has_astronomer": False,
            "shop": {"joker_max": 2},
            "current_round": {"voucher": vk},
            "first_shop_buffoon": case["ante"] > 1,
        }
        result = populate_shop(rng, case["ante"], gs)
        sim_keys = [c.center_key for c in result["jokers"]]
        fix_keys = [j["key"] for j in case["jokers"]]
        assert sim_keys == fix_keys


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------


class TestEconomyCrossValidation:
    CASES = _load("economy.json")

    @pytest.mark.parametrize("case", CASES, ids=[f"money_{c['money']}" for c in CASES])
    def test_earnings(self, case):
        blind = Blind.create("bl_small", ante=1)
        earnings = calculate_round_earnings(
            blind=blind,
            hands_left=case["hands_left"],
            discards_left=case["discards_left"],
            money=case["money"],
            jokers=[],
            game_state={},
        )
        assert earnings.blind_reward == case["blind_reward"]
        assert earnings.unused_hands_bonus == case["unused_hands"]
        assert earnings.interest == case["interest"]
        assert earnings.total == case["total"]


# ---------------------------------------------------------------------------
# Hand levels
# ---------------------------------------------------------------------------


class TestHandLevelsCrossValidation:
    CASES = _load("hand_levels.json")

    @pytest.mark.parametrize("case", CASES, ids=[f"{c['hand_type']}_L{c['level']}" for c in CASES])
    def test_values(self, case):
        from jackdaw.engine.data.hands import HandType

        hl = HandLevels()
        ht = HandType(case["hand_type"])
        if case["level"] > 1:
            hl.level_up(ht, case["level"] - 1)
        chips, mult = hl.get(ht)
        assert chips == case["chips"]
        assert mult == case["mult"]


# ---------------------------------------------------------------------------
# Consumables
# ---------------------------------------------------------------------------


class TestConsumablesCrossValidation:
    CASES = _load("consumables.json")

    def test_chariot_steel(self):
        case = self.CASES[0]
        assert case["consumable"] == "c_chariot"
        target = create_playing_card(Suit.SPADES, Rank.ACE)
        chariot = Card(center_key="c_chariot")
        chariot.ability = {"set": "Tarot", "effect": ""}
        ctx = ConsumableContext(card=chariot, highlighted=[target])
        result = use_consumable(chariot, ctx)
        assert result is not None
        assert result.enhance is not None
        for c, enh in result.enhance:
            c.set_ability(enh)
        assert (
            target.ability.get("name") == case["effect_name"]
            or target.ability.get("effect") == case["effect"]
        )

    def test_strength_rank_change(self):
        case = self.CASES[1]
        assert case["consumable"] == "c_strength"
        target = create_playing_card(Suit.HEARTS, Rank.KING)
        strength = Card(center_key="c_strength")
        strength.ability = {"set": "Tarot", "effect": ""}
        ctx = ConsumableContext(card=strength, highlighted=[target])
        result = use_consumable(strength, ctx)
        assert result is not None
        assert result.change_rank is not None
        assert result.change_rank[0][1] == case["rank_change"]

    def test_mercury_level_up(self):
        case = self.CASES[2]
        assert case["consumable"] == "c_mercury"
        mercury = Card(center_key="c_mercury")
        mercury.ability = {"set": "Planet", "effect": "", "consumeable": {"hand_type": "Pair"}}
        ctx = ConsumableContext(card=mercury, game_state={})
        result = use_consumable(mercury, ctx)
        assert result is not None
        # JSON serializes tuples as lists
        assert [list(x) for x in result.level_up] == case["level_up"]
        assert result.notify_jokers_consumeable == case["notify"]
