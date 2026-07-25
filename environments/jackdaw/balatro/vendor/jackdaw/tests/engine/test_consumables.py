"""Tests for the consumable dispatch system.

Validates the registry, can_use validation, dispatch mechanism,
and representative consumable effects with integration scoring chains.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.consumables import (
    _CONSUMABLE_REGISTRY,
    ConsumableContext,
    ConsumableResult,
    can_use_consumable,
    register_consumable,
    use_consumable,
)
from jackdaw.engine.economy import calculate_round_earnings
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand, score_hand_base
from jackdaw.engine.vouchers import apply_voucher


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
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


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    c = Card()
    c.set_base(f"{_SL[suit]}_{_RL[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _consumable(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Tarot", **ability_kw}
    return c


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


# --- Integration helpers (from consumables_integration) ---


def _c(suit_letter: str, rank_letter: str, enh: str = "c_base") -> Card:
    suits = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
    ranks = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "T": "10",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }
    suit = suits[suit_letter]
    rank = ranks[rank_letter]
    c = Card()
    c.set_base(f"{suit_letter}_{rank_letter}", suit, rank)
    c.set_ability(enh)
    return c


def _integ_consumable(key: str) -> Card:
    c = Card()
    c.set_ability(key)
    return c


def _integ_joker(key: str, **kw) -> Card:
    j = Card()
    j.set_ability(key)
    j.ability.update(kw)
    return j


def _sb() -> Blind:
    return Blind.create("bl_small", ante=1)


def _bb() -> Blind:
    return Blind.create("bl_big", ante=1)


def _rng(seed: str = "INTEG") -> PseudoRandom:
    return PseudoRandom(seed)


def _apply_result(result: ConsumableResult, hand_levels: HandLevels) -> None:
    for hand_type, amount in result.level_up or []:
        hand_levels.level_up(hand_type, amount)


def _apply_copy_card(source: Card, target: Card) -> None:
    if source.base is not None and target.base is not None:
        target.set_base(source.card_key or "", source.base.suit.value, source.base.rank.value)
    target.enhance(source.center_key)
    target.set_edition(source.edition)
    target.set_seal(source.seal)


# ============================================================================
# Test consumable handlers (registered for testing, cleaned up after)
# ============================================================================

_TEST_KEY = "c_test_tarot"
_TEST_KEY_2 = "c_test_spectral"


@pytest.fixture(autouse=True)
def _register_test_consumables():
    """Register test consumables before each test, clean up after."""

    @register_consumable(_TEST_KEY)
    def _test_handler(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        return ConsumableResult(
            enhance=[(h, "m_gold") for h in (ctx.highlighted or [])],
        )

    @register_consumable(_TEST_KEY_2)
    def _test_spectral(card: Card, ctx: ConsumableContext) -> ConsumableResult:
        return ConsumableResult(dollars=20, destroy=ctx.highlighted)

    yield

    _CONSUMABLE_REGISTRY.pop(_TEST_KEY, None)
    _CONSUMABLE_REGISTRY.pop(_TEST_KEY_2, None)


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    def test_register_adds(self):
        assert _TEST_KEY in _CONSUMABLE_REGISTRY


# ============================================================================
# Dispatch
# ============================================================================


class TestDispatch:
    def test_dispatch_returns_result(self):
        c = _consumable(_TEST_KEY)
        highlighted = [_card("Hearts", "5")]
        ctx = ConsumableContext(card=c, highlighted=highlighted)
        result = use_consumable(c, ctx)
        assert result is not None
        assert result.enhance is not None
        assert len(result.enhance) == 1
        assert result.enhance[0][1] == "m_gold"


# ============================================================================
# can_use_consumable — global blockers
# ============================================================================


class TestCanUseGlobalBlockers:
    def test_cards_in_play_blocks(self):
        c = _consumable("c_hermit")
        c.ability["consumeable"] = {}
        assert can_use_consumable(c, cards_in_play=1) is False


# ============================================================================
# can_use_consumable — one test per selection category
# ============================================================================


class TestCanUseSelectionCategories:
    def test_planet_zero_cards(self):
        """Planets require 0 highlighted cards."""
        c = _consumable("c_mercury")
        c.ability["consumeable"] = {"hand_type": "Pair"}
        assert can_use_consumable(c) is True

    def test_chariot_exactly_1_card(self):
        """Chariot: max_highlighted=1, needs exactly 1."""
        c = _consumable("c_chariot")
        c.ability["consumeable"] = {"max_highlighted": 1, "mod_conv": "m_steel"}
        assert can_use_consumable(c, highlighted=[]) is False
        assert can_use_consumable(c, highlighted=[_card("Hearts", "5")]) is True
        assert (
            can_use_consumable(c, highlighted=[_card("Hearts", "5"), _card("Spades", "3")]) is False
        )


# ============================================================================
# Enhancement tarots — one representative (Chariot → Steel)
# ============================================================================


class TestChariot:
    def test_chariot_enhances_to_steel(self):
        c = _consumable("c_chariot")
        ace = _card("Spades", "Ace")
        ctx = ConsumableContext(card=c, highlighted=[ace])
        result = use_consumable(c, ctx)
        assert result is not None
        assert len(result.enhance) == 1
        assert result.enhance[0] == (ace, "m_steel")


# ============================================================================
# Planet cards — Mercury + Black Hole
# ============================================================================


def _planet(key: str) -> Card:
    c = Card()
    c.set_ability(key)
    return c


class TestMercury:
    def test_mercury_levels_pair(self):
        c = _planet("c_mercury")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.level_up == [("Pair", 1)]


class TestBlackHole:
    def test_levels_all_12_hands(self):
        c = _planet("c_black_hole")
        result = use_consumable(c, ConsumableContext(card=c))
        assert result is not None
        assert result.level_up is not None
        assert len(result.level_up) == 12


# ============================================================================
# Spectral cards with interesting logic
# ============================================================================


class _ControlledRng:
    """Minimal RNG stub with scripted return values for random() and seed()."""

    def __init__(self, random_values: list[float] | None = None):
        self._vals = iter(random_values or [])
        self._seed_counter = 0.5

    def random(self, key: str) -> float:
        return next(self._vals)

    def seed(self, key: str) -> float:
        return self._seed_counter

    def element(self, table: list, seed_val: float) -> tuple:
        return (table[0], 0)

    def shuffle(self, _lst: list, _seed_val: float) -> None:
        pass


class TestDeath:
    def test_copy_right_to_left(self):
        c = _consumable("c_death")
        left = _card("Hearts", "5")
        right = _card("Spades", "Ace")
        result = use_consumable(
            c,
            ConsumableContext(card=c, highlighted=[left, right]),
        )
        assert result is not None
        assert result.copy_card is not None
        source, target = result.copy_card
        assert source is right
        assert target is left


class TestHex:
    def test_adds_polychrome_to_chosen_joker(self):
        rng = _ControlledRng()
        c = Card()
        c.set_ability("c_hex")
        joker = _joker("j_test")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[joker], rng=rng))
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["edition"] == {"polychrome": True}
        assert result.add_edition["target"] is joker

    def test_destroys_other_non_eternal_jokers(self):
        rng = _ControlledRng()
        c = Card()
        c.set_ability("c_hex")
        j1 = _joker("j_a")
        j2 = _joker("j_b")
        j3 = _joker("j_c")
        result = use_consumable(c, ConsumableContext(card=c, jokers=[j1, j2, j3], rng=rng))
        assert result.destroy_jokers is not None
        assert j2 in result.destroy_jokers
        assert j3 in result.destroy_jokers
        assert j1 not in result.destroy_jokers


class TestSigil:
    def test_changes_all_cards_to_same_suit(self):
        rng = _ControlledRng()
        c = Card()
        c.set_ability("c_sigil")
        hand = [
            _card("Hearts", "5"),
            _card("Spades", "King"),
            _card("Diamonds", "3"),
            _card("Clubs", "Ace"),
        ]
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result is not None
        assert result.change_suit is not None
        assert len(result.change_suit) == len(hand)
        suits = {suit for _, suit in result.change_suit}
        assert len(suits) == 1, "All cards should be changed to the same suit"


class TestOuija:
    def test_changes_all_cards_to_same_rank(self):
        rng = _ControlledRng()
        c = Card()
        c.set_ability("c_ouija")
        hand = [
            _card("Hearts", "5"),
            _card("Spades", "King"),
            _card("Diamonds", "3"),
            _card("Clubs", "Ace"),
        ]
        result = use_consumable(c, ConsumableContext(card=c, hand_cards=hand, rng=rng))
        assert result is not None
        assert result.change_rank is not None
        assert len(result.change_rank) == len(hand)
        ranks = {rank for _, rank in result.change_rank}
        assert len(ranks) == 1, "All cards should be changed to the same rank"


class TestAura:
    def test_adds_edition_to_highlighted_card(self):
        rng = _ControlledRng([0.5])
        c = Card()
        c.set_ability("c_aura")
        target = _card("Hearts", "Ace")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target], rng=rng))
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["target"] is target


class TestTrance:
    def test_trance_adds_blue_seal(self):
        c = Card()
        c.set_ability("c_trance")
        target = _card("Clubs", "5")
        result = use_consumable(c, ConsumableContext(card=c, highlighted=[target]))
        assert result is not None
        assert result.add_seal == [(target, "Blue")]


# ============================================================================
# Fool (copy mechanic), Hanged Man (destroy), Wheel of Fortune (probability)
# ============================================================================


class TestFool:
    def test_creates_last_tarot_planet(self):
        c = Card()
        c.set_ability("c_fool")
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                game_state={"last_tarot_planet": "c_star"},
            ),
        )
        assert result is not None
        assert result.create is not None
        assert len(result.create) == 1
        assert result.create[0]["forced_key"] == "c_star"
        assert result.create[0]["type"] == "Tarot_Planet"


class TestHangedMan:
    def test_destroys_highlighted(self):
        c = _consumable("c_hanged_man")
        cards = [_card("Hearts", "5"), _card("Spades", "King")]
        result = use_consumable(
            c,
            ConsumableContext(card=c, highlighted=cards),
        )
        assert result is not None
        assert result.destroy is not None
        assert len(result.destroy) == 2
        assert result.destroy[0] is cards[0]
        assert result.destroy[1] is cards[1]


class TestWheelOfFortune:
    def _joker_with_sell(self, key: str, sell: int = 3) -> Card:
        j = _joker(key)
        j.sell_cost = sell
        return j

    def test_success_returns_add_edition(self):
        c = _consumable("c_wheel_of_fortune")
        c.set_ability("c_wheel_of_fortune")
        j = self._joker_with_sell("j_joker")
        rng = _ControlledRng([0.1, 0.6])
        result = use_consumable(
            c,
            ConsumableContext(
                card=c,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["target"] is j
        assert result.add_edition["edition"] == {"holo": True}


# ============================================================================
# Integration: Chariot → Steel scoring chain
# ============================================================================


class TestChariotIntegration:
    def test_apply_chariot_and_score_held(self):
        ace = _c("S", "A")
        chariot = _integ_consumable("c_chariot")

        result = use_consumable(
            chariot,
            ConsumableContext(card=chariot, highlighted=[ace]),
        )
        for card, enh_key in result.enhance:
            card.enhance(enh_key)

        assert ace.ability.get("h_x_mult") == 1.5

        two = _c("C", "2")
        r = score_hand_base(
            played_cards=[two],
            held_cards=[ace],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.hand_type == "High Card"
        assert r.mult == 1.5
        assert r.total == 10


# ============================================================================
# Integration: Mercury → level up scoring chain
# ============================================================================


class TestMercuryIntegration:
    def test_mercury_then_score_pair(self):
        levels = HandLevels()
        mercury = _integ_consumable("c_mercury")

        result = use_consumable(
            mercury,
            ConsumableContext(card=mercury, game_state={}),
        )
        _apply_result(result, levels)

        chips, mult = levels.get("Pair")
        assert chips == 25
        assert mult == 3

        r = score_hand_base(
            played_cards=[_c("H", "5"), _c("S", "5")],
            held_cards=[],
            hand_levels=levels,
            blind=_sb(),
            rng=_rng(),
        )
        assert r.hand_type == "Pair"
        assert r.chips == 35
        assert r.mult == 3
        assert r.total == 105


# ============================================================================
# Integration: Black Hole → all-level scoring chain
# ============================================================================


class TestBlackHoleIntegration:
    def _use_black_hole(self) -> tuple[ConsumableResult, HandLevels]:
        levels = HandLevels()
        bh = _integ_consumable("c_black_hole")
        result = use_consumable(bh, ConsumableContext(card=bh, game_state={}))
        _apply_result(result, levels)
        return result, levels

    def test_score_pair_after_black_hole(self):
        _, levels = self._use_black_hole()
        r = score_hand_base(
            played_cards=[_c("H", "5"), _c("S", "5")],
            held_cards=[],
            hand_levels=levels,
            blind=_sb(),
            rng=_rng(),
        )
        assert r.chips == 35
        assert r.mult == 3
        assert r.total == 105


# ============================================================================
# Integration: Death → copy scoring chain
# ============================================================================


class TestDeathIntegration:
    def test_apply_death_score_verifies_glass_x_mult(self):
        target = _c("C", "2")
        source = _c("H", "7", "m_glass")

        death = _integ_consumable("c_death")
        result = use_consumable(
            death,
            ConsumableContext(card=death, highlighted=[target, source]),
        )
        _apply_copy_card(*result.copy_card)

        r = score_hand_base(
            played_cards=[target],
            held_cards=[],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.hand_type == "High Card"
        assert r.chips == 12
        assert r.mult == 2.0
        assert r.total == 24


# ============================================================================
# Integration: Wheel of Fortune → edition scoring chain
# ============================================================================


class _IntegControlledRng:
    def __init__(self, random_values: list[float]) -> None:
        self._vals = iter(random_values)

    def random(self, key: str) -> float:
        return next(self._vals)

    def seed(self, key: str) -> float:
        return 0.5

    def element(self, table: list, seed_val: float) -> tuple:
        return (table[0], 0)

    def shuffle(self, _lst: list, _seed_val: float) -> None:
        pass


class TestWheelOfFortuneIntegration:
    def test_apply_foil_and_score(self):
        j = _integ_joker("j_joker")
        wof = _integ_consumable("c_wheel_of_fortune")
        rng = _IntegControlledRng([0.1, 0.3])
        result = use_consumable(
            wof,
            ConsumableContext(
                card=wof,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        j.set_edition(result.add_edition["edition"])
        assert j.edition is not None and j.edition.get("foil") is True

        ace = _c("S", "A")
        r = score_hand(
            played_cards=[ace],
            held_cards=[],
            jokers=[j],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.chips == 66
        assert r.mult == 5.0
        assert r.total == 330


# ============================================================================
# Integration: Clearance Sale → cost discount chain
# ============================================================================


class TestClearanceSaleIntegration:
    def test_joker_cost_discounted(self):
        game_state: dict = {}
        apply_voucher("v_clearance_sale", game_state)

        j = Card()
        j.set_ability("j_greedy_joker")
        j.set_cost(discount_percent=game_state["discount_percent"])
        assert j.cost == 4


# ============================================================================
# Integration: Full round earnings chain
# ============================================================================


class TestFullRoundEarnings:
    def _make_jokers(self) -> list[Card]:
        golden = Card()
        golden.center_key = "j_golden"
        golden.ability = {"name": "j_golden", "set": "Joker", "extra": 4}
        golden.sell_cost = 1

        cloud9 = Card()
        cloud9.center_key = "j_cloud_9"
        cloud9.ability = {"name": "j_cloud_9", "set": "Joker", "extra": 1, "nine_tally": 3}
        cloud9.sell_cost = 1

        rental = Card()
        rental.center_key = "j_spare_trousers"
        rental.ability = {"name": "j_spare_trousers", "set": "Joker", "rental": True}
        rental.sell_cost = 1

        return [golden, cloud9, rental]

    def test_full_breakdown(self):
        jokers = self._make_jokers()
        blind = _bb()
        result = calculate_round_earnings(
            blind=blind,
            hands_left=2,
            discards_left=0,
            money=23,
            jokers=jokers,
            game_state={},
        )
        assert result.blind_reward == 4
        assert result.unused_hands_bonus == 2
        assert result.unused_discards_bonus == 0
        assert result.joker_dollars == 7
        assert result.rental_cost == 3
        assert result.interest == 4
        assert result.total == 14
