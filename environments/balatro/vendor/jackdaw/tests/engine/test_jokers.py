"""Consolidated per-joker handler tests.

Tests individual joker handlers that have conditional logic, tricky state,
or RNG dependence. Simpler pattern-only handlers (hand-type containment,
flat bonuses, rank membership) are covered by oracle/pipeline tests.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import (
    _REGISTRY,
    JokerContext,
    JokerResult,
    calculate_joker,
    register,
    registered_jokers,
)
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _joker_card(center_key: str, *, debuff: bool = False, **ability_kw) -> Card:
    """Create a minimal joker Card for testing."""
    c = Card()
    c.center_key = center_key
    c.debuff = debuff
    c.ability = {"name": center_key, **ability_kw}
    return c


_SUIT_LETTER = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
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


def _playing_card(
    suit: str,
    rank: str,
    *,
    enhancement: str = "c_base",
) -> Card:
    """Create a playing card with proper base/suit for is_suit checks."""
    c = Card()
    c.set_base(f"{_SUIT_LETTER[suit]}_{_RANK_LETTER[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _wild_card(suit: str = "Hearts", rank: str = "5") -> Card:
    """Create a Wild Card (matches any suit in is_suit)."""
    c = _playing_card(suit, rank, enhancement="m_wild")
    return c


def _suit_ctx(other_card: Card, **kw) -> JokerContext:
    """Build an individual/play context with other_card."""
    return JokerContext(individual=True, cardarea="play", other_card=other_card, **kw)


def _held_ctx(other_card: Card, held_cards: list[Card] | None = None) -> JokerContext:
    return JokerContext(
        individual=True,
        cardarea="hand",
        other_card=other_card,
        held_cards=held_cards,
    )


def _poker_hands_with(*types: str) -> dict[str, list]:
    """Build a poker_hands dict where listed types have a non-empty entry."""
    all_types = [
        "Flush Five",
        "Flush House",
        "Five of a Kind",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Flush",
        "Straight",
        "Three of a Kind",
        "Two Pair",
        "Pair",
        "High Card",
    ]
    return {t: [["placeholder"]] if t in types else [] for t in all_types}


# ============================================================================
# Test joker handlers (registered only for testing, cleaned up after)
# ============================================================================

_TEST_KEY = "j_test_dummy"
_TEST_KEY_2 = "j_test_second"


@pytest.fixture(autouse=True)
def _register_test_jokers():
    """Register test jokers before each test, clean up after."""

    @register(_TEST_KEY)
    def _dummy(card: Card, ctx: JokerContext) -> JokerResult | None:
        if ctx.joker_main:
            return JokerResult(mult_mod=4.0)
        if ctx.individual and ctx.cardarea == "play":
            return JokerResult(chips=10.0)
        if ctx.repetition and ctx.cardarea == "play":
            return JokerResult(repetitions=1)
        return None

    @register(_TEST_KEY_2)
    def _second(card: Card, ctx: JokerContext) -> JokerResult | None:
        if ctx.joker_main:
            return JokerResult(Xmult_mod=1.5)
        return None

    yield

    _REGISTRY.pop(_TEST_KEY, None)
    _REGISTRY.pop(_TEST_KEY_2, None)


# ============================================================================
# Registry
# ============================================================================


class TestRegistry:
    def test_register_adds_to_registry(self):
        assert _TEST_KEY in _REGISTRY

    def test_registered_jokers_returns_sorted(self):
        keys = registered_jokers()
        assert _TEST_KEY in keys
        assert _TEST_KEY_2 in keys
        assert keys == sorted(keys)

    def test_register_overwrites(self):
        """Re-registering the same key replaces the handler."""

        @register(_TEST_KEY)
        def _replacement(card: Card, ctx: JokerContext) -> JokerResult | None:
            if ctx.joker_main:
                return JokerResult(mult_mod=999.0)
            return None

        card = _joker_card(_TEST_KEY)
        result = calculate_joker(card, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 999.0


# ============================================================================
# Dispatch
# ============================================================================


class TestDispatch:
    def test_joker_main_dispatch(self):
        card = _joker_card(_TEST_KEY)
        result = calculate_joker(card, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 4.0

    def test_individual_play_dispatch(self):
        card = _joker_card(_TEST_KEY)
        ctx = JokerContext(individual=True, cardarea="play")
        result = calculate_joker(card, ctx)
        assert result is not None
        assert result.chips == 10.0

    def test_repetition_dispatch(self):
        card = _joker_card(_TEST_KEY)
        ctx = JokerContext(repetition=True, cardarea="play")
        result = calculate_joker(card, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_unhandled_context_returns_none(self):
        """Handler returns None for contexts it doesn't handle."""
        card = _joker_card(_TEST_KEY)
        result = calculate_joker(card, JokerContext(end_of_round=True))
        assert result is None

    def test_second_joker_dispatch(self):
        card = _joker_card(_TEST_KEY_2)
        result = calculate_joker(card, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 1.5


# ============================================================================
# Debuff handling
# ============================================================================


class TestDebuff:
    def test_debuffed_joker_returns_none(self):
        card = _joker_card(_TEST_KEY, debuff=True)
        result = calculate_joker(card, JokerContext(joker_main=True))
        assert result is None

    def test_non_debuffed_joker_returns_result(self):
        card = _joker_card(_TEST_KEY, debuff=False)
        result = calculate_joker(card, JokerContext(joker_main=True))
        assert result is not None


# ============================================================================
# RNG-dependent jokers
# ============================================================================


class TestMisprintHandler:
    """j_misprint: random mult between 0 and 23 each hand."""

    def test_misprint_with_rng(self):
        card = _joker_card("j_misprint", extra={"min": 0, "max": 23})
        rng = PseudoRandom("TEST")
        ctx = JokerContext(joker_main=True, rng=rng)
        result = calculate_joker(card, ctx)
        assert result is not None
        assert 0 <= result.mult_mod <= 23

    def test_misprint_without_rng_returns_zero(self):
        card = _joker_card("j_misprint", extra={"min": 0, "max": 23})
        ctx = JokerContext(joker_main=True, rng=None)
        result = calculate_joker(card, ctx)
        assert result is not None
        assert result.mult_mod == 0

    def test_misprint_deterministic(self):
        """Same seed produces same roll."""
        card = _joker_card("j_misprint", extra={"min": 0, "max": 23})
        r1 = calculate_joker(card, JokerContext(joker_main=True, rng=PseudoRandom("SEED1")))
        r2 = calculate_joker(card, JokerContext(joker_main=True, rng=PseudoRandom("SEED1")))
        assert r1.mult_mod == r2.mult_mod

    def test_misprint_individual_returns_none(self):
        card = _joker_card("j_misprint", extra={"min": 0, "max": 23})
        ctx = JokerContext(individual=True, cardarea="play")
        result = calculate_joker(card, ctx)
        assert result is None


class TestBloodstoneHandler:
    """j_bloodstone: probabilistic x1.5 mult per Heart scored."""

    def test_heart_with_rng_triggers(self):
        """With deterministic seed, Heart should trigger (or not) consistently."""
        joker = _joker_card("j_bloodstone", extra={"odds": 2, "Xmult": 1.5})
        rng = PseudoRandom("BLOODTEST")
        ctx = _suit_ctx(
            _playing_card("Hearts", "Ace"),
            rng=rng,
            probabilities_normal=1.0,
        )
        result = calculate_joker(joker, ctx)
        # With odds=2 and normal=1, probability is 0.5
        # Result depends on RNG — just verify structure
        if result is not None:
            assert result.x_mult == 1.5

    def test_heart_without_rng_no_effect(self):
        joker = _joker_card("j_bloodstone", extra={"odds": 2, "Xmult": 1.5})
        ctx = _suit_ctx(_playing_card("Hearts", "Ace"))
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_spade_no_effect(self):
        joker = _joker_card("j_bloodstone", extra={"odds": 2, "Xmult": 1.5})
        rng = PseudoRandom("BLOODTEST")
        ctx = _suit_ctx(
            _playing_card("Spades", "Ace"),
            rng=rng,
            probabilities_normal=1.0,
        )
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_high_probability_always_triggers(self):
        """With very high probabilities_normal, should always trigger."""
        joker = _joker_card("j_bloodstone", extra={"odds": 2, "Xmult": 1.5})
        rng = PseudoRandom("BTEST")
        ctx = _suit_ctx(
            _playing_card("Hearts", "5"),
            rng=rng,
            probabilities_normal=1000.0,
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 1.5

    def test_deterministic(self):
        """Same seed → same result."""
        joker = _joker_card("j_bloodstone", extra={"odds": 2, "Xmult": 1.5})
        r1 = calculate_joker(
            joker,
            _suit_ctx(
                _playing_card("Hearts", "5"),
                rng=PseudoRandom("S1"),
                probabilities_normal=1.0,
            ),
        )
        r2 = calculate_joker(
            joker,
            _suit_ctx(
                _playing_card("Hearts", "5"),
                rng=PseudoRandom("S1"),
                probabilities_normal=1.0,
            ),
        )
        assert (r1 is None) == (r2 is None)
        if r1 is not None:
            assert r1.x_mult == r2.x_mult


class TestBusinessCard:
    """j_business: face card played → 1/2 chance → +$2."""

    def test_face_with_high_probability(self):
        """High probabilities_normal guarantees trigger."""
        joker = _joker_card("j_business", extra=2)
        ctx = _suit_ctx(
            _playing_card("Hearts", "King"),
            rng=PseudoRandom("BIZ"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 2

    def test_non_face_no_effect(self):
        joker = _joker_card("j_business", extra=2)
        ctx = _suit_ctx(
            _playing_card("Hearts", "5"),
            rng=PseudoRandom("BIZ"),
            probabilities_normal=1000.0,
        )
        assert calculate_joker(joker, ctx) is None

    def test_face_without_rng_no_effect(self):
        joker = _joker_card("j_business", extra=2)
        ctx = _suit_ctx(_playing_card("Hearts", "King"))
        assert calculate_joker(joker, ctx) is None

    def test_deterministic(self):
        joker = _joker_card("j_business", extra=2)
        r1 = calculate_joker(
            joker,
            _suit_ctx(
                _playing_card("Hearts", "Jack"),
                rng=PseudoRandom("S1"),
                probabilities_normal=1.0,
            ),
        )
        r2 = calculate_joker(
            joker,
            _suit_ctx(
                _playing_card("Hearts", "Jack"),
                rng=PseudoRandom("S1"),
                probabilities_normal=1.0,
            ),
        )
        assert (r1 is None) == (r2 is None)


# ============================================================================
# Multi-condition / state jokers
# ============================================================================


class TestCardSharpHandler:
    """Card Sharp: x3 mult if same hand type played twice this round."""

    def test_card_sharp_first_pair_no_effect(self):
        levels = HandLevels()
        levels.record_play("Pair")  # played_this_round = 1
        card = _joker_card("j_card_sharp", extra={"Xmult": 3})
        ctx = JokerContext(
            joker_main=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        result = calculate_joker(card, ctx)
        assert result is None

    def test_card_sharp_second_pair_triggers(self):
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")  # played_this_round = 2
        card = _joker_card("j_card_sharp", extra={"Xmult": 3})
        ctx = JokerContext(
            joker_main=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        result = calculate_joker(card, ctx)
        assert result is not None
        assert result.Xmult_mod == 3

    def test_card_sharp_different_type_no_effect(self):
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")
        card = _joker_card("j_card_sharp", extra={"Xmult": 3})
        ctx = JokerContext(
            joker_main=True,
            scoring_name="Flush",  # different type — Flush played 0 times
            hand_levels=levels,
        )
        result = calculate_joker(card, ctx)
        assert result is None

    def test_card_sharp_after_round_reset(self):
        """After reset_round_counts, played_this_round resets to 0."""
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")
        levels.reset_round_counts()
        levels.record_play("Pair")  # played_this_round = 1 again
        card = _joker_card("j_card_sharp", extra={"Xmult": 3})
        ctx = JokerContext(
            joker_main=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        result = calculate_joker(card, ctx)
        assert result is None


class TestPhotographHandler:
    """j_photograph: x2 mult on FIRST face card only."""

    def test_first_king_triggers(self):
        k1 = _playing_card("Hearts", "King")
        k2 = _playing_card("Spades", "King")
        five = _playing_card("Diamonds", "5")
        scoring = [five, k1, k2]
        joker = _joker_card("j_photograph", extra=2)
        # First face card is k1
        ctx = _suit_ctx(k1, scoring_hand=scoring)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 2

    def test_second_king_no_effect(self):
        k1 = _playing_card("Hearts", "King")
        k2 = _playing_card("Spades", "King")
        five = _playing_card("Diamonds", "5")
        scoring = [five, k1, k2]
        joker = _joker_card("j_photograph", extra=2)
        # k2 is NOT the first face card
        ctx = _suit_ctx(k2, scoring_hand=scoring)
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_no_face_no_effect(self):
        five = _playing_card("Hearts", "5")
        three = _playing_card("Spades", "3")
        scoring = [five, three]
        joker = _joker_card("j_photograph", extra=2)
        ctx = _suit_ctx(five, scoring_hand=scoring)
        assert calculate_joker(joker, ctx) is None

    def test_pareidolia_first_card_triggers(self):
        """With Pareidolia, first card in scoring_hand is the first face card."""
        five = _playing_card("Hearts", "5")
        three = _playing_card("Spades", "3")
        scoring = [five, three]
        joker = _joker_card("j_photograph", extra=2)
        ctx = _suit_ctx(five, scoring_hand=scoring, pareidolia=True)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 2

    def test_pareidolia_second_card_no_effect(self):
        five = _playing_card("Hearts", "5")
        three = _playing_card("Spades", "3")
        scoring = [five, three]
        joker = _joker_card("j_photograph", extra=2)
        ctx = _suit_ctx(three, scoring_hand=scoring, pareidolia=True)
        assert calculate_joker(joker, ctx) is None


class TestIdolHandler:
    """j_idol: x2 mult if scored card matches idol_card rank AND suit."""

    def test_matching_rank_and_suit_triggers(self):
        joker = _joker_card("j_idol", extra=2)
        ctx = _suit_ctx(
            _playing_card("Hearts", "Ace"),
            idol_card={"id": 14, "suit": "Hearts"},
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 2

    def test_matching_rank_wrong_suit_no_effect(self):
        """Rank matches but suit doesn't → no effect."""
        joker = _joker_card("j_idol", extra=2)
        ctx = _suit_ctx(
            _playing_card("Spades", "Ace"),
            idol_card={"id": 14, "suit": "Hearts"},
        )
        assert calculate_joker(joker, ctx) is None

    def test_matching_suit_wrong_rank_no_effect(self):
        """Suit matches but rank doesn't → no effect."""
        joker = _joker_card("j_idol", extra=2)
        ctx = _suit_ctx(
            _playing_card("Hearts", "King"),
            idol_card={"id": 14, "suit": "Hearts"},
        )
        assert calculate_joker(joker, ctx) is None

    def test_no_idol_card_no_effect(self):
        joker = _joker_card("j_idol", extra=2)
        ctx = _suit_ctx(_playing_card("Hearts", "Ace"))
        assert calculate_joker(joker, ctx) is None

    def test_wild_card_matches_suit(self):
        """Wild Card matches any suit, so only rank needs to match."""
        joker = _joker_card("j_idol", extra=2)
        wild = _wild_card("Clubs", "Ace")
        ctx = _suit_ctx(wild, idol_card={"id": 14, "suit": "Diamonds"})
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 2


class TestLoyaltyCard:
    """Loyalty Card: x4 every 6th hand (every=5, triggers when remainder==every)."""

    def test_sixth_hand_triggers(self):
        """hands_played=5 (delta=5 from create at 0): triggers."""
        joker = _joker_card(
            "j_loyalty_card",
            extra={"Xmult": 4, "every": 5},
            hands_played_at_create=0,
        )
        ctx = JokerContext(joker_main=True, hands_played=5)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 4

    def test_first_hand_no_effect(self):
        joker = _joker_card(
            "j_loyalty_card",
            extra={"Xmult": 4, "every": 5},
            hands_played_at_create=0,
        )
        ctx = JokerContext(joker_main=True, hands_played=0)
        assert calculate_joker(joker, ctx) is None

    def test_twelfth_hand_triggers(self):
        """delta=11 from create: (4-11)%6 = (-7)%6 = 5 = every → triggers."""
        joker = _joker_card(
            "j_loyalty_card",
            extra={"Xmult": 4, "every": 5},
            hands_played_at_create=0,
        )
        ctx = JokerContext(joker_main=True, hands_played=11)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 4

    def test_seventh_hand_no_effect(self):
        """delta=6: (4-6)%6 = (-2)%6 = 4 ≠ 5 → no effect."""
        joker = _joker_card(
            "j_loyalty_card",
            extra={"Xmult": 4, "every": 5},
            hands_played_at_create=0,
        )
        ctx = JokerContext(joker_main=True, hands_played=6)
        assert calculate_joker(joker, ctx) is None

    def test_created_mid_run(self):
        """Created at hands_played=10, triggers at hands_played=15."""
        joker = _joker_card(
            "j_loyalty_card",
            extra={"Xmult": 4, "every": 5},
            hands_played_at_create=10,
        )
        ctx = JokerContext(joker_main=True, hands_played=15)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 4


class TestMatador:
    """j_matador: +$8 when boss blind's debuff effect triggers."""

    def _make_blind(self, *, triggered: bool) -> object:
        from jackdaw.engine.blind import Blind

        b = Blind.create("bl_eye", ante=1)
        b.triggered = triggered
        return b

    def test_triggered_boss_earns_money(self):
        joker = _joker_card("j_matador", extra=8)
        blind = self._make_blind(triggered=True)
        ctx = JokerContext(debuffed_hand=True, blind=blind)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 8

    def test_non_triggered_boss_no_effect(self):
        joker = _joker_card("j_matador", extra=8)
        blind = self._make_blind(triggered=False)
        ctx = JokerContext(debuffed_hand=True, blind=blind)
        assert calculate_joker(joker, ctx) is None

    def test_no_blind_no_effect(self):
        joker = _joker_card("j_matador", extra=8)
        ctx = JokerContext(debuffed_hand=True)
        assert calculate_joker(joker, ctx) is None

    def test_not_debuffed_hand_no_effect(self):
        joker = _joker_card("j_matador", extra=8)
        blind = self._make_blind(triggered=True)
        ctx = JokerContext(debuffed_hand=False, blind=blind)
        assert calculate_joker(joker, ctx) is None

    def test_disabled_blind_triggered(self):
        """Disabled blind with triggered=True still pays (disabled prevents
        future debuffs, but Matador checks the triggered flag)."""
        joker = _joker_card("j_matador", extra=8)
        blind = self._make_blind(triggered=True)
        blind.disabled = True
        ctx = JokerContext(debuffed_hand=True, blind=blind)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 8


# ============================================================================
# Game-state-dependent jokers (joker_main context)
# ============================================================================


class TestAcrobat:
    def test_last_hand_triggers(self):
        joker = _joker_card("j_acrobat", extra=3)
        ctx = JokerContext(joker_main=True, hands_left=0)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 3

    def test_hands_remaining_no_effect(self):
        joker = _joker_card("j_acrobat", extra=3)
        ctx = JokerContext(joker_main=True, hands_left=2)
        assert calculate_joker(joker, ctx) is None


class TestMysticSummit:
    def test_zero_discards_triggers(self):
        joker = _joker_card("j_mystic_summit", extra={"mult": 15, "d_remaining": 0})
        ctx = JokerContext(joker_main=True, discards_left=0)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.mult_mod == 15

    def test_discards_remaining_no_effect(self):
        joker = _joker_card("j_mystic_summit", extra={"mult": 15, "d_remaining": 0})
        ctx = JokerContext(joker_main=True, discards_left=3)
        assert calculate_joker(joker, ctx) is None


class TestDriversLicense:
    def test_sixteen_enhanced_triggers(self):
        joker = _joker_card("j_drivers_license", extra=3)
        ctx = JokerContext(joker_main=True, enhanced_card_count=16)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 3

    def test_fifteen_enhanced_no_effect(self):
        joker = _joker_card("j_drivers_license", extra=3)
        ctx = JokerContext(joker_main=True, enhanced_card_count=15)
        assert calculate_joker(joker, ctx) is None


class TestBlackboard:
    def test_all_black_triggers(self):
        joker = _joker_card("j_blackboard", extra=3)
        held = [
            _playing_card("Spades", "5"),
            _playing_card("Clubs", "King"),
            _playing_card("Spades", "Ace"),
        ]
        ctx = JokerContext(joker_main=True, held_cards=held)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 3

    def test_one_heart_no_effect(self):
        joker = _joker_card("j_blackboard", extra=3)
        held = [
            _playing_card("Spades", "5"),
            _playing_card("Hearts", "King"),
            _playing_card("Clubs", "Ace"),
        ]
        ctx = JokerContext(joker_main=True, held_cards=held)
        assert calculate_joker(joker, ctx) is None

    def test_empty_held_no_effect(self):
        joker = _joker_card("j_blackboard", extra=3)
        ctx = JokerContext(joker_main=True, held_cards=[])
        assert calculate_joker(joker, ctx) is None

    def test_wild_card_counts(self):
        """Wild Card is_suit returns True for any suit including Clubs/Spades."""
        joker = _joker_card("j_blackboard", extra=3)
        held = [_wild_card("Hearts", "5")]
        ctx = JokerContext(joker_main=True, held_cards=held)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 3


class TestFlowerPot:
    def test_all_four_suits_triggers(self):
        joker = _joker_card("j_flower_pot", extra=3)
        scoring = [
            _playing_card("Hearts", "5"),
            _playing_card("Diamonds", "King"),
            _playing_card("Spades", "Ace"),
            _playing_card("Clubs", "3"),
        ]
        ctx = JokerContext(joker_main=True, scoring_hand=scoring)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 3

    def test_missing_suit_no_effect(self):
        joker = _joker_card("j_flower_pot", extra=3)
        scoring = [
            _playing_card("Hearts", "5"),
            _playing_card("Diamonds", "King"),
            _playing_card("Spades", "Ace"),
        ]
        ctx = JokerContext(joker_main=True, scoring_hand=scoring)
        assert calculate_joker(joker, ctx) is None

    def test_wild_fills_missing_suit(self):
        joker = _joker_card("j_flower_pot", extra=3)
        scoring = [
            _playing_card("Hearts", "5"),
            _playing_card("Diamonds", "King"),
            _playing_card("Spades", "Ace"),
            _wild_card("Hearts", "3"),
        ]
        ctx = JokerContext(joker_main=True, scoring_hand=scoring)
        result = calculate_joker(joker, ctx)
        assert result is not None


class TestSeeingDouble:
    def test_club_and_heart_triggers(self):
        joker = _joker_card("j_seeing_double", extra=2)
        scoring = [
            _playing_card("Clubs", "5"),
            _playing_card("Hearts", "King"),
        ]
        ctx = JokerContext(joker_main=True, scoring_hand=scoring)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.Xmult_mod == 2

    def test_only_clubs_no_effect(self):
        joker = _joker_card("j_seeing_double", extra=2)
        scoring = [
            _playing_card("Clubs", "5"),
            _playing_card("Clubs", "King"),
        ]
        ctx = JokerContext(joker_main=True, scoring_hand=scoring)
        assert calculate_joker(joker, ctx) is None

    def test_no_clubs_no_effect(self):
        joker = _joker_card("j_seeing_double", extra=2)
        scoring = [
            _playing_card("Hearts", "5"),
            _playing_card("Spades", "King"),
        ]
        ctx = JokerContext(joker_main=True, scoring_hand=scoring)
        assert calculate_joker(joker, ctx) is None


# ============================================================================
# Held-card jokers (individual context, cardarea='hand')
# ============================================================================


class TestRaisedFist:
    """j_raised_fist: +2× lowest held card's rank as mult."""

    def test_lowest_card_triggers(self):
        two = _playing_card("Hearts", "2")
        king = _playing_card("Spades", "King")
        held = [king, two]
        joker = _joker_card("j_raised_fist")
        result = calculate_joker(joker, _held_ctx(two, held))
        assert result is not None
        assert result.h_mult == 4  # 2 × nominal(2) = 4

    def test_non_lowest_no_effect(self):
        two = _playing_card("Hearts", "2")
        king = _playing_card("Spades", "King")
        held = [king, two]
        joker = _joker_card("j_raised_fist")
        assert calculate_joker(joker, _held_ctx(king, held)) is None

    def test_debuffed_card_returns_message(self):
        two = _playing_card("Hearts", "2")
        two.debuff = True
        held = [two]
        joker = _joker_card("j_raised_fist")
        result = calculate_joker(joker, _held_ctx(two, held))
        assert result is not None
        assert result.message == "Debuffed"
        assert result.h_mult == 0


class TestShootTheMoon:
    """j_shoot_the_moon: +13 mult per Queen held."""

    def test_queen_triggers(self):
        queen = _playing_card("Hearts", "Queen")
        joker = _joker_card("j_shoot_the_moon", extra=13)
        result = calculate_joker(joker, _held_ctx(queen))
        assert result is not None
        assert result.h_mult == 13

    def test_king_no_effect(self):
        king = _playing_card("Hearts", "King")
        joker = _joker_card("j_shoot_the_moon", extra=13)
        assert calculate_joker(joker, _held_ctx(king)) is None

    def test_debuffed_queen(self):
        queen = _playing_card("Hearts", "Queen")
        queen.debuff = True
        joker = _joker_card("j_shoot_the_moon", extra=13)
        result = calculate_joker(joker, _held_ctx(queen))
        assert result is not None
        assert result.message == "Debuffed"


class TestBaron:
    """j_baron: x1.5 per King held."""

    def test_king_triggers(self):
        king = _playing_card("Spades", "King")
        joker = _joker_card("j_baron", extra=1.5)
        result = calculate_joker(joker, _held_ctx(king))
        assert result is not None
        assert result.x_mult == 1.5

    def test_queen_no_effect(self):
        queen = _playing_card("Hearts", "Queen")
        joker = _joker_card("j_baron", extra=1.5)
        assert calculate_joker(joker, _held_ctx(queen)) is None

    def test_debuffed_king(self):
        king = _playing_card("Hearts", "King")
        king.debuff = True
        joker = _joker_card("j_baron", extra=1.5)
        result = calculate_joker(joker, _held_ctx(king))
        assert result is not None
        assert result.message == "Debuffed"


# ============================================================================
# Economy jokers with conditions
# ============================================================================


class TestReservedParking:
    """j_reserved_parking: face card held → 1/2 chance → +$1."""

    def test_face_held_high_probability(self):
        joker = _joker_card("j_reserved_parking", extra={"odds": 2, "dollars": 1})
        king = _playing_card("Hearts", "King")
        ctx = JokerContext(
            individual=True,
            cardarea="hand",
            other_card=king,
            rng=PseudoRandom("PARK"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 1

    def test_non_face_no_effect(self):
        joker = _joker_card("j_reserved_parking", extra={"odds": 2, "dollars": 1})
        five = _playing_card("Hearts", "5")
        ctx = JokerContext(
            individual=True,
            cardarea="hand",
            other_card=five,
            rng=PseudoRandom("PARK"),
            probabilities_normal=1000.0,
        )
        assert calculate_joker(joker, ctx) is None

    def test_debuffed_face_no_effect(self):
        """Debuffed King: is_face() returns False → handler never fires."""
        joker = _joker_card("j_reserved_parking", extra={"odds": 2, "dollars": 1})
        king = _playing_card("Hearts", "King")
        king.debuff = True
        ctx = JokerContext(
            individual=True,
            cardarea="hand",
            other_card=king,
            rng=PseudoRandom("PARK"),
            probabilities_normal=1000.0,
        )
        assert calculate_joker(joker, ctx) is None


class TestFaceless:
    """j_faceless: +$5 if ≥3 face cards discarded."""

    def test_three_face_cards(self):
        joker = _joker_card("j_faceless", extra={"dollars": 5, "faces": 3})
        discarded = [
            _playing_card("Hearts", "Jack"),
            _playing_card("Spades", "Queen"),
            _playing_card("Clubs", "King"),
        ]
        # Fires on last card in full_hand
        ctx = JokerContext(
            discard=True,
            other_card=discarded[-1],
            full_hand=discarded,
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 5

    def test_two_face_cards_no_effect(self):
        joker = _joker_card("j_faceless", extra={"dollars": 5, "faces": 3})
        discarded = [
            _playing_card("Hearts", "Jack"),
            _playing_card("Spades", "Queen"),
            _playing_card("Clubs", "5"),
        ]
        ctx = JokerContext(
            discard=True,
            other_card=discarded[-1],
            full_hand=discarded,
        )
        assert calculate_joker(joker, ctx) is None

    def test_not_last_card_no_effect(self):
        """Only fires when other_card is the last in full_hand."""
        joker = _joker_card("j_faceless", extra={"dollars": 5, "faces": 3})
        discarded = [
            _playing_card("Hearts", "Jack"),
            _playing_card("Spades", "Queen"),
            _playing_card("Clubs", "King"),
        ]
        ctx = JokerContext(
            discard=True,
            other_card=discarded[0],
            full_hand=discarded,
        )
        assert calculate_joker(joker, ctx) is None


class TestMail:
    """j_mail: +$5 per discarded card matching mail_card rank."""

    def test_matching_rank(self):
        joker = _joker_card("j_mail", extra=5)
        ace = _playing_card("Hearts", "Ace")
        ctx = JokerContext(
            discard=True,
            other_card=ace,
            mail_card_id=14,
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 5

    def test_non_matching_rank(self):
        joker = _joker_card("j_mail", extra=5)
        king = _playing_card("Hearts", "King")
        ctx = JokerContext(
            discard=True,
            other_card=king,
            mail_card_id=14,
        )
        assert calculate_joker(joker, ctx) is None

    def test_debuffed_card_no_effect(self):
        joker = _joker_card("j_mail", extra=5)
        ace = _playing_card("Hearts", "Ace")
        ace.debuff = True
        ctx = JokerContext(
            discard=True,
            other_card=ace,
            mail_card_id=14,
        )
        assert calculate_joker(joker, ctx) is None

    def test_no_mail_card_no_effect(self):
        joker = _joker_card("j_mail", extra=5)
        ace = _playing_card("Hearts", "Ace")
        ctx = JokerContext(discard=True, other_card=ace)
        assert calculate_joker(joker, ctx) is None


class TestTrading:
    """j_trading: first discard, single card → +$3, destroy card."""

    def test_first_discard_single_card(self):
        joker = _joker_card("j_trading", extra=3)
        card = _playing_card("Hearts", "5")
        ctx = JokerContext(
            discard=True,
            full_hand=[card],
            discards_used=0,
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 3
        assert result.remove is True
        assert result.extra == {"destroy": True}

    def test_first_discard_multiple_cards_no_effect(self):
        joker = _joker_card("j_trading", extra=3)
        cards = [_playing_card("Hearts", "5"), _playing_card("Spades", "3")]
        ctx = JokerContext(
            discard=True,
            full_hand=cards,
            discards_used=0,
        )
        assert calculate_joker(joker, ctx) is None

    def test_second_discard_no_effect(self):
        joker = _joker_card("j_trading", extra=3)
        card = _playing_card("Hearts", "5")
        ctx = JokerContext(
            discard=True,
            full_hand=[card],
            discards_used=1,
        )
        assert calculate_joker(joker, ctx) is None


class TestToDoList:
    """j_todo_list: +$4 if hand matches to_do_poker_hand."""

    def test_matching_hand(self):
        joker = _joker_card(
            "j_todo_list",
            extra={"dollars": 4, "poker_hand": "High Card"},
            to_do_poker_hand="Pair",
        )
        ctx = JokerContext(joker_main=True, scoring_name="Pair")
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.dollars == 4

    def test_non_matching_hand(self):
        joker = _joker_card(
            "j_todo_list",
            extra={"dollars": 4, "poker_hand": "High Card"},
            to_do_poker_hand="Pair",
        )
        ctx = JokerContext(joker_main=True, scoring_name="Flush")
        assert calculate_joker(joker, ctx) is None

    def test_no_target_no_effect(self):
        joker = _joker_card(
            "j_todo_list",
            extra={"dollars": 4, "poker_hand": "High Card"},
        )
        ctx = JokerContext(joker_main=True, scoring_name="Pair")
        assert calculate_joker(joker, ctx) is None


# ============================================================================
# Suit-conditional edge cases
# ============================================================================


class TestGreedyJoker:
    """j_greedy_joker: +3 mult per Diamond scored."""

    def test_diamond_triggers(self):
        joker = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        ctx = _suit_ctx(_playing_card("Diamonds", "Ace"))
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.mult == 3

    def test_spade_no_effect(self):
        joker = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        ctx = _suit_ctx(_playing_card("Spades", "Ace"))
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_wild_card_triggers(self):
        """Wild Card matches any suit → triggers."""
        joker = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        ctx = _suit_ctx(_wild_card("Hearts", "5"))
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.mult == 3

    def test_smeared_hearts_triggers(self):
        """Smeared: Hearts ↔ Diamonds interchangeable → Hearts triggers."""
        joker = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        ctx = _suit_ctx(_playing_card("Hearts", "5"), smeared=True)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.mult == 3

    def test_smeared_clubs_no_effect(self):
        """Smeared: Clubs is black, Diamonds is red → no match."""
        joker = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        ctx = _suit_ctx(_playing_card("Clubs", "5"), smeared=True)
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_joker_main_context_no_effect(self):
        """Wrong context phase → None."""
        joker = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        ctx = JokerContext(
            joker_main=True,
            other_card=_playing_card("Diamonds", "Ace"),
        )
        result = calculate_joker(joker, ctx)
        assert result is None


class TestAncientJoker:
    """j_ancient: x1.5 mult if scored card matches ancient_suit."""

    def test_matching_suit_triggers(self):
        joker = _joker_card("j_ancient", extra=1.5)
        ctx = _suit_ctx(
            _playing_card("Spades", "King"),
            ancient_suit="Spades",
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 1.5

    def test_non_matching_suit_no_effect(self):
        joker = _joker_card("j_ancient", extra=1.5)
        ctx = _suit_ctx(
            _playing_card("Hearts", "King"),
            ancient_suit="Spades",
        )
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_no_ancient_suit_no_effect(self):
        joker = _joker_card("j_ancient", extra=1.5)
        ctx = _suit_ctx(_playing_card("Hearts", "King"))
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_wild_card_triggers(self):
        joker = _joker_card("j_ancient", extra=1.5)
        ctx = _suit_ctx(_wild_card(), ancient_suit="Diamonds")
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 1.5


class TestGluttenousJoker:
    """j_gluttenous_joker: +3 mult per Club scored (typo in key matches source)."""

    def test_club_triggers(self):
        joker = _joker_card(
            "j_gluttenous_joker",
            extra={"s_mult": 3, "suit": "Clubs"},
        )
        ctx = _suit_ctx(_playing_card("Clubs", "Jack"))
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.mult == 3

    def test_heart_no_effect(self):
        joker = _joker_card(
            "j_gluttenous_joker",
            extra={"s_mult": 3, "suit": "Clubs"},
        )
        ctx = _suit_ctx(_playing_card("Hearts", "Jack"))
        result = calculate_joker(joker, ctx)
        assert result is None

    def test_smeared_spade_triggers(self):
        """Smeared: Spades ↔ Clubs interchangeable."""
        joker = _joker_card(
            "j_gluttenous_joker",
            extra={"s_mult": 3, "suit": "Clubs"},
        )
        ctx = _suit_ctx(_playing_card("Spades", "5"), smeared=True)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.mult == 3


# ============================================================================
# Rank-conditional edge cases
# ============================================================================


class TestScaryFaceHandler:
    """j_scary_face: +30 chips for face cards."""

    def test_king_triggers(self):
        joker = _joker_card("j_scary_face", extra=30)
        ctx = _suit_ctx(_playing_card("Spades", "King"))
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.chips == 30

    def test_queen_triggers(self):
        joker = _joker_card("j_scary_face", extra=30)
        ctx = _suit_ctx(_playing_card("Hearts", "Queen"))
        assert calculate_joker(joker, ctx) is not None

    def test_jack_triggers(self):
        joker = _joker_card("j_scary_face", extra=30)
        ctx = _suit_ctx(_playing_card("Clubs", "Jack"))
        assert calculate_joker(joker, ctx) is not None

    def test_ten_no_effect(self):
        joker = _joker_card("j_scary_face", extra=30)
        ctx = _suit_ctx(_playing_card("Hearts", "10"))
        assert calculate_joker(joker, ctx) is None

    def test_pareidolia_every_card(self):
        """With Pareidolia, ALL cards are face cards."""
        joker = _joker_card("j_scary_face", extra=30)
        ctx = _suit_ctx(_playing_card("Hearts", "2"), pareidolia=True)
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.chips == 30

    def test_pareidolia_ace(self):
        joker = _joker_card("j_scary_face", extra=30)
        ctx = _suit_ctx(_playing_card("Hearts", "Ace"), pareidolia=True)
        assert calculate_joker(joker, ctx) is not None


class TestTribouletHandler:
    """j_triboulet: x2 mult for King or Queen."""

    def test_king_triggers(self):
        joker = _joker_card("j_triboulet", extra=2)
        ctx = _suit_ctx(_playing_card("Hearts", "King"))
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 2

    def test_queen_triggers(self):
        joker = _joker_card("j_triboulet", extra=2)
        ctx = _suit_ctx(_playing_card("Spades", "Queen"))
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.x_mult == 2

    def test_jack_no_effect(self):
        """Jack is a face card but not King/Queen."""
        joker = _joker_card("j_triboulet", extra=2)
        ctx = _suit_ctx(_playing_card("Clubs", "Jack"))
        assert calculate_joker(joker, ctx) is None

    def test_ace_no_effect(self):
        joker = _joker_card("j_triboulet", extra=2)
        ctx = _suit_ctx(_playing_card("Hearts", "Ace"))
        assert calculate_joker(joker, ctx) is None


class TestEvenStevenHandler:
    """j_even_steven: +4 mult for even numbered cards (2/4/6/8/10)."""

    def test_ace_no_effect(self):
        """Ace (id=14) is outside 0-10 range."""
        joker = _joker_card("j_even_steven", extra=4)
        ctx = _suit_ctx(_playing_card("Hearts", "Ace"))
        assert calculate_joker(joker, ctx) is None


class TestOddToddHandler:
    """j_odd_todd: +31 chips for odd numbered cards (3/5/7/9) and Ace."""

    def test_ace_triggers(self):
        """Ace is special-cased as odd."""
        joker = _joker_card("j_odd_todd", extra=31)
        ctx = _suit_ctx(_playing_card("Hearts", "Ace"))
        assert calculate_joker(joker, ctx) is not None


class TestHackHandler:
    """j_hack: retrigger 2/3/4/5 cards."""

    def test_three_retriggers(self):
        joker = _joker_card("j_hack", extra=1)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_playing_card("Hearts", "3"),
        )
        result = calculate_joker(joker, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_six_no_effect(self):
        joker = _joker_card("j_hack", extra=1)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_playing_card("Hearts", "6"),
        )
        assert calculate_joker(joker, ctx) is None


# ============================================================================
# Blueprint and Brainstorm (copy/delegation)
# ============================================================================


class TestBlueprint:
    """j_blueprint: copies the joker to its right."""

    def test_copies_joker_to_right(self):
        """Blueprint left of j_joker → copies +4 mult."""
        bp = _joker_card("j_blueprint")
        joker = _joker_card("j_joker", mult=4)
        jokers = [bp, joker]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(bp, ctx)
        assert result is not None
        assert result.mult_mod == 4

    def test_copies_suit_joker(self):
        """Blueprint left of j_greedy_joker with Diamond scored → +3 mult."""
        bp = _joker_card("j_blueprint")
        greedy = _joker_card(
            "j_greedy_joker",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )
        jokers = [bp, greedy]
        ctx = JokerContext(
            individual=True,
            cardarea="play",
            jokers=jokers,
            other_card=_playing_card("Diamonds", "Ace"),
        )
        result = calculate_joker(bp, ctx)
        assert result is not None
        assert result.mult == 3

    def test_nothing_to_right(self):
        """Blueprint is rightmost joker → no effect."""
        bp = _joker_card("j_blueprint")
        joker = _joker_card("j_joker", mult=4)
        jokers = [joker, bp]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(bp, ctx) is None

    def test_debuffed_target_no_effect(self):
        bp = _joker_card("j_blueprint")
        joker = _joker_card("j_joker", debuff=True, mult=4)
        jokers = [bp, joker]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(bp, ctx) is None

    def test_no_joker_list_no_effect(self):
        bp = _joker_card("j_blueprint")
        ctx = JokerContext(joker_main=True)
        assert calculate_joker(bp, ctx) is None

    def test_two_blueprints_adjacent(self):
        """Two Blueprints: right one has nothing to copy, left copies right
        which returns None."""
        bp1 = _joker_card("j_blueprint")
        bp2 = _joker_card("j_blueprint")
        jokers = [bp1, bp2]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        # bp1 copies bp2, bp2 has nothing to its right → None
        assert calculate_joker(bp1, ctx) is None

    def test_blueprint_copies_target_state(self):
        """Blueprint reads the target's current ability state."""
        bp = _joker_card("j_blueprint")
        joker = _joker_card("j_joker", mult=12)  # modified mult
        jokers = [bp, joker]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(bp, ctx)
        assert result is not None
        assert result.mult_mod == 12  # uses target's current state


class TestBrainstorm:
    """j_brainstorm: copies the leftmost joker."""

    def test_copies_leftmost(self):
        joker = _joker_card("j_joker", mult=4)
        brain = _joker_card("j_brainstorm")
        jokers = [joker, brain]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(brain, ctx)
        assert result is not None
        assert result.mult_mod == 4

    def test_brainstorm_is_leftmost_skips_self(self):
        """Brainstorm is leftmost → copies the second joker."""
        brain = _joker_card("j_brainstorm")
        joker = _joker_card("j_joker", mult=4)
        jokers = [brain, joker]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(brain, ctx)
        assert result is not None
        assert result.mult_mod == 4

    def test_brainstorm_alone_no_effect(self):
        brain = _joker_card("j_brainstorm")
        jokers = [brain]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(brain, ctx) is None

    def test_debuffed_leftmost_no_effect(self):
        joker = _joker_card("j_joker", debuff=True, mult=4)
        brain = _joker_card("j_brainstorm")
        jokers = [joker, brain]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(brain, ctx) is None


class TestBlueprintBrainstormChain:
    """Blueprint copying Brainstorm which copies the leftmost."""

    def test_blueprint_brainstorm_chain(self):
        """[j_joker, j_blueprint, j_brainstorm]:
        Blueprint copies Brainstorm, Brainstorm copies j_joker → +4 mult."""
        joker = _joker_card("j_joker", mult=4)
        bp = _joker_card("j_blueprint")
        brain = _joker_card("j_brainstorm")
        jokers = [joker, bp, brain]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(bp, ctx)
        assert result is not None
        assert result.mult_mod == 4

    def test_infinite_loop_prevention(self):
        """[j_blueprint, j_brainstorm]: Blueprint copies Brainstorm,
        Brainstorm copies Blueprint (leftmost), Blueprint copies Brainstorm...
        Loop cap prevents infinite recursion."""
        bp = _joker_card("j_blueprint")
        brain = _joker_card("j_brainstorm")
        jokers = [bp, brain]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        # Should terminate without error due to blueprint counter cap
        result = calculate_joker(bp, ctx)
        # Eventually returns None when cap exceeded
        assert result is None
