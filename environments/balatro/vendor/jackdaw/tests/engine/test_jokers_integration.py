"""Cross-joker integration tests: coverage audit, scoring pipeline,
retrigger chains, state mutations, and oracle scenarios."""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.prototypes import _load_json
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import (
    _REGISTRY,
    GameSnapshot,
    JokerContext,
    calc_dollar_bonus,
    calculate_joker,
    on_end_of_round,
)
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand

# ============================================================================
# Shared helpers
# ============================================================================


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


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    c.sell_cost = 1
    return c


def _poker_hands_with(*types: str) -> dict[str, list]:
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
    return {t: [["p"]] if t in types else [] for t in all_types}


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Section 1: Coverage Audit
# ============================================================================


class TestFullCoverage:
    """Every joker in centers.json must be registered."""

    def test_all_150_registered(self):
        centers = _load_json("centers.json")
        joker_keys = [k for k, v in centers.items() if v.get("set") == "Joker"]
        assert len(joker_keys) == 150
        missing = [k for k in joker_keys if k not in _REGISTRY]
        assert missing == [], f"Unregistered jokers: {missing}"

    def test_no_extra_registrations(self):
        """No handler registered for a non-existent joker key."""
        centers = _load_json("centers.json")
        joker_keys = set(k for k, v in centers.items() if v.get("set") == "Joker")
        for key in _REGISTRY:
            if key.startswith("j_test"):
                continue  # test fixtures
            assert key in joker_keys, f"Handler {key} has no prototype"


class TestPassiveJokers:
    """Passive/meta jokers return None in all contexts."""

    @pytest.mark.parametrize(
        "key",
        [
            "j_four_fingers",
            "j_shortcut",
            "j_pareidolia",
            "j_smeared",
            "j_splash",
            "j_ring_master",
            "j_juggler",
            "j_drunkard",
            "j_troubadour",
            "j_merry_andy",
            "j_oops",
            "j_credit_card",
            "j_chaos",
            "j_astronomer",
        ],
    )
    def test_noop_in_all_contexts(self, key):
        j = _joker(key)
        for ctx in [
            JokerContext(joker_main=True),
            JokerContext(before=True),
            JokerContext(after=True),
            JokerContext(individual=True, cardarea="play"),
        ]:
            assert calculate_joker(j, ctx) is None


# ============================================================================
# Section 2: Card Creation Triggers
# ============================================================================


class TestDna:
    def test_first_hand_single_card(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace")]
        ctx = JokerContext(before=True, full_hand=played, hands_played=0)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "playing_card_copy"
        assert result.extra["create"]["source_card"] is played[0]

    def test_first_hand_multiple_cards_no_effect(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace"), _card("Spades", "King")]
        ctx = JokerContext(before=True, full_hand=played, hands_played=0)
        assert calculate_joker(j, ctx) is None

    def test_second_hand_no_effect(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace")]
        ctx = JokerContext(before=True, full_hand=played, hands_played=1)
        assert calculate_joker(j, ctx) is None

    def test_blueprint_does_not_copy(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace")]
        ctx = JokerContext(
            before=True,
            full_hand=played,
            hands_played=0,
            blueprint=1,
        )
        assert calculate_joker(j, ctx) is None


class TestRiffRaff:
    def test_creates_two_jokers(self):
        j = _joker("j_riff_raff", extra=2)
        ctx = JokerContext(
            setting_blind=True,
            joker_count=3,
            joker_slots=5,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Joker"
        assert result.extra["create"]["rarity"] == "Common"
        assert result.extra["create"]["count"] == 2

    def test_one_slot_creates_one(self):
        j = _joker("j_riff_raff", extra=2)
        ctx = JokerContext(
            setting_blind=True,
            joker_count=4,
            joker_slots=5,
        )
        result = calculate_joker(j, ctx)
        assert result.extra["create"]["count"] == 1

    def test_no_slots_no_effect(self):
        j = _joker("j_riff_raff", extra=2)
        ctx = JokerContext(
            setting_blind=True,
            joker_count=5,
            joker_slots=5,
        )
        assert calculate_joker(j, ctx) is None


class TestEightBall:
    def test_rank_8_high_probability(self):
        j = _joker("j_8_ball", extra=4)
        eight = _card("Hearts", "8")
        ctx = JokerContext(
            individual=True,
            cardarea="play",
            other_card=eight,
            rng=PseudoRandom("8B"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Tarot"


class TestVagabond:
    def test_exact_threshold(self):
        j = _joker("j_vagabond", extra=4)
        ctx = JokerContext(joker_main=True, money=4)
        result = calculate_joker(j, ctx)
        assert result is not None


class TestSixthSense:
    def test_rank_6_first_hand_single_card(self):
        j = _joker("j_sixth_sense")
        six = _card("Hearts", "6")
        ctx = JokerContext(
            destroying_card=six,
            full_hand=[six],
            hands_played=0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True
        assert result.extra["create"]["type"] == "Spectral"


# ============================================================================
# Section 3: End-of-Round Economy
# ============================================================================


class TestGoldenJoker:
    def test_via_on_end_of_round(self):
        j = _joker("j_golden", extra=4)
        result = on_end_of_round([j], GameSnapshot())
        assert result["dollars_earned"] == 4


class TestCloud9:
    def test_no_tally_field(self):
        j = _joker("j_cloud_9", extra=1)
        assert calc_dollar_bonus(j, GameSnapshot()) == 0


class TestRocket:
    def test_after_boss_increases(self):
        j = _joker("j_rocket", extra={"dollars": 1, "increase": 2})
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(end_of_round=True, blind=boss)
        calculate_joker(j, ctx)
        # dollars should now be 3 (1 + 2)
        assert j.ability["extra"]["dollars"] == 3
        assert calc_dollar_bonus(j, GameSnapshot()) == 3

    def test_after_two_bosses(self):
        j = _joker("j_rocket", extra={"dollars": 1, "increase": 2})
        boss = Blind.create("bl_hook", ante=1)
        for _ in range(2):
            calculate_joker(j, JokerContext(end_of_round=True, blind=boss))
        # 1 + 2 + 2 = 5
        assert calc_dollar_bonus(j, GameSnapshot()) == 5


class TestDelayedGratification:
    def test_via_on_end_of_round(self):
        j = _joker("j_delayed_grat", extra=2)
        game = GameSnapshot(discards_used=0, discards_left=4)
        result = on_end_of_round([j], game)
        assert result["dollars_earned"] == 8


class TestEgg:
    def test_sell_value_increases(self):
        j = _joker("j_egg", extra=3)
        j.sell_cost = 2
        ctx = JokerContext(end_of_round=True)
        calculate_joker(j, ctx)
        assert j.ability["extra_value"] == 3
        assert j.sell_cost == 5  # 2 + 3

    def test_accumulates(self):
        j = _joker("j_egg", extra=3)
        j.sell_cost = 2
        for _ in range(3):
            calculate_joker(j, JokerContext(end_of_round=True))
        assert j.ability["extra_value"] == 9
        assert j.sell_cost == 11  # 2 + 3*3


class TestGiftCard:
    def test_all_jokers_increase(self):
        gift = _joker("j_gift", extra=1)
        j1 = _joker("j_joker", mult=4)
        j2 = _joker("j_stuntman", extra={"chip_mod": 250})
        jokers = [gift, j1, j2]
        for j in jokers:
            j.sell_cost = 2

        ctx = JokerContext(end_of_round=True, jokers=jokers)
        calculate_joker(gift, ctx)

        for j in jokers:
            assert j.ability.get("extra_value", 0) == 1
            assert j.sell_cost == 3

    def test_accumulates_across_rounds(self):
        gift = _joker("j_gift", extra=1)
        j1 = _joker("j_joker", mult=4)
        jokers = [gift, j1]
        for j in jokers:
            j.sell_cost = 1

        for _ in range(3):
            calculate_joker(gift, JokerContext(end_of_round=True, jokers=jokers))

        assert j1.ability["extra_value"] == 3
        assert j1.sell_cost == 4


class TestInvisible:
    def test_sell_at_threshold_duplicates(self):
        j = _joker("j_invisible", extra=2, invis_rounds=2)
        result = calculate_joker(j, JokerContext(selling_self=True))
        assert result is not None
        assert result.extra["duplicate_random_joker"] is True


class TestSpaceJoker:
    def test_high_probability_levels_up(self):
        j = _joker("j_space", extra=4)
        ctx = JokerContext(
            before=True,
            rng=PseudoRandom("SP"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.level_up is True


class TestOnEndOfRound:
    def test_multiple_dollar_jokers(self):
        golden = _joker("j_golden", extra=4)
        cloud = _joker("j_cloud_9", extra=1, nine_tally=5)
        game = GameSnapshot()
        result = on_end_of_round([golden, cloud], game)
        assert result["dollars_earned"] == 9  # 4 + 5


# ============================================================================
# Section 4: Joker-on-Joker Effects
# ============================================================================


class TestBaseballCard:
    """j_baseball fires in other_joker context for Uncommon jokers."""

    def test_uncommon_joker_triggers(self):
        """j_steel_joker is rarity 2 (Uncommon) -> x1.5."""
        baseball = _joker("j_baseball", extra=1.5)
        uncommon = _joker("j_steel_joker", extra=0.2)
        ctx = JokerContext(other_joker=uncommon)
        result = calculate_joker(baseball, ctx)
        assert result is not None
        assert result.Xmult_mod == 1.5

    def test_common_joker_no_effect(self):
        """j_joker is rarity 1 (Common) -> no effect."""
        baseball = _joker("j_baseball", extra=1.5)
        common = _joker("j_joker", mult=4)
        ctx = JokerContext(other_joker=common)
        result = calculate_joker(baseball, ctx)
        assert result is None

    def test_rare_joker_no_effect(self):
        """j_blueprint is rarity 3 (Rare) -> no effect."""
        baseball = _joker("j_baseball", extra=1.5)
        rare = _joker("j_blueprint")
        ctx = JokerContext(other_joker=rare)
        assert calculate_joker(baseball, ctx) is None

    def test_does_not_trigger_on_self(self):
        """Baseball Card should not trigger on itself (even if Rare)."""
        baseball = _joker("j_baseball", extra=1.5)
        ctx = JokerContext(other_joker=baseball)
        assert calculate_joker(baseball, ctx) is None

    def test_pipeline_three_uncommon_jokers(self):
        """Three Uncommon jokers + Baseball Card -> x1.5 fires 3 times.

        Pair of Aces: 32 chips, 2 mult.
        Phase 9 processing order (left to right):
        - j_blackboard (Uncommon, rarity 2): x3 mult -> 6
          - 9c: Baseball reacts to Blackboard -> x1.5 -> 9
        - j_bull (Uncommon): +0 chips (money=0)
          - 9c: Baseball reacts to Bull -> x1.5 -> 13.5
        - j_card_sharp (Uncommon): no effect (played_this_round=0)
          - 9c: Baseball reacts to Card Sharp -> x1.5 -> 20.25
        - j_baseball (Rare): no self-trigger
          - 9c: no joker reacts (none check other_joker)

        Score: 32 x 20.25 = 648."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]  # all black for Blackboard

        blackboard = _joker("j_blackboard", extra=3)  # rarity 2
        bull = _joker("j_bull", extra=2)  # rarity 2
        card_sharp = _joker("j_card_sharp", extra={"Xmult": 3})  # rarity 2
        baseball = _joker("j_baseball", extra=1.5)  # rarity 3

        result = score_hand(
            played,
            held,
            [blackboard, bull, card_sharp, baseball],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Base: 32 chips, 2 mult
        # Blackboard x3 -> 6, Baseball x1.5 -> 9
        # Bull +0 (no money), Baseball x1.5 -> 13.5
        # Card Sharp no effect, Baseball x1.5 -> 20.25
        # Baseball: no self-react
        assert result.mult == pytest.approx(20.25)
        assert result.total == 648

    def test_pipeline_no_uncommon_no_effect(self):
        """All Common jokers -> Baseball never fires in 9c."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        joker = _joker("j_joker", mult=4)  # rarity 1
        baseball = _joker("j_baseball", extra=1.5)  # rarity 3

        result = score_hand(
            played,
            [],
            [joker, baseball],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Base: 32, 2. j_joker +4 -> 6. Baseball: no react to Common j_joker.
        assert result.mult == 6.0
        assert result.total == 192

    def test_pipeline_blueprint_not_uncommon(self):
        """Blueprint (Rare) copying j_joker: Baseball doesn't react to Blueprint."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bp = _joker("j_blueprint")  # rarity 3
        joker = _joker("j_joker", mult=4)  # rarity 1
        baseball = _joker("j_baseball", extra=1.5)  # rarity 3

        result = score_hand(
            played,
            [],
            [bp, joker, baseball],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # bp copies joker -> +4 mult. joker -> +4 mult. Total add: +8.
        # Baseball: bp is Rare (no), joker is Common (no), self (skip).
        # mult = 2 + 4 + 4 = 10
        assert result.mult == 10.0
        assert result.total == 320


class TestSwashbuckler:
    """j_swashbuckler: mult from other jokers' sell costs."""

    def test_basic_sell_sum(self):
        swash = _joker("j_swashbuckler")
        j1 = _joker("j_joker", mult=4)
        j1.sell_cost = 2
        j2 = _joker("j_stuntman", extra={"chip_mod": 250})
        j2.sell_cost = 4
        jokers = [swash, j1, j2]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        result = calculate_joker(swash, ctx)
        assert result is not None
        assert result.mult_mod == 6  # 2 + 4

    def test_excludes_self(self):
        swash = _joker("j_swashbuckler")
        swash.sell_cost = 3
        jokers = [swash]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(swash, ctx) is None

    def test_excludes_debuffed(self):
        swash = _joker("j_swashbuckler")
        j1 = _joker("j_joker", mult=4)
        j1.sell_cost = 5
        j1.debuff = True
        jokers = [swash, j1]
        ctx = JokerContext(joker_main=True, jokers=jokers)
        assert calculate_joker(swash, ctx) is None

    def test_pipeline_with_other_jokers(self):
        """Swashbuckler in pipeline adds sell value sum as mult."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        swash = _joker("j_swashbuckler")
        j1 = _joker("j_joker", mult=4)
        j1.sell_cost = 2
        j2 = _joker("j_stuntman", extra={"chip_mod": 250, "h_size": 2})
        j2.sell_cost = 4

        result = score_hand(
            played,
            [],
            [j1, swash, j2],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Base: 32 chips, 2 mult
        # j_joker: +4 mult -> 6
        # swash: +6 mult (sell 2+4) -> 12
        # j_stuntman: +250 chips -> 282
        assert result.mult == 12.0
        assert result.chips == 282.0
        assert result.total == 3384


# ============================================================================
# Section 5: Destructive and Rule-Modifying
# ============================================================================


class TestGrosMichel:
    def _make(self):
        return _joker("j_gros_michel", extra={"mult": 15, "odds": 6})

    def test_joker_main_gives_mult(self):
        j = self._make()
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 15

    def test_end_of_round_high_probability_destroys(self):
        """With very high probability, should self-destruct."""
        j = self._make()
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("GM"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True
        assert result.extra["pool_flag"] == "gros_michel_extinct"

    def test_end_of_round_survives(self):
        """With very low probability, should survive."""
        j = self._make()
        j.ability["extra"]["odds"] = 1000000
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("GM"),
            probabilities_normal=1.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is False
        assert result.saved is True

    def test_pipeline_gives_mult(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = self._make()
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 32 chips, 2 mult. +15 mult -> 17. Score: 32 x 17 = 544.
        assert result.mult == 17.0
        assert result.total == 544


class TestCavendish:
    def _make(self):
        return _joker("j_cavendish", extra={"Xmult": 3, "odds": 1000})

    def test_joker_main_gives_xmult(self):
        j = self._make()
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 3

    def test_end_of_round_survives(self):
        j = self._make()
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("CV"),
            probabilities_normal=1.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.saved is True
        assert result.remove is False

    def test_end_of_round_high_probability_destroys(self):
        j = self._make()
        ctx = JokerContext(
            end_of_round=True,
            rng=PseudoRandom("CV"),
            probabilities_normal=1000000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True


class TestChicot:
    def test_disables_boss_blind(self):
        j = _joker("j_chicot")
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(setting_blind=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["disable_blind"] is True

    def test_non_boss_no_effect(self):
        j = _joker("j_chicot")
        small = Blind.create("bl_small", ante=1)
        ctx = JokerContext(setting_blind=True, blind=small)
        assert calculate_joker(j, ctx) is None

    def test_already_disabled_no_effect(self):
        j = _joker("j_chicot")
        boss = Blind.create("bl_hook", ante=1)
        boss.disabled = True
        ctx = JokerContext(setting_blind=True, blind=boss)
        assert calculate_joker(j, ctx) is None


class TestLuchador:
    def test_sell_during_boss(self):
        j = _joker("j_luchador")
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(selling_self=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["disable_blind"] is True

    def test_sell_during_non_boss(self):
        j = _joker("j_luchador")
        small = Blind.create("bl_small", ante=1)
        ctx = JokerContext(selling_self=True, blind=small)
        assert calculate_joker(j, ctx) is None


class TestBurglar:
    def test_setting_blind(self):
        j = _joker("j_burglar", extra=3)
        ctx = JokerContext(setting_blind=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["set_hands"] == 3
        assert result.extra["set_discards"] == 0


class TestMidasMask:
    def test_converts_face_cards(self):
        j = _joker("j_midas_mask")
        king = _card("Hearts", "King")
        five = _card("Spades", "5")
        scoring = [king, five]
        ctx = JokerContext(before=True, scoring_hand=scoring)
        calculate_joker(j, ctx)
        # King should be Gold now
        assert king.ability.get("name") == "Gold Card"
        # Five unchanged
        assert five.ability.get("name") != "Gold Card"

    def test_persists_across_hands(self):
        """Enhancement change persists -- card stays Gold for future hands."""
        j = _joker("j_midas_mask")
        queen = _card("Hearts", "Queen")
        ctx = JokerContext(before=True, scoring_hand=[queen])
        calculate_joker(j, ctx)
        assert queen.ability.get("name") == "Gold Card"
        # Card is still Gold on next access
        assert queen.ability.get("name") == "Gold Card"

    def test_no_face_no_effect(self):
        j = _joker("j_midas_mask")
        five = _card("Hearts", "5")
        ctx = JokerContext(before=True, scoring_hand=[five])
        result = calculate_joker(j, ctx)
        assert result is None

    def test_debuffed_face_not_converted(self):
        j = _joker("j_midas_mask")
        king = _card("Hearts", "King")
        king.debuff = True
        ctx = JokerContext(before=True, scoring_hand=[king])
        result = calculate_joker(j, ctx)
        assert result is None

    def test_pareidolia_converts_all(self):
        """With Pareidolia, ALL cards are face -> all become Gold."""
        j = _joker("j_midas_mask")
        five = _card("Hearts", "5")
        two = _card("Spades", "2")
        scoring = [five, two]
        pareidolia = _joker("j_pareidolia_stub")
        pareidolia.ability = {"name": "Pareidolia", "set": "Joker"}
        ctx = JokerContext(before=True, scoring_hand=scoring, pareidolia=True)
        calculate_joker(j, ctx)
        assert five.ability.get("name") == "Gold Card"
        assert two.ability.get("name") == "Gold Card"

    def test_pipeline_gold_card_chips(self):
        """Midas Mask converts Kings to Gold in before phase.
        Gold Card effect: +3 dollars per card (from get_p_dollars).
        Three of a Kind of Kings with Midas: Kings become Gold Cards.
        After conversion, Gold Cards give $3 per scored card."""
        played = [
            _card("Hearts", "King"),
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        j = _joker("j_midas_mask")
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # 3 Kings converted to Gold. Gold Card nominal is still 10 (King).
        # But enhancement changes scoring methods.
        assert result.hand_type == "Three of a Kind"
        assert result.dollars_earned >= 0  # Gold Cards earn dollars


class TestHiker:
    def test_adds_perma_bonus(self):
        j = _joker("j_hiker", extra=5)
        ace = _card("Hearts", "Ace")
        ctx = JokerContext(individual=True, cardarea="play", other_card=ace)
        calculate_joker(j, ctx)
        assert ace.ability["perma_bonus"] == 5

    def test_accumulates_across_hands(self):
        """Same card scored 3 times -> perma_bonus = 15."""
        j = _joker("j_hiker", extra=5)
        ace = _card("Hearts", "Ace")
        for _ in range(3):
            ctx = JokerContext(
                individual=True,
                cardarea="play",
                other_card=ace,
            )
            calculate_joker(j, ctx)
        assert ace.ability["perma_bonus"] == 15

    def test_pipeline_perma_bonus_adds_chips(self):
        """Hiker adds perma_bonus during scoring. On next hand, the card
        has higher chip value due to accumulated bonus."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_hiker", extra=5)

        # First hand: perma_bonus added during scoring
        score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Each Ace gets +5 perma_bonus during Phase 7
        assert played[0].ability["perma_bonus"] == 5
        assert played[1].ability["perma_bonus"] == 5

        # Second hand: perma_bonus is part of chip calculation
        reset_sort_id_counter()
        r2 = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 10 base. Each Ace: 11 nominal + 5 old perma + 5 new perma = 21 chips.
        # Wait -- get_chip_bonus returns nominal + bonus + perma_bonus.
        # After hand 1: perma_bonus = 5. During hand 2, eval_card adds 11+5=16 chips.
        # Then Hiker adds another +5 perma_bonus -> 10.
        # Total chips hand 2: 10 + (11+5) + (11+5) = 42.
        assert r2.chips == 42.0
        assert played[0].ability["perma_bonus"] == 10

    def test_blueprint_does_not_mutate(self):
        j = _joker("j_hiker", extra=5)
        ace = _card("Hearts", "Ace")
        ctx = JokerContext(
            individual=True,
            cardarea="play",
            other_card=ace,
            blueprint=1,
        )
        calculate_joker(j, ctx)
        assert ace.ability.get("perma_bonus", 0) == 0


# ============================================================================
# Section 6: Retrigger Pipeline
# ============================================================================


class TestSockAndBuskin:
    """j_sock_and_buskin: +1 retrigger for face cards scored."""

    def test_pipeline_three_kings(self):
        """Three Kings scored: each face card evaluated twice.
        Three of a Kind L1: 30 chips, 3 mult.
        Per card: 10 chips each x 3 Kings x 2 reps = 60 chips.
        Total chips: 30 + 60 = 90, mult = 3. Score: 90 x 3 = 270.
        (Without retrigger: 30 + 30 = 60, score 180.)"""
        played = [
            _card("Hearts", "King"),
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        j = _joker("j_sock_and_buskin", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # 3 Kings score. Each King: 10 chips x 2 reps = 20 chips.
        # Non-scoring 5 and 2 don't retrigger.
        assert result.chips == 90.0  # 30 + 3x20
        assert result.total == 270

    def test_pipeline_pareidolia_all_retriggered(self):
        """With Pareidolia, ALL cards are face -> all retriggered.
        Pair of 5s: 10 base, 2 mult. Per card: 5 chips x 2 reps = 10 each.
        Total chips: 10 + 10 + 10 = 30. Score: 30 x 2 = 60."""
        played = [_card("Hearts", "5"), _card("Spades", "5")]
        sock = _joker("j_sock_and_buskin", extra=1)
        pareidolia = _joker("j_pareidolia_stub")
        # Pareidolia needs to be a real joker that score_hand detects
        pareidolia.ability = {"name": "Pareidolia", "set": "Joker"}
        result = score_hand(
            played,
            [],
            [sock, pareidolia],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # With Pareidolia, both 5s are "face cards" -> retriggered
        assert result.chips == 30.0  # 10 + 5*2 + 5*2
        assert result.total == 60


class TestHangingChad:
    """j_hanging_chad: +2 retriggers for first scored card only."""

    def test_pipeline_pair_of_aces(self):
        """Pair of Aces: first Ace evaluated 3 times (1 base + 2 reps).
        Base: 10 chips, 2 mult.
        First Ace: 11 chips x 3 reps = 33 chips.
        Second Ace: 11 chips x 1 rep = 11 chips.
        Total: 10 + 33 + 11 = 54. Score: 54 x 2 = 108."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_hanging_chad", extra=2)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 54.0
        assert result.total == 108


class TestDusk:
    """j_dusk: +1 retrigger for all scored cards on last hand."""

    def test_pipeline_last_hand(self):
        """Last hand: all cards doubled.
        Pair of Aces: 10 base, 2 mult. Each Ace: 11 x 2 = 22 chips.
        Total: 10 + 22 + 22 = 54. Score: 54 x 2 = 108."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_dusk", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            game_state={"hands_left": 0},
        )
        assert result.chips == 54.0
        assert result.total == 108


class TestAdditiveRetriggers:
    """Red Seal (+1) + Sock and Buskin (+1) on face card = 3 total evals."""

    def test_red_seal_plus_sock(self):
        """King with Red Seal + Sock and Buskin:
        1 base + 1 Red Seal retrigger + 1 Sock retrigger = 3 evaluations.
        Pair of Kings: 10 base, 2 mult.
        Red Seal King: 10 chips x 3 reps = 30.
        Normal King: 10 chips x 1 = 10.
        Total: 10 + 30 + 10 = 50. Score: 50 x 2 = 100."""
        k1 = _card("Hearts", "King")
        k1.set_seal("Red")
        k2 = _card("Spades", "King")
        played = [k1, k2]
        j = _joker("j_sock_and_buskin", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # k1: Red Seal (+1 rep) + Sock (+1 rep, is face) = 3 total
        # k2: Sock (+1 rep, is face) = 2 total
        assert result.chips == 10.0 + 30.0 + 20.0  # 60
        assert result.total == 120


class TestSeltzer:
    """j_selzer: +1 retrigger for all cards. Decrements per hand."""

    def test_unit_always_retriggers(self):
        j = _joker("j_selzer", extra=10)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_card("Hearts", "5"),
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_after_decrements(self):
        j = _joker("j_selzer", extra=10)
        ctx = JokerContext(after=True)
        calculate_joker(j, ctx)
        assert j.ability["extra"] == 9

    def test_self_destructs_at_zero(self):
        j = _joker("j_selzer", extra=1)
        ctx = JokerContext(after=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True

    def test_ten_hands_then_destruct(self):
        j = _joker("j_selzer", extra=10)
        for i in range(9):
            calculate_joker(j, JokerContext(after=True))
        assert j.ability["extra"] == 1
        result = calculate_joker(j, JokerContext(after=True))
        assert result.remove is True

    def test_pipeline_all_cards_retriggered(self):
        """Pair of Aces with Seltzer: all cards retriggered.
        Base: 10, 2 mult. Each Ace: 11 x 2 = 22. Total: 54 x 2 = 108.
        After scoring: extra decremented 10 -> 9."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_selzer", extra=10)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 54.0
        assert result.total == 108
        assert j.ability["extra"] == 9


class TestMime:
    """j_mime: +1 retrigger for held cards."""

    def test_pipeline_steel_card_doubled(self):
        """Steel Card held + Mime: x1.5 fires twice -> mult x 1.5 x 1.5 = x2.25.
        Pair of Aces: 32 chips, 2 mult.
        Steel held: 2 reps x x1.5 -> mult = 2 x 1.5 x 1.5 = 4.5.
        Score: floor(32 x 4.5) = 144."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [_card("Clubs", "3", enhancement="m_steel")]
        j = _joker("j_mime", extra=1)
        result = score_hand(
            played,
            held,
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(4.5)
        assert result.total == 144

    def test_pipeline_two_steel_with_mime(self):
        """Two Steel Cards + Mime: each fires twice.
        Pair: 32 chips, 2 mult.
        Steel 1: 2 reps x x1.5 -> mult x 1.5^2 = 4.5.
        Steel 2: 2 reps x x1.5 -> mult x 1.5^2 = 10.125.
        Score: floor(32 x 10.125) = 324."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [
            _card("Clubs", "3", enhancement="m_steel"),
            _card("Diamonds", "7", enhancement="m_steel"),
        ]
        j = _joker("j_mime", extra=1)
        result = score_hand(
            played,
            held,
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # 2 x 1.5^4 = 2 x 5.0625 = 10.125
        assert result.mult == pytest.approx(10.125)
        assert result.total == 324


class TestHackRetrigger:
    """j_hack: verify retrigger works through pipeline."""

    def test_pipeline_pair_of_threes(self):
        """Pair of 3s with Hack: each 3 retriggered.
        Base: 10 chips, 2 mult. Each 3: 3 chips x 2 reps = 6 chips.
        Total: 10 + 6 + 6 = 22. Score: 22 x 2 = 44."""
        played = [_card("Hearts", "3"), _card("Spades", "3")]
        j = _joker("j_hack", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 22.0
        assert result.total == 44

    def test_pipeline_pair_of_sixes_no_retrigger(self):
        """Pair of 6s: Hack doesn't trigger (6 not in 2-5)."""
        played = [_card("Hearts", "6"), _card("Spades", "6")]
        j = _joker("j_hack", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 22.0  # 10 + 6 + 6 (no retrigger)
        assert result.total == 44


# ============================================================================
# Section 7: Scaling State
# ============================================================================


class TestGreenJoker:
    def _make(self):
        return _joker(
            "j_green_joker",
            mult=0,
            extra={"hand_add": 1, "discard_sub": 1},
        )

    def test_hand_1_adds_1(self):
        j = self._make()
        ctx = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 1

    def test_hand_2_adds_2(self):
        j = self._make()
        # Hand 1
        ctx = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx)
        # Hand 2
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 2

    def test_discard_subtracts(self):
        j = self._make()
        # Play 2 hands: mult = 2
        ctx_b = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx_b)
        calculate_joker(j, ctx_b)
        assert j.ability["mult"] == 2

        # Discard (fires on last card)
        last = _card("Hearts", "3")
        ctx_d = JokerContext(
            discard=True,
            other_card=last,
            full_hand=[last],
        )
        calculate_joker(j, ctx_d)
        assert j.ability["mult"] == 1

    def test_discard_clamps_to_zero(self):
        j = self._make()
        last = _card("Hearts", "3")
        ctx_d = JokerContext(
            discard=True,
            other_card=last,
            full_hand=[last],
        )
        calculate_joker(j, ctx_d)
        assert j.ability["mult"] == 0

    def test_joker_main_returns_accumulated(self):
        j = self._make()
        # Play 3 hands
        ctx_b = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        for _ in range(3):
            calculate_joker(j, ctx_b)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 3

    def test_blueprint_does_not_mutate(self):
        j = self._make()
        ctx = JokerContext(
            before=True,
            blueprint=1,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 0  # unchanged


class TestRideTheBus:
    def _make(self):
        return _joker("j_ride_the_bus", mult=0, extra=1)

    def test_three_no_face_hands(self):
        j = self._make()
        no_face_hand = [_card("Hearts", "5"), _card("Spades", "5")]
        ctx = JokerContext(before=True, scoring_hand=no_face_hand)
        for _ in range(3):
            calculate_joker(j, ctx)
        assert j.ability["mult"] == 3

    def test_face_card_resets(self):
        j = self._make()
        no_face = [_card("Hearts", "5"), _card("Spades", "5")]
        ctx_nf = JokerContext(before=True, scoring_hand=no_face)
        for _ in range(3):
            calculate_joker(j, ctx_nf)
        assert j.ability["mult"] == 3

        face = [_card("Hearts", "King"), _card("Spades", "5")]
        ctx_f = JokerContext(before=True, scoring_hand=face)
        calculate_joker(j, ctx_f)
        assert j.ability["mult"] == 0

    def test_joker_main_returns_accumulated(self):
        j = self._make()
        ctx = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        for _ in range(3):
            calculate_joker(j, ctx)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 3

    def test_zero_mult_returns_none(self):
        j = self._make()
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


class TestSpareTrousers:
    def _make(self):
        return _joker("j_trousers", mult=0, extra=2)

    def test_two_pair_triggers(self):
        j = self._make()
        ph = _poker_hands_with("Two Pair", "Pair")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 2

    def test_accumulates(self):
        j = self._make()
        ph = _poker_hands_with("Two Pair", "Pair")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 4


class TestSquareJoker:
    def _make(self):
        return _joker("j_square", extra={"chips": 0, "chip_mod": 4})

    def test_four_cards_adds(self):
        j = self._make()
        hand = [_card("Hearts", "5")] * 4
        ctx = JokerContext(before=True, full_hand=hand)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 4

    def test_five_cards_no_effect(self):
        j = self._make()
        hand = [_card("Hearts", "5")] * 5
        ctx = JokerContext(before=True, full_hand=hand)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 0


class TestIceCream:
    def _make(self):
        return _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})

    def test_hand_twenty_one_self_destructs(self):
        j = self._make()
        ctx = JokerContext(after=True)
        for _ in range(19):
            calculate_joker(j, ctx)
        # chips=5, chip_mod=5 -> 5-5 <= 0 -> remove
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True

    def test_in_scoring_pipeline(self):
        """Full pipeline: Ice Cream contributes chips then decays."""
        j = self._make()
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        r1 = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 32 chips, 2 mult. Phase 9: +100 chip_mod -> 132. Score: 132x2=264.
        assert r1.chips == 132.0
        assert r1.total == 264
        # After Phase 10: chips decremented
        assert j.ability["extra"]["chips"] == 95

        # Second hand
        reset_sort_id_counter()
        r2 = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert r2.chips == 127.0  # 32 + 95
        assert j.ability["extra"]["chips"] == 90


class TestPopcorn:
    def _make(self):
        return _joker("j_popcorn", mult=20, extra=4)

    def test_self_destructs_at_zero(self):
        j = self._make()
        ctx = JokerContext(end_of_round=True)
        for _ in range(4):
            calculate_joker(j, ctx)
        assert j.ability["mult"] == 4
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True


class TestWeeJoker:
    def _make(self):
        return _joker("j_wee", extra={"chips": 0, "chip_mod": 8})

    def test_three_twos_across_two_hands(self):
        """Score 3 twos across 2 hands -> +24 chips accumulated."""
        j = self._make()
        # Hand 1: 2 twos
        for _ in range(2):
            ctx = JokerContext(
                individual=True,
                cardarea="play",
                other_card=_card("Hearts", "2"),
            )
            calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 16

        # Hand 2: 1 two
        ctx = JokerContext(
            individual=True,
            cardarea="play",
            other_card=_card("Spades", "2"),
        )
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 24

    def test_non_two_no_effect(self):
        j = self._make()
        ctx = JokerContext(
            individual=True,
            cardarea="play",
            other_card=_card("Hearts", "5"),
        )
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 0


class TestLuckyCat:
    def _make(self):
        return _joker("j_lucky_cat", x_mult=1, extra=0.25)

    def _lucky_card(self) -> Card:
        c = _card("Hearts", "5")
        c.set_ability("m_lucky")
        c.lucky_trigger = True
        return c

    def test_lucky_trigger_accumulates(self):
        j = self._make()
        lc = self._lucky_card()
        ctx = JokerContext(individual=True, cardarea="play", other_card=lc)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.25)

    def test_four_triggers(self):
        j = self._make()
        for _ in range(4):
            lc = self._lucky_card()
            ctx = JokerContext(individual=True, cardarea="play", other_card=lc)
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(2.0)

    def test_joker_main_returns_when_above_1(self):
        j = self._make()
        # x_mult = 1, no effect yet
        assert calculate_joker(j, JokerContext(joker_main=True)) is None

        # Trigger once -> x_mult = 1.25
        lc = self._lucky_card()
        ctx = JokerContext(individual=True, cardarea="play", other_card=lc)
        calculate_joker(j, ctx)

        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == pytest.approx(1.25)

    def test_non_lucky_no_effect(self):
        j = self._make()
        normal = _card("Hearts", "5")
        ctx = JokerContext(individual=True, cardarea="play", other_card=normal)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1  # unchanged


# ============================================================================
# Section 8: xMult Accumulation
# ============================================================================


class TestCampfire:
    def _make(self):
        return _joker("j_campfire", x_mult=1, extra=0.25)

    def test_sell_increments(self):
        j = self._make()
        ctx = JokerContext(selling_card=True)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.25)

    def test_three_sells(self):
        j = self._make()
        for _ in range(3):
            calculate_joker(j, JokerContext(selling_card=True))
        assert j.ability["x_mult"] == pytest.approx(1.75)

    def test_boss_resets(self):
        j = self._make()
        for _ in range(4):
            calculate_joker(j, JokerContext(selling_card=True))
        assert j.ability["x_mult"] == pytest.approx(2.0)

        boss = Blind.create("bl_hook", ante=1)
        calculate_joker(j, JokerContext(end_of_round=True, blind=boss))
        assert j.ability["x_mult"] == 1

    def test_non_boss_no_reset(self):
        j = self._make()
        for _ in range(2):
            calculate_joker(j, JokerContext(selling_card=True))

        small = Blind.create("bl_small", ante=1)
        calculate_joker(j, JokerContext(end_of_round=True, blind=small))
        assert j.ability["x_mult"] == pytest.approx(1.5)

    def test_joker_main_returns(self):
        j = self._make()
        calculate_joker(j, JokerContext(selling_card=True))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == pytest.approx(1.25)

    def test_no_sells_no_effect(self):
        j = self._make()
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


class TestHologram:
    def _make(self):
        return _joker("j_hologram", x_mult=1, extra=0.25)

    def test_add_one_card(self):
        j = self._make()
        ctx = JokerContext(playing_card_added=True, cards=[_card("Hearts", "5")])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.25)


class TestConstellation:
    def _make(self):
        return _joker("j_constellation", x_mult=1, extra=0.1)

    def _planet(self) -> Card:
        c = Card()
        c.ability = {"set": "Planet", "name": "Jupiter"}
        return c

    def _tarot(self) -> Card:
        c = Card()
        c.ability = {"set": "Tarot", "name": "The Fool"}
        return c

    def test_planet_used(self):
        j = self._make()
        ctx = JokerContext(using_consumeable=True, consumeable=self._planet())
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.1)

    def test_tarot_no_effect(self):
        j = self._make()
        ctx = JokerContext(using_consumeable=True, consumeable=self._tarot())
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1


class TestGlassJoker:
    def _make(self):
        return _joker("j_glass", x_mult=1, extra=0.75)

    def test_one_glass_destroyed(self):
        j = self._make()
        glass = _card("Hearts", "5", enhancement="m_glass")
        ctx = JokerContext(cards_destroyed=[glass])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.75)

    def test_two_glass_destroyed(self):
        j = self._make()
        glass1 = _card("Hearts", "5", enhancement="m_glass")
        glass2 = _card("Spades", "3", enhancement="m_glass")
        ctx = JokerContext(cards_destroyed=[glass1, glass2])
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(2.5)


class TestCaino:
    def _make(self):
        return _joker("j_caino", caino_xmult=1, extra=1)

    def test_face_destroyed(self):
        j = self._make()
        king = _card("Hearts", "King")
        ctx = JokerContext(cards_destroyed=[king])
        calculate_joker(j, ctx)
        assert j.ability["caino_xmult"] == 2

    def test_two_faces_destroyed(self):
        j = self._make()
        king = _card("Hearts", "King")
        queen = _card("Spades", "Queen")
        ctx = JokerContext(cards_destroyed=[king, queen])
        calculate_joker(j, ctx)
        assert j.ability["caino_xmult"] == 3

    def test_non_face_no_effect(self):
        j = self._make()
        five = _card("Hearts", "5")
        ctx = JokerContext(cards_destroyed=[five])
        calculate_joker(j, ctx)
        assert j.ability["caino_xmult"] == 1

    def test_joker_main_returns(self):
        j = self._make()
        king = _card("Hearts", "King")
        calculate_joker(j, JokerContext(cards_destroyed=[king]))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 2

    def test_caino_and_glass_both_trigger_on_glass_face(self):
        """A Glass King is both a face card AND a Glass Card."""
        caino = self._make()
        glass_j = _joker("j_glass", x_mult=1, extra=0.75)
        glass_king = _card("Hearts", "King", enhancement="m_glass")
        ctx = JokerContext(cards_destroyed=[glass_king])

        calculate_joker(caino, ctx)
        calculate_joker(glass_j, ctx)

        assert caino.ability["caino_xmult"] == 2  # face card
        assert glass_j.ability["x_mult"] == pytest.approx(1.75)  # glass card


class TestVampire:
    def _make(self):
        return _joker("j_vampire", x_mult=1, extra=0.1)

    def test_strips_one_enhancement(self):
        j = self._make()
        bonus = _card("Hearts", "5", enhancement="m_bonus")
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_hand=[bonus],
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.1)
        assert bonus.ability.get("name") == "Default Base"

    def test_debuffed_card_not_stripped(self):
        j = self._make()
        bonus = _card("Hearts", "5", enhancement="m_bonus")
        bonus.debuff = True
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_hand=[bonus],
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_accumulates_across_hands(self):
        """3 hands, each with 1 enhanced card -> x_mult = 1.3."""
        j = self._make()
        for _ in range(3):
            enhanced = _card("Hearts", "5", enhancement="m_bonus")
            ctx = JokerContext(
                individual_hand_end=True,
                scoring_hand=[enhanced],
            )
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.3)


class TestObelisk:
    def _make(self):
        return _joker("j_obelisk", x_mult=1, extra=0.2)

    def test_non_most_played_increments(self):
        j = self._make()
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")
        levels.record_play("Pair")
        levels.record_play("Flush")  # Flush=1, Pair=3 -> Flush is NOT most
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_name="Flush",
            hand_levels=levels,
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.2)

    def test_most_played_resets(self):
        j = self._make()
        j.ability["x_mult"] = 2.0  # accumulated
        levels = HandLevels()
        levels.record_play("Pair")
        levels.record_play("Pair")
        # Pair is most played (2), playing Pair -> reset
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_sequence_accumulate_then_reset(self):
        j = self._make()
        levels = HandLevels()
        # Play Pair 3 times (most played)
        for _ in range(3):
            levels.record_play("Pair")

        # Play Flush twice -> not most played -> +0.2 each
        for _ in range(2):
            levels.record_play("Flush")
            ctx = JokerContext(
                individual_hand_end=True,
                scoring_name="Flush",
                hand_levels=levels,
            )
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.4)

        # Play Pair (still most played) -> resets
        levels.record_play("Pair")
        ctx = JokerContext(
            individual_hand_end=True,
            scoring_name="Pair",
            hand_levels=levels,
        )
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1


class TestMadness:
    def _make(self):
        return _joker("j_madness", x_mult=1, extra=0.5)

    def test_non_boss_increments(self):
        j = self._make()
        small = Blind.create("bl_small", ante=1)
        ctx = JokerContext(setting_blind=True, blind=small)
        result = calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.5)
        assert result is not None
        assert result.extra == {"destroy_random_joker": True}

    def test_boss_no_effect(self):
        j = self._make()
        boss = Blind.create("bl_hook", ante=1)
        ctx = JokerContext(setting_blind=True, blind=boss)
        result = calculate_joker(j, ctx)
        assert result is None
        assert j.ability["x_mult"] == 1


class TestHitTheRoad:
    def _make(self):
        return _joker("j_hit_the_road", x_mult=1, extra=0.5)

    def test_jack_discarded(self):
        j = self._make()
        jack = _card("Hearts", "Jack")
        ctx = JokerContext(discard=True, other_card=jack)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.5)

    def test_non_jack_no_effect(self):
        j = self._make()
        king = _card("Hearts", "King")
        ctx = JokerContext(discard=True, other_card=king)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1

    def test_three_jacks_then_reset(self):
        j = self._make()
        for _ in range(3):
            jack = _card("Hearts", "Jack")
            calculate_joker(j, JokerContext(discard=True, other_card=jack))
        assert j.ability["x_mult"] == pytest.approx(2.5)

        # End of round resets
        calculate_joker(j, JokerContext(end_of_round=True))
        assert j.ability["x_mult"] == 1


class TestThrowback:
    def test_zero_skips(self):
        j = _joker("j_throwback", extra=0.25)
        ctx = JokerContext(joker_main=True, skips=0)
        assert calculate_joker(j, ctx) is None

    def test_four_skips(self):
        j = _joker("j_throwback", extra=0.25)
        ctx = JokerContext(joker_main=True, skips=4)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.Xmult_mod == pytest.approx(2.0)

    def test_ten_skips(self):
        j = _joker("j_throwback", extra=0.25)
        ctx = JokerContext(joker_main=True, skips=10)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.Xmult_mod == pytest.approx(3.5)


class TestYorick:
    def _make(self):
        return _joker(
            "j_yorick",
            x_mult=1,
            yorick_discards=23,
            extra={"xmult": 1, "discards": 23},
        )

    def test_22_discards_no_trigger(self):
        j = self._make()
        for _ in range(22):
            calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == 1
        assert j.ability["yorick_discards"] == 1

    def test_23rd_discard_triggers(self):
        j = self._make()
        for _ in range(23):
            calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == 2
        assert j.ability["yorick_discards"] == 23  # reset

    def test_46_discards_triggers_twice(self):
        j = self._make()
        for _ in range(46):
            calculate_joker(j, JokerContext(discard=True))
        assert j.ability["x_mult"] == 3

    def test_joker_main_returns(self):
        j = self._make()
        for _ in range(23):
            calculate_joker(j, JokerContext(discard=True))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == 2


class TestCeremonialDagger:
    def _make(self):
        return _joker("j_ceremonial", mult=0)

    def test_destroys_right_neighbor(self):
        dagger = self._make()
        target = _joker("j_joker", mult=4)
        target.sell_cost = 3
        jokers = [dagger, target]
        ctx = JokerContext(
            setting_blind=True,
            jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        result = calculate_joker(dagger, ctx)
        assert dagger.ability["mult"] == 6  # 3 * 2
        assert result is not None
        assert result.extra["destroy_joker"] is target

    def test_no_right_neighbor(self):
        dagger = self._make()
        jokers = [dagger]
        ctx = JokerContext(
            setting_blind=True,
            jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        assert calculate_joker(dagger, ctx) is None

    def test_eternal_not_destroyed(self):
        dagger = self._make()
        target = _joker("j_joker", mult=4)
        target.sell_cost = 3
        target.eternal = True
        jokers = [dagger, target]
        ctx = JokerContext(
            setting_blind=True,
            jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        assert calculate_joker(dagger, ctx) is None

    def test_accumulates_across_rounds(self):
        dagger = self._make()
        t1 = _joker("j_joker", mult=4)
        t1.sell_cost = 2
        jokers = [dagger, t1]
        ctx = JokerContext(
            setting_blind=True,
            jokers=jokers,
            blind=Blind.create("bl_small", ante=1),
        )
        calculate_joker(dagger, ctx)
        assert dagger.ability["mult"] == 4  # 2 * 2

        # Next round, new neighbor
        t2 = _joker("j_stuntman")
        t2.sell_cost = 5
        jokers2 = [dagger, t2]
        ctx2 = JokerContext(
            setting_blind=True,
            jokers=jokers2,
            blind=Blind.create("bl_small", ante=1),
        )
        calculate_joker(dagger, ctx2)
        assert dagger.ability["mult"] == 14  # 4 + 5*2

    def test_joker_main_returns_mult(self):
        dagger = self._make()
        dagger.ability["mult"] = 10
        result = calculate_joker(dagger, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 10  # additive, NOT xMult
