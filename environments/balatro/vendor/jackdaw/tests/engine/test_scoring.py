"""Scoring pipeline tests.

Consolidated tests covering score_hand_base (Phases 1-8, 12) and
score_hand (Phases 1-14) including joker effects, editions, retriggers,
boss blinds, Plasma Deck, Glass destruction, and joker decay/save.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import ScoreResult, score_hand, score_hand_base


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
    return c


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# 1. Baseline arithmetic (score_hand_base)
# ============================================================================


class TestBaseline:
    """Plain pair of Aces, level 1 — the reference score."""

    def test_pair_of_aces(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        result = score_hand_base(
            played,
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "Pair"
        assert result.chips == 32.0  # 10 + 11 + 11
        assert result.mult == 2.0
        assert result.total == 64
        assert result.debuffed is False
        assert isinstance(result, ScoreResult)
        assert isinstance(result.breakdown, list)
        assert len(result.breakdown) > 0


# ============================================================================
# 2. Enhancements — one test per type that affects scoring
# ============================================================================


class TestEnhancements:
    def test_bonus_card(self):
        """Bonus: +30 chips."""
        played = [_card("Hearts", "5", "m_bonus"), _card("Spades", "5")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.chips == 50.0  # 10 + 35 + 5
        assert r.total == 100

    def test_mult_card(self):
        """Mult Card: +4 mult."""
        played = [_card("Hearts", "5", "m_mult"), _card("Spades", "5")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == 6.0  # 2 + 4
        assert r.total == 120

    def test_glass_card(self):
        """Glass Card: x2 mult when scored."""
        glass_5 = _card("Hearts", "5", "m_glass")
        played = [glass_5, _card("Spades", "5")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == 4.0  # 2 × 2
        assert r.total == 80

    def test_steel_card_held(self):
        """Steel Card: x1.5 mult when held."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [_card("Diamonds", "5", "m_steel")]
        r = score_hand_base(played, held, HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == pytest.approx(3.0)  # 2 × 1.5
        assert r.total == 96

    def test_stone_card(self):
        """Stone Card: 50 chips, ignores rank nominal."""
        sc = _card("Hearts", "Ace", "m_stone")
        played = [_card("Hearts", "5"), _card("Spades", "5"), sc]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        # Pair: 10 base + 5 + 5 = 20 from pair. Stone adds 50.
        assert r.chips == 70.0
        assert r.total == 140


# ============================================================================
# 3. Editions — one test per type
# ============================================================================


class TestEditions:
    def test_foil(self):
        """Foil: +50 chips from edition."""
        c = _card("Hearts", "Ace")
        c.set_edition({"foil": True})
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.chips == 82.0  # 32 + 50
        assert r.total == 164

    def test_holo(self):
        """Holo: +10 mult from edition."""
        c = _card("Hearts", "Ace")
        c.set_edition({"holo": True})
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == 12.0  # 2 + 10

    def test_polychrome(self):
        """Polychrome: x1.5 mult from edition."""
        c = _card("Hearts", "Ace")
        c.set_edition({"polychrome": True})
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == pytest.approx(3.0)  # 2 × 1.5


# ============================================================================
# 4. Red Seal retrigger
# ============================================================================


class TestRedSealRetrigger:
    def test_effects_double(self):
        """Red Seal: card evaluated twice — chips from both reps."""
        c = _card("Hearts", "Ace")
        c.set_seal("Red")
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        # 10 + 11×2 + 11 = 43 chips, 2 mult
        assert r.chips == 43.0
        assert r.total == 86


# ============================================================================
# 5. Boss blinds
# ============================================================================


class TestBossBlind:
    def test_flint_halves(self):
        """The Flint: halves base chips and mult."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        blind = Blind.create("bl_flint", ante=1)
        r = score_hand_base(played, [], HandLevels(), blind, PseudoRandom("T"))
        # Pair base 10→5 chips, 2→1 mult; per card +22
        assert r.chips == 27.0
        assert r.mult == 1.0
        assert r.total == 27

    def test_eye_debuffs_repeat_hand(self):
        """The Eye: repeat hand type → debuffed (score = 0)."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        blind = Blind.create("bl_eye", ante=1)
        # First Pair: allowed
        r1 = score_hand_base(played, [], HandLevels(), blind, PseudoRandom("T"))
        assert r1.debuffed is False
        assert r1.total > 0
        # Second Pair: blocked
        reset_sort_id_counter()
        played2 = [_card("Hearts", "King"), _card("Spades", "King")]
        r2 = score_hand_base(played2, [], HandLevels(), blind, PseudoRandom("T"))
        assert r2.debuffed is True
        assert r2.total == 0


# ============================================================================
# 6. Jokers in pipeline — ordering matters
# ============================================================================


class TestJokerOrdering:
    def test_additive_then_multiplicative(self):
        """j_joker (+4) then j_blackboard (x3): (2+4)×3 = 18."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]  # all black for Blackboard
        joker = _joker("j_joker", mult=4)
        bb = _joker("j_blackboard", extra=3)
        r = score_hand(
            played,
            held,
            [joker, bb],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == pytest.approx(18.0)
        assert r.total == 576

    def test_multiplicative_then_additive(self):
        """j_blackboard (x3) then j_joker (+4): (2×3)+4 = 10. DIFFERENT."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]
        bb = _joker("j_blackboard", extra=3)
        joker = _joker("j_joker", mult=4)
        r = score_hand(
            played,
            held,
            [bb, joker],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == pytest.approx(10.0)
        assert r.total == 320


# ============================================================================
# 7. Joker edition effects
# ============================================================================


class TestJokerEdition:
    def test_foil_adds_chips(self):
        """Foil joker: +50 chips BEFORE joker effect."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"foil": True})
        r = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.chips == 82.0
        assert r.mult == 6.0
        assert r.total == 492

    def test_holo_adds_mult(self):
        """Holo joker: +10 mult BEFORE joker effect."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"holo": True})
        r = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == 16.0
        assert r.total == 512

    def test_polychrome_multiplies_after(self):
        """Polychrome joker: x1.5 AFTER joker effect."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"polychrome": True})
        r = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == pytest.approx(9.0)
        assert r.total == 288


# ============================================================================
# 8. Phase 10: Plasma Deck averaging
# ============================================================================


class TestPlasmaDeck:
    def test_averages_chips_and_mult(self):
        """Plasma Deck: (chips+mult)/2 for both. 32+2=34, each=17. 17×17=289."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        r = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            back_key="b_plasma",
        )
        assert r.chips == 17.0
        assert r.mult == 17.0
        assert r.total == 289


# ============================================================================
# 9. Phase 11: Glass Card shatter + joker reaction
# ============================================================================


class TestGlassDestruction:
    def test_glass_guaranteed_shatter(self):
        """With high probabilities, Glass Card always shatters."""
        glass_5 = _card("Hearts", "5", "m_glass")
        played = [glass_5, _card("Spades", "5")]
        r = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_5 in r.cards_destroyed

    def test_caino_reacts_to_shattered_face(self):
        """Glass King shatters → Caino gains +1 xMult."""
        glass_king = _card("Hearts", "King", "m_glass")
        played = [
            glass_king,
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        caino = _joker("j_caino", caino_xmult=1, extra=1)
        r = score_hand(
            played,
            [],
            [caino],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_king in r.cards_destroyed
        assert caino.ability["caino_xmult"] == 2

    def test_glass_joker_reacts_to_shatter(self):
        """Glass Card shatters → Glass Joker gains +0.75 xMult."""
        glass_5 = _card("Hearts", "5", "m_glass")
        played = [glass_5, _card("Spades", "5")]
        glass_j = _joker("j_glass", x_mult=1, extra=0.75)
        r = score_hand(
            played,
            [],
            [glass_j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_5 in r.cards_destroyed
        assert glass_j.ability["x_mult"] == pytest.approx(1.75)


# ============================================================================
# 10. Phase 13: Ice Cream decay + Mr. Bones save
# ============================================================================


class TestAfterPhase:
    def test_ice_cream_decays(self):
        """Ice Cream: +100 chips in Phase 9, then -5 in Phase 13."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        ice = _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})
        r = score_hand(
            played,
            [],
            [ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.total == 264  # (32+100) × 2
        assert ice.ability["extra"]["chips"] == 95

    def test_ice_cream_self_destructs(self):
        """Ice Cream at 5 chips → decays to 0 → removed."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        ice = _joker("j_ice_cream", extra={"chips": 5, "chip_mod": 5})
        r = score_hand(
            played,
            [],
            [ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert ice in r.jokers_removed

    def test_mr_bones_saves_losing_hand(self):
        """Mr. Bones saves when score < blind_chips and hands_left == 0."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bones = _joker("j_mr_bones")
        r = score_hand(
            played,
            [],
            [bones],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            blind_chips=300,
            game_state={"hands_left": 0},
        )
        assert r.total == 64
        assert r.saved is True
        assert bones in r.jokers_removed
