"""Cross-validation: Python scoring pipeline vs Lua source.

Loads pre-generated Lua scoring output and verifies the Python
score_hand_base produces identical chips/mult/total values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand, score_hand_base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "scoring_oracle.json"


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
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
    c = Card()
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


# ============================================================================
# Reproduce each Lua scenario in Python
# ============================================================================


def _run_pair_aces_basic():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    return score_hand_base(
        played,
        [],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_three_kings_foil():
    reset_sort_id_counter()
    k1 = _card("Hearts", "King")
    k1.set_edition({"foil": True})
    played = [
        k1,
        _card("Spades", "King"),
        _card("Clubs", "King"),
        _card("Diamonds", "5"),
        _card("Hearts", "2"),
    ]
    return score_hand_base(
        played,
        [],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_flush_glass():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "2"),
        _card("Hearts", "5"),
        _card("Hearts", "8"),
        _card("Hearts", "Jack"),
        _card("Hearts", "Ace", enhancement="m_glass"),
    ]
    return score_hand_base(
        played,
        [],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_full_house_steel_held():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "King"),
        _card("Spades", "King"),
        _card("Clubs", "King"),
        _card("Diamonds", "5"),
        _card("Hearts", "5"),
    ]
    held = [_card("Clubs", "3", enhancement="m_steel")]
    return score_hand_base(
        played,
        held,
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_aces_red_seal():
    reset_sort_id_counter()
    c = _card("Hearts", "Ace")
    c.set_seal("Red")
    played = [c, _card("Spades", "Ace")]
    return score_hand_base(
        played,
        [],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_aces_flint():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    return score_hand_base(
        played,
        [],
        HandLevels(),
        Blind.create("bl_flint", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_eye_debuffed():
    reset_sort_id_counter()
    levels = HandLevels()
    blind = Blind.create("bl_eye", ante=1)

    # First pair: allowed (registers "Pair" in The Eye)
    p1 = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    score_hand_base(p1, [], levels, blind, PseudoRandom("ORACLE"))

    # Second pair: blocked
    reset_sort_id_counter()
    p2 = [_card("Hearts", "King"), _card("Spades", "King")]
    return score_hand_base(p2, [], levels, blind, PseudoRandom("ORACLE"))


def _joker(center_key: str, **ability_kw) -> Card:
    """Create a minimal joker Card for oracle testing."""
    c = Card()
    c.center_key = center_key
    c.ability = {"name": center_key, "set": "Joker", **ability_kw}
    return c


# --- Joker scenarios (use score_hand) ---


def _run_pair_aces_joker():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    j = _joker("j_joker", mult=4)
    return score_hand(
        played,
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_flush_hearts_lusty():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "2"),
        _card("Hearts", "5"),
        _card("Hearts", "8"),
        _card("Hearts", "Jack"),
        _card("Hearts", "Ace"),
    ]
    j = _joker("j_lusty_joker", extra={"s_mult": 3, "suit": "Hearts"})
    return score_hand(
        played,
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_three_kings_scary_face():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "King"),
        _card("Spades", "King"),
        _card("Clubs", "King"),
        _card("Diamonds", "5"),
        _card("Hearts", "2"),
    ]
    j = _joker("j_scary_face", extra=30)
    return score_hand(
        played,
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_full_house_jolly():
    reset_sort_id_counter()
    played = [
        _card("Hearts", "King"),
        _card("Spades", "King"),
        _card("Clubs", "King"),
        _card("Diamonds", "5"),
        _card("Hearts", "5"),
    ]
    j = _joker("j_jolly", t_mult=8, type="Pair")
    return score_hand(
        played,
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_pair_duo_xmult():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    j = _joker("j_duo", x_mult=2, type="Pair")
    return score_hand(
        played,
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_joker_then_blackboard():
    reset_sort_id_counter()
    played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
    held = [_card("Spades", "5")]
    j1 = _joker("j_joker", mult=4)
    j2 = _joker("j_blackboard", extra=3)
    return score_hand(
        played,
        held,
        [j1, j2],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_blackboard_then_joker():
    reset_sort_id_counter()
    played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
    held = [_card("Spades", "5")]
    j1 = _joker("j_blackboard", extra=3)
    j2 = _joker("j_joker", mult=4)
    return score_hand(
        played,
        held,
        [j1, j2],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_foil_joker():
    reset_sort_id_counter()
    played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
    j = _joker("j_joker", mult=4)
    j.set_edition({"foil": True})
    return score_hand(
        played,
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


# --- Complex joker interaction scenarios ---


def _run_green_joker_4_hands():
    """Green Joker accumulates +1 per hand over 4 hands → mult=4."""
    green = _joker("j_green_joker", mult=0, extra={"hand_add": 1, "discard_sub": 1})
    blind = Blind.create("bl_small", ante=1)
    # Play 3 prior hands to accumulate
    for _ in range(3):
        reset_sort_id_counter()
        score_hand(
            [_card("Hearts", "5"), _card("Spades", "5")],
            [],
            [green],
            HandLevels(),
            blind,
            PseudoRandom("ORACLE"),
        )
    # 4th hand: Green fires before (+1 → total 4), then joker_main returns 4
    reset_sort_id_counter()
    return score_hand(
        [_card("Hearts", "Ace"), _card("Spades", "Ace")],
        [],
        [green],
        HandLevels(),
        blind,
        PseudoRandom("ORACLE"),
    )


def _run_steel_joker_2_steel():
    reset_sort_id_counter()
    j = _joker("j_steel_joker", extra=0.2)
    return score_hand(
        [_card("Hearts", "Ace"), _card("Spades", "Ace")],
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
        game_state={"steel_tally": 2},
    )


def _run_sock_buskin_3_kings():
    reset_sort_id_counter()
    j = _joker("j_sock_and_buskin", extra=1)
    return score_hand(
        [
            _card("Hearts", "King"),
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ],
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_red_seal_dusk_last_hand():
    reset_sort_id_counter()
    red_ace = _card("Hearts", "Ace")
    red_ace.set_seal("Red")
    j = _joker("j_dusk", extra=1)
    return score_hand(
        [red_ace, _card("Spades", "Ace")],
        [],
        [j],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
        game_state={"hands_left": 0},
    )


def _run_blueprint_green_joker():
    reset_sort_id_counter()
    green = _joker("j_green_joker", mult=5, extra={"hand_add": 1, "discard_sub": 1})
    bp = _joker("j_blueprint")
    jokers = [bp, green]
    return score_hand(
        [_card("Hearts", "Ace"), _card("Spades", "Ace")],
        [],
        jokers,
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_baseball_2_uncommon():
    reset_sort_id_counter()
    steel = _joker("j_steel_joker", extra=0.2)
    bb = _joker("j_blackboard", extra=3)
    baseball = _joker("j_baseball", extra=1.5)
    jokers = [steel, bb, baseball]
    return score_hand(
        [_card("Spades", "Ace"), _card("Clubs", "Ace")],
        [_card("Spades", "5")],
        jokers,
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
        game_state={"steel_tally": 1},
    )


def _run_glass_king_caino():
    reset_sort_id_counter()
    glass_king = _card("Hearts", "King", enhancement="m_glass")
    caino = _joker("j_caino", caino_xmult=1, extra=1)
    return score_hand(
        [
            glass_king,
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ],
        [],
        [caino],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
        probabilities_normal=100.0,  # force shatter
    )


def _run_ice_cream_hand1():
    reset_sort_id_counter()
    ice = _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})
    return score_hand(
        [_card("Hearts", "Ace"), _card("Spades", "Ace")],
        [],
        [ice],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


def _run_ice_cream_hand2():
    """Ice Cream after 1 prior hand: chips=95."""
    reset_sort_id_counter()
    ice = _joker("j_ice_cream", extra={"chips": 95, "chip_mod": 5})
    return score_hand(
        [_card("Hearts", "Ace"), _card("Spades", "Ace")],
        [],
        [ice],
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("ORACLE"),
    )


# Map test names to Python runners
RUNNERS = {
    "pair_aces_basic": _run_pair_aces_basic,
    "three_kings_foil": _run_three_kings_foil,
    "flush_glass": _run_flush_glass,
    "full_house_steel_held": _run_full_house_steel_held,
    "pair_aces_red_seal": _run_pair_aces_red_seal,
    "pair_aces_flint": _run_pair_aces_flint,
    "pair_eye_debuffed": _run_pair_eye_debuffed,
    "pair_aces_joker": _run_pair_aces_joker,
    "flush_hearts_lusty": _run_flush_hearts_lusty,
    "three_kings_scary_face": _run_three_kings_scary_face,
    "full_house_jolly": _run_full_house_jolly,
    "pair_duo_xmult": _run_pair_duo_xmult,
    "joker_then_blackboard": _run_joker_then_blackboard,
    "blackboard_then_joker": _run_blackboard_then_joker,
    "foil_joker": _run_foil_joker,
    "green_joker_4_hands": _run_green_joker_4_hands,
    "steel_joker_2_steel": _run_steel_joker_2_steel,
    "sock_buskin_3_kings": _run_sock_buskin_3_kings,
    "red_seal_dusk_last_hand": _run_red_seal_dusk_last_hand,
    "blueprint_green_joker": _run_blueprint_green_joker,
    "baseball_2_uncommon": _run_baseball_2_uncommon,
    "glass_king_caino": _run_glass_king_caino,
    "ice_cream_hand1": _run_ice_cream_hand1,
    "ice_cream_hand2": _run_ice_cream_hand2,
}


# ============================================================================
# Fixture-based cross-validation
# ============================================================================


class TestScoringOracle:
    """Validate Python scoring against Lua ground truth."""

    @pytest.fixture(scope="class")
    def fixture(self) -> list[dict]:
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
        with open(FIXTURE_PATH) as f:
            return json.load(f)["tests"]

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_total_matches(self, fixture, test_name):
        """floor(chips * mult) must match Lua exactly."""
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.total == lua_test["total"], (
            f"{test_name}: Python total={py_result.total}, Lua total={lua_test['total']}"
        )

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_chips_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.chips == pytest.approx(lua_test["chips"], abs=0.01), (
            f"{test_name}: Python chips={py_result.chips}, Lua chips={lua_test['chips']}"
        )

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_mult_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.mult == pytest.approx(lua_test["mult"], abs=0.01), (
            f"{test_name}: Python mult={py_result.mult}, Lua mult={lua_test['mult']}"
        )

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_hand_type_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.hand_type == lua_test["hand_type"]

    @pytest.mark.parametrize("test_name", list(RUNNERS.keys()))
    def test_debuffed_match(self, fixture, test_name):
        lua_test = next(t for t in fixture if t["name"] == test_name)
        py_result = RUNNERS[test_name]()
        assert py_result.debuffed == lua_test["debuffed"]
