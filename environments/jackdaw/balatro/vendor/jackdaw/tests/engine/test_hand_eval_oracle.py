"""Cross-validation: Python hand evaluator vs Lua source.

Loads pre-generated Lua output from tests/fixtures/hand_eval_oracle.json
and verifies the Python evaluate_poker_hand produces identical results.
Optionally runs the Lua oracle live via LuaJIT subprocess.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.hand_eval import evaluate_hand

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "hand_eval_oracle.json"
ORACLE_SCRIPT = PROJECT_ROOT / "scripts" / "lua_hand_eval_oracle.lua"


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


# Map test name → (hand_cards, joker_keys)
# Must match the order and cards in lua_hand_eval_oracle.lua
TEST_HANDS: dict[str, tuple[list[Card], list[str]]] = {}


def _build_test_hands() -> dict[str, tuple[list[Card], list[str]]]:
    """Build the same hands as the Lua oracle (must match exactly)."""
    reset_sort_id_counter()
    return {
        "high_card": (
            [
                _card("Hearts", "2"),
                _card("Spades", "5"),
                _card("Clubs", "8"),
                _card("Diamonds", "Jack"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "pair": (
            [
                _card("Hearts", "5"),
                _card("Spades", "5"),
                _card("Clubs", "8"),
                _card("Diamonds", "Jack"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "two_pair": (
            [
                _card("Hearts", "5"),
                _card("Spades", "5"),
                _card("Clubs", "Jack"),
                _card("Diamonds", "Jack"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "three_of_a_kind": (
            [
                _card("Hearts", "King"),
                _card("Spades", "King"),
                _card("Clubs", "King"),
                _card("Diamonds", "5"),
                _card("Hearts", "2"),
            ],
            [],
        ),
        "straight": (
            [
                _card("Hearts", "4"),
                _card("Spades", "5"),
                _card("Clubs", "6"),
                _card("Diamonds", "7"),
                _card("Hearts", "8"),
            ],
            [],
        ),
        "ace_low_straight": (
            [
                _card("Hearts", "Ace"),
                _card("Spades", "2"),
                _card("Clubs", "3"),
                _card("Diamonds", "4"),
                _card("Hearts", "5"),
            ],
            [],
        ),
        "flush": (
            [
                _card("Hearts", "2"),
                _card("Hearts", "5"),
                _card("Hearts", "8"),
                _card("Hearts", "Jack"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "full_house": (
            [
                _card("Hearts", "King"),
                _card("Spades", "King"),
                _card("Clubs", "King"),
                _card("Diamonds", "5"),
                _card("Hearts", "5"),
            ],
            [],
        ),
        "four_of_a_kind": (
            [
                _card("Hearts", "7"),
                _card("Spades", "7"),
                _card("Clubs", "7"),
                _card("Diamonds", "7"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "straight_flush": (
            [
                _card("Hearts", "4"),
                _card("Hearts", "5"),
                _card("Hearts", "6"),
                _card("Hearts", "7"),
                _card("Hearts", "8"),
            ],
            [],
        ),
        "five_of_a_kind": (
            [
                _card("Hearts", "Ace"),
                _card("Spades", "Ace"),
                _card("Clubs", "Ace"),
                _card("Diamonds", "Ace"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "flush_five": (
            [
                _card("Hearts", "Ace"),
                _card("Hearts", "Ace"),
                _card("Hearts", "Ace"),
                _card("Hearts", "Ace"),
                _card("Hearts", "Ace"),
            ],
            [],
        ),
        "flush_house": (
            [
                _card("Hearts", "King"),
                _card("Hearts", "King"),
                _card("Hearts", "King"),
                _card("Hearts", "5"),
                _card("Hearts", "5"),
            ],
            [],
        ),
        "four_fingers_flush": (
            [
                _card("Clubs", "3"),
                _card("Clubs", "7"),
                _card("Clubs", "10"),
                _card("Clubs", "King"),
                _card("Hearts", "Ace"),
            ],
            ["j_four_fingers"],
        ),
        "shortcut_straight": (
            [
                _card("Hearts", "3"),
                _card("Spades", "5"),
                _card("Clubs", "6"),
                _card("Diamonds", "7"),
                _card("Hearts", "8"),
            ],
            ["j_shortcut"],
        ),
        "no_wrap": (
            [
                _card("Hearts", "Queen"),
                _card("Spades", "King"),
                _card("Clubs", "Ace"),
                _card("Diamonds", "2"),
                _card("Hearts", "3"),
            ],
            [],
        ),
        "wild_in_flush": (
            [
                _card("Spades", "2"),
                _card("Spades", "5"),
                _card("Spades", "8"),
                _card("Spades", "Jack"),
                _card("Hearts", "Ace", enhancement="m_wild"),
            ],
            [],
        ),
    }


def _find_lua() -> str | None:
    for name in ["luajit", "lua"]:
        p = shutil.which(name)
        if p:
            return p
    for candidate in [
        Path.home() / "AppData/Local/Programs/LuaJIT/bin/luajit.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


# ============================================================================
# Fixture-based tests (always run)
# ============================================================================


class TestFixtureOracle:
    """Validate Python evaluator against pre-generated Lua output."""

    @pytest.fixture(scope="class")
    def fixture(self) -> dict:
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
        with open(FIXTURE_PATH) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def hands(self) -> dict:
        return _build_test_hands()

    def test_all_detected_hands_match(self, fixture, hands):
        """For every test case, Python detects the same hand type as Lua."""
        for lua_test in fixture["tests"]:
            name = lua_test["name"]
            if name not in hands:
                continue
            cards, joker_keys = hands[name]
            jokers = [create_joker(k) for k in joker_keys]
            result = evaluate_hand(cards, jokers=jokers or None)
            assert result.detected_hand == lua_test["detected"], (
                f"{name}: Python={result.detected_hand!r}, Lua={lua_test['detected']!r}"
            )

    def test_all_populated_hands_match(self, fixture, hands):
        """For every test case, same set of hands are populated."""
        for lua_test in fixture["tests"]:
            name = lua_test["name"]
            if name not in hands:
                continue
            cards, joker_keys = hands[name]
            jokers = [create_joker(k) for k in joker_keys]
            result = evaluate_hand(cards, jokers=jokers or None)
            py_populated = sorted(h for h in result.poker_hands if result.poker_hands[h])
            lua_populated = sorted(lua_test["populated"])
            assert py_populated == lua_populated, (
                f"{name}: Python={py_populated}, Lua={lua_populated}"
            )

    @pytest.mark.parametrize(
        "test_name",
        [
            "high_card",
            "pair",
            "two_pair",
            "three_of_a_kind",
            "straight",
            "ace_low_straight",
            "flush",
            "full_house",
            "four_of_a_kind",
            "straight_flush",
            "five_of_a_kind",
            "flush_five",
            "flush_house",
            "four_fingers_flush",
            "shortcut_straight",
            "no_wrap",
            "wild_in_flush",
        ],
    )
    def test_individual_hand(self, fixture, hands, test_name):
        """Parametrized: each test case individually."""
        lua_test = next(t for t in fixture["tests"] if t["name"] == test_name)
        cards, joker_keys = hands[test_name]
        jokers = [create_joker(k) for k in joker_keys]
        result = evaluate_hand(cards, jokers=jokers or None)
        assert result.detected_hand == lua_test["detected"]


# ============================================================================
# Live Lua oracle (optional)
# ============================================================================


class TestLiveOracle:
    @pytest.fixture(scope="class")
    def lua_path(self):
        path = _find_lua()
        if not path:
            pytest.skip("No Lua interpreter found")
        return path

    @pytest.fixture(scope="class")
    def live_data(self, lua_path) -> dict:
        result = subprocess.run(
            [lua_path, str(ORACLE_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            pytest.fail(f"Oracle failed: {result.stderr}")
        return json.loads(result.stdout)

    def test_live_matches_fixture(self, live_data):
        """Live Lua output should match saved fixture."""
        if not FIXTURE_PATH.exists():
            pytest.skip("No fixture to compare")
        with open(FIXTURE_PATH) as f:
            fixture = json.load(f)

        for lua_t, fix_t in zip(live_data["tests"], fixture["tests"]):
            assert lua_t["detected"] == fix_t["detected"], (
                f"{lua_t['name']}: live={lua_t['detected']}, fixture={fix_t['detected']}"
            )

    def test_all_hands_match_python(self, live_data):
        hands = _build_test_hands()
        for lua_test in live_data["tests"]:
            name = lua_test["name"]
            if name not in hands:
                continue
            cards, joker_keys = hands[name]
            jokers = [create_joker(k) for k in joker_keys]
            result = evaluate_hand(cards, jokers=jokers or None)
            assert result.detected_hand == lua_test["detected"]
