"""Tests for jackdaw.engine.tags — Tag.apply, TagResult, and tag generation.

Coverage
--------
* Tag construction, economy (capped), garbage, skip, orbital, boss, investment.
* Double Tag, Juggle Tag, D6 Tag, edition tags, coupon tag.
* Wrong-context returns None (parametrized).
* assign_ante_blinds: known seed, structure, game_state mutation.
* min_ante filtering at ante 1.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.data.hands import HandType
from jackdaw.engine.data.prototypes import TAGS
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.tags import Tag, assign_ante_blinds, generate_blind_tags

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag(key: str) -> Tag:
    return Tag(key)


def _gs(**kwargs) -> dict:
    """Minimal game_state dict for tag tests."""
    defaults = {
        "dollars": 10,
        "hands_played": 5,
        "unused_discards": 3,
        "skips": 2,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Tag construction
# ---------------------------------------------------------------------------


class TestTagConstruction:
    def test_key_and_name(self):
        t = Tag("tag_economy")
        assert t.key == "tag_economy"
        assert t.name == "Economy Tag"


# ---------------------------------------------------------------------------
# Immediate tags
# ---------------------------------------------------------------------------


class TestTagEconomy:
    def test_dollars_equals_current_balance(self):
        gs = _gs(dollars=10)
        result = _tag("tag_economy").apply("immediate", gs)
        assert result is not None
        assert result.dollars == 10

    def test_capped_at_max(self):
        gs = _gs(dollars=100)
        result = _tag("tag_economy").apply("immediate", gs)
        assert result.dollars == 40  # config max


class TestTagOrbital:
    def test_returns_level_up(self):
        rng = PseudoRandom("TEST_ORBITAL")
        result = _tag("tag_orbital").apply("immediate", _gs(), rng=rng)
        assert result is not None
        assert result.level_up is not None
        hand_type, levels = result.level_up
        assert isinstance(hand_type, HandType)
        assert levels == 3  # config["levels"]


# ---------------------------------------------------------------------------
# new_blind_choice tags
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# tag_add — Double Tag
# ---------------------------------------------------------------------------


class TestTagDouble:
    def test_fires_when_different_tag_added(self):
        result = _tag("tag_double").apply("tag_add", _gs(), added_tag_key="tag_economy")
        assert result is not None
        assert result.double is True


# ===========================================================================
# Tag generation (merged from test_tag_generation.py)
# ===========================================================================


class TestAssignAnteBlindsKnownSeed:
    def test_tutorial_ante1_boss(self):
        rng = PseudoRandom("TUTORIAL")
        result = assign_ante_blinds(1, rng, {})
        assert result["blind_choices"]["Boss"] == "bl_hook"


_MIN_ANTE_2_TAGS: frozenset[str] = frozenset(
    k for k, v in TAGS.items() if v.min_ante is not None and v.min_ante > 1
)


class TestMinAnteFiltering:
    @pytest.mark.parametrize(
        "seed",
        ["AAA", "BBB", "CCC"],
    )
    def test_ante1_never_returns_min_ante2_tag(self, seed):
        result = generate_blind_tags(1, PseudoRandom(seed), {})
        assert result["Small"] not in _MIN_ANTE_2_TAGS, (
            f"seed={seed!r}: got {result['Small']!r} which requires min_ante>=2"
        )
        assert result["Big"] not in _MIN_ANTE_2_TAGS, (
            f"seed={seed!r}: got {result['Big']!r} which requires min_ante>=2"
        )
