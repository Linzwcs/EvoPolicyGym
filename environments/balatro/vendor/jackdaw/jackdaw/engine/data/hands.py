"""Poker hand types, base values, and level-up increments.

Data from ``Game:init_game_object()`` (game.lua:2001-2014), documented in
``docs/balatro-internals/scoring-pipeline.md`` and ``docs/balatro-internals/g-table.md``.

The hand name strings are the exact keys used in ``G.GAME.hands`` and by
``evaluate_poker_hand`` for detection and lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HandType(StrEnum):
    """Poker hand types in detection priority order (highest first).

    The string values match the Lua source's hand name keys exactly —
    they are used as dictionary keys in ``G.GAME.hands[handname]``.
    """

    FLUSH_FIVE = "Flush Five"
    FLUSH_HOUSE = "Flush House"
    FIVE_OF_A_KIND = "Five of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    FOUR_OF_A_KIND = "Four of a Kind"
    FULL_HOUSE = "Full House"
    FLUSH = "Flush"
    STRAIGHT = "Straight"
    THREE_OF_A_KIND = "Three of a Kind"
    TWO_PAIR = "Two Pair"
    PAIR = "Pair"
    HIGH_CARD = "High Card"


@dataclass(frozen=True)
class HandBaseData:
    """Base scoring values and level-up increments for a poker hand type.

    At level *L*, the hand provides:

    - ``chips = s_chips + l_chips * (L - 1)``
    - ``mult  = s_mult  + l_mult  * (L - 1)``

    At level 1 (the starting level), chips = s_chips and mult = s_mult.
    """

    s_chips: int  # starting chips (level 1)
    s_mult: int  # starting mult (level 1)
    l_chips: int  # chips gained per level-up
    l_mult: int  # mult gained per level-up
    order: int  # display sort order (1 = Flush Five, 12 = High Card)
    visible: bool  # shown in hand info by default (secret hands start False)

    def chips_at(self, level: int) -> int:
        """Chip value at the given level."""
        return max(0, self.s_chips + self.l_chips * (level - 1))

    def mult_at(self, level: int) -> int:
        """Mult value at the given level."""
        return max(1, self.s_mult + self.l_mult * (level - 1))


# ---------------------------------------------------------------------------
# Hand base data table
# ---------------------------------------------------------------------------
# Values from game.lua init_game_object() lines 2001-2014.

HAND_BASE: dict[HandType, HandBaseData] = {
    HandType.FLUSH_FIVE: HandBaseData(
        s_chips=160,
        s_mult=16,
        l_chips=50,
        l_mult=3,
        order=1,
        visible=False,
    ),
    HandType.FLUSH_HOUSE: HandBaseData(
        s_chips=140,
        s_mult=14,
        l_chips=40,
        l_mult=4,
        order=2,
        visible=False,
    ),
    HandType.FIVE_OF_A_KIND: HandBaseData(
        s_chips=120,
        s_mult=12,
        l_chips=35,
        l_mult=3,
        order=3,
        visible=False,
    ),
    HandType.STRAIGHT_FLUSH: HandBaseData(
        s_chips=100,
        s_mult=8,
        l_chips=40,
        l_mult=4,
        order=4,
        visible=True,
    ),
    HandType.FOUR_OF_A_KIND: HandBaseData(
        s_chips=60,
        s_mult=7,
        l_chips=30,
        l_mult=3,
        order=5,
        visible=True,
    ),
    HandType.FULL_HOUSE: HandBaseData(
        s_chips=40,
        s_mult=4,
        l_chips=25,
        l_mult=2,
        order=6,
        visible=True,
    ),
    HandType.FLUSH: HandBaseData(
        s_chips=35,
        s_mult=4,
        l_chips=15,
        l_mult=2,
        order=7,
        visible=True,
    ),
    HandType.STRAIGHT: HandBaseData(
        s_chips=30,
        s_mult=4,
        l_chips=30,
        l_mult=3,
        order=8,
        visible=True,
    ),
    HandType.THREE_OF_A_KIND: HandBaseData(
        s_chips=30,
        s_mult=3,
        l_chips=20,
        l_mult=2,
        order=9,
        visible=True,
    ),
    HandType.TWO_PAIR: HandBaseData(
        s_chips=20,
        s_mult=2,
        l_chips=20,
        l_mult=1,
        order=10,
        visible=True,
    ),
    HandType.PAIR: HandBaseData(
        s_chips=10,
        s_mult=2,
        l_chips=15,
        l_mult=1,
        order=11,
        visible=True,
    ),
    HandType.HIGH_CARD: HandBaseData(
        s_chips=5,
        s_mult=1,
        l_chips=10,
        l_mult=1,
        order=12,
        visible=True,
    ),
}

# ---------------------------------------------------------------------------
# Detection priority order (highest to lowest)
# ---------------------------------------------------------------------------
# This is the order used by evaluate_poker_hand (misc_functions.lua:376)
# and get_poker_hand_info (state_events.lua:540) — first match wins.

HAND_ORDER: list[HandType] = [
    HandType.FLUSH_FIVE,
    HandType.FLUSH_HOUSE,
    HandType.FIVE_OF_A_KIND,
    HandType.STRAIGHT_FLUSH,
    HandType.FOUR_OF_A_KIND,
    HandType.FULL_HOUSE,
    HandType.FLUSH,
    HandType.STRAIGHT,
    HandType.THREE_OF_A_KIND,
    HandType.TWO_PAIR,
    HandType.PAIR,
    HandType.HIGH_CARD,
]
