"""Game enums matching the Balatro Lua source.

String values are chosen to match the Lua source's actual key strings
so they can be used directly for lookups against P_CENTERS, card fields,
and game state comparisons.

Source references:
  - GameState: globals.lua G.STATES
  - Suit/Rank: P_CARDS keys and Card:set_base
  - Enhancement: P_CENTERS Enhanced set keys (m_*)
  - Edition: card.edition field keys (foil, holo, polychrome, negative)
  - Seal: P_SEALS keys (Gold, Red, Blue, Purple)
  - Rarity: P_CENTERS rarity field values
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

# ---------------------------------------------------------------------------
# Game state machine
# ---------------------------------------------------------------------------


class GameState(IntEnum):
    """Game states from globals.lua G.STATES.

    Values match the Lua source exactly (used in state machine dispatch).
    """

    SELECTING_HAND = 1
    HAND_PLAYED = 2
    DRAW_TO_HAND = 3
    GAME_OVER = 4
    SHOP = 5
    PLAY_TAROT = 6
    BLIND_SELECT = 7
    ROUND_EVAL = 8
    TAROT_PACK = 9
    PLANET_PACK = 10
    # 11 = MENU, 12 = TUTORIAL, 13 = SPLASH, 14 = SANDBOX (not in simulator)
    SPECTRAL_PACK = 15
    # 16 = DEMO_CTA (not in simulator)
    STANDARD_PACK = 17
    BUFFOON_PACK = 18
    NEW_ROUND = 19


class GameStage(IntEnum):
    """High-level game mode from globals.lua G.STAGES."""

    MAIN_MENU = 1
    RUN = 2
    SANDBOX = 3


# ---------------------------------------------------------------------------
# Card properties
# ---------------------------------------------------------------------------


class Suit(StrEnum):
    """Playing card suits. Values match P_CARDS suit field strings."""

    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"


class Rank(StrEnum):
    """Playing card ranks. Values match P_CARDS value field strings."""

    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "Jack"
    QUEEN = "Queen"
    KING = "King"
    ACE = "Ace"


class Enhancement(StrEnum):
    """Card enhancements. Values match P_CENTERS Enhanced set keys (m_*)."""

    NONE = "none"
    BONUS = "m_bonus"
    MULT = "m_mult"
    WILD = "m_wild"
    GLASS = "m_glass"
    STEEL = "m_steel"
    STONE = "m_stone"
    GOLD = "m_gold"
    LUCKY = "m_lucky"


class Edition(StrEnum):
    """Card editions. Values match card.edition table keys in Lua.

    Note: P_CENTERS uses ``e_foil``, ``e_holo``, etc., but the runtime
    card.edition field uses the short names: ``{foil = true}``.
    """

    NONE = "none"
    FOIL = "foil"
    HOLOGRAPHIC = "holo"
    POLYCHROME = "polychrome"
    NEGATIVE = "negative"


class Seal(StrEnum):
    """Card seals. Values match P_SEALS keys."""

    NONE = "none"
    GOLD = "Gold"
    RED = "Red"
    BLUE = "Blue"
    PURPLE = "Purple"


class Rarity(IntEnum):
    """Joker rarity tiers. Values match P_CENTERS rarity field."""

    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    LEGENDARY = 4


# ---------------------------------------------------------------------------
# Lookup tables for card numeric values
# ---------------------------------------------------------------------------

RANK_CHIPS: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 10,
    Rank.QUEEN: 10,
    Rank.KING: 10,
    Rank.ACE: 11,
}
"""Chip value each rank contributes when scored (matches Card:get_chip_bonus)."""

RANK_ID: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}
"""Numeric ordering ID per rank (matches Card:get_id, used for hand detection)."""

SUIT_NOMINAL: dict[Suit, float] = {
    Suit.SPADES: 0.04,
    Suit.HEARTS: 0.03,
    Suit.CLUBS: 0.02,
    Suit.DIAMONDS: 0.01,
}
"""Suit tiebreaker values for card sorting (matches Card:get_nominal)."""

# Edition cost surcharges (added to card.cost in Card:set_cost)
EDITION_COST: dict[Edition, int] = {
    Edition.NONE: 0,
    Edition.FOIL: 2,
    Edition.HOLOGRAPHIC: 3,
    Edition.POLYCHROME: 5,
    Edition.NEGATIVE: 5,
}
"""Extra cost added to a card's price per edition type."""

# Edition scoring bonuses (returned by Card:get_edition)
EDITION_CHIPS: dict[Edition, int] = {
    Edition.NONE: 0,
    Edition.FOIL: 50,
    Edition.HOLOGRAPHIC: 0,
    Edition.POLYCHROME: 0,
    Edition.NEGATIVE: 0,
}

EDITION_MULT: dict[Edition, int] = {
    Edition.NONE: 0,
    Edition.FOIL: 0,
    Edition.HOLOGRAPHIC: 10,
    Edition.POLYCHROME: 0,
    Edition.NEGATIVE: 0,
}

EDITION_X_MULT: dict[Edition, float] = {
    Edition.NONE: 1.0,
    Edition.FOIL: 1.0,
    Edition.HOLOGRAPHIC: 1.0,
    Edition.POLYCHROME: 1.5,
    Edition.NEGATIVE: 1.0,
}
