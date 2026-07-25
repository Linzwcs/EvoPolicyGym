"""Serialize engine objects into balatrobot's JSON wire format.

The reverse direction of ``balatrobot_adapter.py`` — converts jackdaw
engine state INTO the JSON shapes that balatrobot expects.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.data.hands import HAND_BASE, HandType
from jackdaw.engine.data.prototypes import BLINDS
from jackdaw.engine.hand_levels import HandLevels

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_SUIT_TO_LETTER: dict[str, str] = {
    "Hearts": "H",
    "Diamonds": "D",
    "Clubs": "C",
    "Spades": "S",
}

_RANK_TO_LETTER: dict[str, str] = {
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

_SET_MAP: dict[str, str] = {
    "": "DEFAULT",
    "Default": "DEFAULT",
    "Enhanced": "ENHANCED",
    "Joker": "JOKER",
    "Tarot": "TAROT",
    "Planet": "PLANET",
    "Spectral": "SPECTRAL",
    "Voucher": "VOUCHER",
    "Booster": "BOOSTER",
}

_SEAL_MAP: dict[str | None, str | None] = {
    None: None,
    "Gold": "GOLD",
    "Red": "RED",
    "Blue": "BLUE",
    "Purple": "PURPLE",
}

_ENHANCEMENT_MAP: dict[str, str | None] = {
    "c_base": None,
    "m_bonus": "BONUS",
    "m_mult": "MULT",
    "m_wild": "WILD",
    "m_glass": "GLASS",
    "m_steel": "STEEL",
    "m_stone": "STONE",
    "m_gold": "GOLD",
    "m_lucky": "LUCKY",
}

_PHASE_TO_STATE: dict[GamePhase, str] = {
    GamePhase.BLIND_SELECT: "BLIND_SELECT",
    GamePhase.SELECTING_HAND: "SELECTING_HAND",
    GamePhase.ROUND_EVAL: "ROUND_EVAL",
    GamePhase.SHOP: "SHOP",
    GamePhase.PACK_OPENING: "SMODS_BOOSTER_OPENED",
    GamePhase.GAME_OVER: "GAME_OVER",
}

_DECK_MAP: dict[str, str] = {
    "b_red": "RED",
    "b_blue": "BLUE",
    "b_yellow": "YELLOW",
    "b_green": "GREEN",
    "b_black": "BLACK",
    "b_magic": "MAGIC",
    "b_nebula": "NEBULA",
    "b_ghost": "GHOST",
    "b_abandoned": "ABANDONED",
    "b_checkered": "CHECKERED",
    "b_zodiac": "ZODIAC",
    "b_painted": "PAINTED",
    "b_anaglyph": "ANAGLYPH",
    "b_plasma": "PLASMA",
    "b_erratic": "ERRATIC",
}

_STAKE_MAP: dict[int, str] = {
    1: "WHITE",
    2: "RED",
    3: "GREEN",
    4: "BLACK",
    5: "BLUE",
    6: "PURPLE",
    7: "ORANGE",
    8: "GOLD",
}

_BLIND_STATUS_MAP: dict[str, str] = {
    "Select": "SELECT",
    "Current": "CURRENT",
    "Skipped": "SKIPPED",
    "Defeated": "DEFEATED",
}


# ---------------------------------------------------------------------------
# Tag data (lazy-loaded)
# ---------------------------------------------------------------------------

_tags_cache: dict[str, Any] | None = None


def _load_tags() -> dict[str, Any]:
    global _tags_cache  # noqa: PLW0603
    if _tags_cache is None:
        from jackdaw.engine.data.prototypes import _load_json

        _tags_cache = _load_json("tags.json")
    return _tags_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edition_to_str(edition: dict[str, Any] | None) -> str | None:
    """Map an engine edition dict to balatrobot's edition string."""
    if not edition:
        return None
    if edition.get("foil"):
        return "FOIL"
    if edition.get("holo"):
        return "HOLO"
    if edition.get("polychrome"):
        return "POLYCHROME"
    if edition.get("negative"):
        return "NEGATIVE"
    return None


# ---------------------------------------------------------------------------
# Card serialization
# ---------------------------------------------------------------------------


def serialize_card(card: Card) -> dict[str, Any]:
    """Convert an engine Card to balatrobot's JSON card schema.

    Pure function — does not mutate the card.
    """
    is_playing_card = card.base is not None
    ability_set = card.ability.get("set", "")

    # key
    if is_playing_card:
        key = card.card_key or ""
    else:
        key = card.center_key

    # label
    if is_playing_card:
        label = f"{card.base.rank.value} of {card.base.suit.value}"
    else:
        label = card.ability.get("name", "")

    # value
    if is_playing_card:
        suit_letter = _SUIT_TO_LETTER.get(card.base.suit.value)
        rank_letter = _RANK_TO_LETTER.get(card.base.rank.value)
    else:
        suit_letter = None
        rank_letter = None

    # value.effect
    if ability_set == "Joker":
        effect = card.ability.get("name", "")
    elif is_playing_card:
        eff = card.ability.get("effect", "")
        effect = eff if eff else ""
    else:
        effect = card.ability.get("effect", "")

    # enhancement
    enhancement = _ENHANCEMENT_MAP.get(card.center_key)

    return {
        "id": card.sort_id,
        "key": key,
        "set": _SET_MAP.get(ability_set, ability_set),
        "label": label,
        "value": {
            "suit": suit_letter,
            "rank": rank_letter,
            "effect": effect,
        },
        "modifier": {
            "seal": _SEAL_MAP.get(card.seal),
            "edition": _edition_to_str(card.edition),
            "enhancement": enhancement,
            "eternal": card.eternal,
            "perishable": card.perish_tally if card.perishable else None,
            "rental": card.rental,
        },
        "state": {
            "debuff": card.debuff,
            "hidden": card.facing == "back",
            "highlight": False,
        },
        "cost": {
            "sell": card.sell_cost,
            "buy": card.cost,
        },
    }


# ---------------------------------------------------------------------------
# Area serialization
# ---------------------------------------------------------------------------


def serialize_area(
    cards: list[Card],
    limit: int,
    highlighted_limit: int = 0,
) -> dict[str, Any]:
    """Convert a card area (hand, jokers, etc.) to balatrobot's area schema."""
    return {
        "count": len(cards),
        "limit": limit,
        "highlighted_limit": highlighted_limit,
        "cards": [serialize_card(c) for c in cards],
    }


# ---------------------------------------------------------------------------
# Poker hand serialization
# ---------------------------------------------------------------------------


def serialize_hands(hand_levels: HandLevels) -> dict[str, dict[str, Any]]:
    """Convert HandLevels to balatrobot's hands schema.

    Only includes hands where ``visible`` is True (secret hands stay
    hidden until leveled).
    """
    result: dict[str, dict[str, Any]] = {}
    for ht in HandType:
        hs = hand_levels.get_state(ht)
        if not hs.visible:
            continue
        result[ht.value] = {
            "order": HAND_BASE[ht].order,
            "level": hs.level,
            "chips": hs.chips,
            "mult": hs.mult,
            "played": hs.played,
            "played_this_round": hs.played_this_round,
            "example": [],
        }
    return result


# ---------------------------------------------------------------------------
# Blind serialization
# ---------------------------------------------------------------------------


def _tag_info(tag_key: str | None) -> tuple[str, str]:
    """Return ``(tag_name, tag_effect)`` for a tag key, or empty strings."""
    if not tag_key:
        return "", ""
    tags = _load_tags()
    tag = tags.get(tag_key)
    if tag is None:
        return "", ""
    return tag.get("name", ""), ""


def serialize_blind(blind_type: str, gs: dict[str, Any]) -> dict[str, Any]:
    """Serialize one blind entry (small/big/boss) to balatrobot schema."""
    rr = gs.get("round_resets", {})
    blind_states = rr.get("blind_states", {})
    blind_choices = rr.get("blind_choices", {})
    blind_tags = rr.get("blind_tags", {})

    type_upper = blind_type.upper()
    status_raw = blind_states.get(blind_type, "")
    status = _BLIND_STATUS_MAP.get(status_raw, "UPCOMING")

    blind_key = blind_choices.get(blind_type, "")
    proto = BLINDS.get(blind_key)
    name = proto.name if proto else ""

    # Effect description
    if blind_type in ("Small", "Big"):
        effect = "No special effect"
    elif proto and proto.boss is not None:
        effect = name
    else:
        effect = ""

    # Score (chip target)
    active_blind: Blind | None = gs.get("blind")
    if active_blind and status_raw == "Current":
        score = active_blind.chips
    elif proto:
        ante = rr.get("ante", 1)
        scaling = gs.get("modifiers", {}).get("scaling", 1)
        ante_scaling = gs.get("starting_params", {}).get("ante_scaling", 1.0)
        computed = Blind.create(blind_key, ante, scaling, ante_scaling)
        score = computed.chips
    else:
        score = 0

    # Tag info (small/big only)
    tag_key = blind_tags.get(blind_type)
    tag_name, tag_effect = _tag_info(tag_key)

    return {
        "type": type_upper,
        "status": status,
        "name": name,
        "effect": effect,
        "score": score,
        "tag_name": tag_name,
        "tag_effect": tag_effect,
    }


def serialize_blinds(gs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Serialize all three blind entries."""
    return {
        "small": serialize_blind("Small", gs),
        "big": serialize_blind("Big", gs),
        "boss": serialize_blind("Boss", gs),
    }


# ---------------------------------------------------------------------------
# Top-level game state serialization
# ---------------------------------------------------------------------------


def game_state_to_bot_response(gs: dict[str, Any]) -> dict[str, Any]:
    """Convert engine game_state to balatrobot's full JSON gamestate response."""
    phase = gs.get("phase", "")
    rr = gs.get("round_resets", {})
    cr = gs.get("current_round", {})

    hand = gs.get("hand", [])
    deck = gs.get("deck", [])
    jokers = gs.get("jokers", [])
    consumables = gs.get("consumables", [])
    shop_cards = gs.get("shop_cards", [])
    shop_vouchers = gs.get("shop_vouchers", [])
    shop_boosters = gs.get("shop_boosters", [])
    pack_cards = gs.get("pack_cards", [])

    return {
        "state": _PHASE_TO_STATE.get(phase, str(phase)),
        "round_num": gs.get("round", 0),
        "ante_num": rr.get("ante", 1),
        "money": gs.get("dollars", 0),
        "deck": _DECK_MAP.get(gs.get("selected_back_key", ""), ""),
        "stake": _STAKE_MAP.get(gs.get("stake", 1), "WHITE"),
        "seed": gs["rng"].seed_str if "rng" in gs else gs.get("seed", ""),
        "won": gs.get("won", False),
        "used_vouchers": gs.get("used_vouchers", {}),
        "hands": serialize_hands(gs["hand_levels"]) if "hand_levels" in gs else {},
        "round": {
            "hands_left": cr.get("hands_left", 0),
            "hands_played": cr.get("hands_played", 0),
            "discards_left": cr.get("discards_left", 0),
            "discards_used": cr.get("discards_used", 0),
            "reroll_cost": cr.get("reroll_cost", 5),
            "chips": gs.get("chips", 0),
        },
        "blinds": serialize_blinds(gs),
        "jokers": serialize_area(jokers, gs.get("joker_slots", 5)),
        "consumables": serialize_area(consumables, gs.get("consumable_slots", 2)),
        "cards": serialize_area(deck, len(deck)),
        "hand": serialize_area(hand, gs.get("hand_size", 8), highlighted_limit=5),
        "shop": serialize_area(shop_cards, len(shop_cards)),
        "vouchers": serialize_area(shop_vouchers, len(shop_vouchers)),
        "packs": serialize_area(shop_boosters, len(shop_boosters)),
        "pack": serialize_area(pack_cards, gs.get("pack_choices_remaining", 0)),
    }
