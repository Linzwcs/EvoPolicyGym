"""Player profile state — discovery, unlock, and career tracking.

Tracks which items are discovered (seen), unlocked (available for pool
generation), and career statistics.  Used by pool filtering to determine
which cards can appear in shops and packs.

Source: ``game.lua`` P_CENTERS ``unlocked`` field, ``common_events.lua``
``check_for_unlock``, save file ``profile.jkr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jackdaw.engine.data.prototypes import BACKS, TAGS

# ---------------------------------------------------------------------------
# Locked items — items with unlocked=false in centers.json
# ---------------------------------------------------------------------------

_LOCKED_ITEMS: set[str] | None = None


def _get_locked_items() -> set[str]:
    """Return the set of center keys that start locked (unlocked=false)."""
    global _LOCKED_ITEMS  # noqa: PLW0603
    if _LOCKED_ITEMS is None:
        from jackdaw.engine.data.prototypes import _load_json

        centers = _load_json("centers.json")
        _LOCKED_ITEMS = {k for k, v in centers.items() if v.get("unlocked") is False}
    return _LOCKED_ITEMS


def _get_all_center_keys() -> set[str]:
    """Return the set of all center keys."""
    from jackdaw.engine.data.prototypes import _load_json

    centers = _load_json("centers.json")
    return set(centers.keys())


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    """Player profile state for pool filtering and unlock tracking.

    Attributes
    ----------
    discovered:
        Center keys that have been seen in any run.  Items not in this
        set may be excluded from softlock-filtered pools (e.g. Tags with
        ``requires`` pointing to an undiscovered item).
    unlocked:
        Center keys available for pool generation.  Items with
        ``unlocked=false`` in the prototype are excluded from pools
        unless they're in this set.  Legendary jokers (rarity 4) bypass
        this check.
    career_stats:
        Persistent career statistics (hands played, cards purchased, etc.).
    """

    discovered: set[str] = field(default_factory=set)
    unlocked: set[str] = field(default_factory=set)
    career_stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profile constructors
# ---------------------------------------------------------------------------


def default_profile() -> Profile:
    """Fully-discovered, fully-unlocked profile.

    All items are discoverable and available for pool generation.
    This matches a player who has completed all unlocks — suitable
    for RL training where the agent should see the full game.
    """
    all_keys = _get_all_center_keys()
    tag_keys = set(TAGS.keys())
    back_keys = set(BACKS.keys())
    all_discovered = all_keys | tag_keys | back_keys

    return Profile(
        discovered=all_discovered,
        unlocked=all_discovered,
    )


def fresh_profile() -> Profile:
    """Starting profile — matches a brand-new Balatro save.

    105 of 150 jokers are unlocked.  All Tarots, Planets, Spectrals,
    Boosters, and Enhancements are unlocked.  45 Jokers, 16 Vouchers,
    and 14 Backs are locked.  No items are discovered.
    """
    locked = _get_locked_items()
    all_keys = _get_all_center_keys()
    unlocked = all_keys - locked

    return Profile(
        discovered=set(),  # nothing seen yet
        unlocked=unlocked,
    )


# ---------------------------------------------------------------------------
# Wire profile into game_state
# ---------------------------------------------------------------------------


def apply_profile_to_game_state(
    profile: Profile,
    game_state: dict[str, Any],
) -> None:
    """Store profile data in game_state for pool filtering.

    Sets ``game_state["discovered"]`` and updates pool filtering
    parameters to respect unlock state.
    """
    game_state["discovered"] = profile.discovered
    game_state["profile_unlocked"] = profile.unlocked
