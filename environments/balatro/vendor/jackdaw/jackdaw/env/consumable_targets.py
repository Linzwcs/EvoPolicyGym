"""Consumable targeting system for the RL environment.

Given a specific consumable card, determines:
- What card selection constraints apply (min/max targets, filters)
- Which hand cards are valid targets
- Whether a specific target selection is legal

Uses ``_resolve_consumable_config`` from the engine as the canonical source
for ``max_highlighted``/``min_highlighted`` fields, with hardcoded overrides
for special-case consumables (Aura, Sigil, Ouija, etc.).

Source of truth: jackdaw/engine/consumables.py (can_use_consumable, handler
registrations) and jackdaw/engine/data/centers.json (config fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jackdaw.engine.consumables import (
    _ALWAYS_USABLE,
    _NEED_CONSUMABLE_SLOT,
    _NEED_ELIGIBLE_JOKER,
    _NEED_HAND_CARDS,
    _NEED_JOKER_SLOT,
    _PLANET_KEYS,
    _resolve_consumable_config,
)

if TYPE_CHECKING:
    from jackdaw.engine.card import Card

# ---------------------------------------------------------------------------
# ConsumableTargetSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumableTargetSpec:
    """Describes targeting requirements for a specific consumable.

    The RL agent uses this to know how many (and which) hand cards to
    select when using a consumable.
    """

    needs_card_targets: bool
    """True if the consumable requires hand card selection."""

    min_targets: int
    """Minimum cards to select (0 if no targets needed)."""

    max_targets: int
    """Maximum cards to select."""

    exact_targets: int | None
    """If not None, must select exactly this many cards."""

    target_filter: str
    """Filter for valid targets: "any", "no_edition", "all_hand", etc."""

    description: str
    """Human-readable description (for debugging)."""


# ---------------------------------------------------------------------------
# No-target sentinel
# ---------------------------------------------------------------------------

_NO_TARGETS = ConsumableTargetSpec(
    needs_card_targets=False,
    min_targets=0,
    max_targets=0,
    exact_targets=None,
    target_filter="any",
    description="no card targets",
)

# ---------------------------------------------------------------------------
# Special-case overrides (not derivable from config alone)
# ---------------------------------------------------------------------------

# Aura: needs exactly 1 highlighted card with no edition.
# Config is empty [] but can_use_consumable has a hardcoded check.
_AURA_SPEC = ConsumableTargetSpec(
    needs_card_targets=True,
    min_targets=1,
    max_targets=1,
    exact_targets=1,
    target_filter="no_edition",
    description="Aura: 1 card without edition",
)

# Sigil/Ouija: affect ALL hand cards — no player selection.
_ALL_HAND_SPEC = ConsumableTargetSpec(
    needs_card_targets=False,
    min_targets=0,
    max_targets=0,
    exact_targets=None,
    target_filter="all_hand",
    description="targets all hand cards (no selection)",
)

_SPECIAL_SPECS: dict[str, ConsumableTargetSpec] = {
    "c_aura": _AURA_SPEC,
    "c_sigil": _ALL_HAND_SPEC,
    "c_ouija": _ALL_HAND_SPEC,
}

# Keys that never need card targets (various reasons)
_NO_TARGET_KEYS = (
    _PLANET_KEYS
    | _ALWAYS_USABLE
    | _NEED_CONSUMABLE_SLOT
    | _NEED_JOKER_SLOT
    | _NEED_ELIGIBLE_JOKER
    | _NEED_HAND_CARDS
    | frozenset({"c_ankh", "c_black_hole"})
)


# ---------------------------------------------------------------------------
# get_consumable_target_spec
# ---------------------------------------------------------------------------


def get_consumable_target_spec(
    card: Card,
    game_state: dict[str, Any] | None = None,
) -> ConsumableTargetSpec:
    """Determine targeting requirements for a consumable.

    Parameters
    ----------
    card:
        The consumable card (must have ``center_key`` and ``ability``).
    game_state:
        Current game state (unused for most consumables, reserved for
        future context-dependent targeting).

    Returns
    -------
    ConsumableTargetSpec
        The targeting constraints for this consumable.
    """
    key = card.center_key

    # 1. Check special-case overrides
    if key in _SPECIAL_SPECS:
        return _SPECIAL_SPECS[key]

    # 2. Check known no-target keys
    if key in _NO_TARGET_KEYS:
        return _NO_TARGETS

    # 3. Check config for max_highlighted (the general case)
    cfg = _resolve_consumable_config(card)

    # Planet-like: cards with hand_type in config → no targets
    if cfg.get("hand_type"):
        return _NO_TARGETS

    max_h = cfg.get("max_highlighted")
    if max_h:
        min_h = cfg.get("min_highlighted", 1)
        # mod_num overrides the effective max (used by can_use_consumable)
        mod_num = cfg.get("mod_num", max_h)
        exact = mod_num if min_h == mod_num else None
        return ConsumableTargetSpec(
            needs_card_targets=True,
            min_targets=min_h,
            max_targets=mod_num,
            exact_targets=exact,
            target_filter="any",
            description=f"{key}: {min_h}-{mod_num} cards",
        )

    # 4. Fallback: no targets
    return _NO_TARGETS


# ---------------------------------------------------------------------------
# get_valid_target_cards
# ---------------------------------------------------------------------------


def get_valid_target_cards(
    card: Card,
    hand: list[Card],
    game_state: dict[str, Any] | None = None,
) -> list[int]:
    """Return indices of hand cards that are valid targets for this consumable.

    Parameters
    ----------
    card:
        The consumable card.
    hand:
        Current hand cards.
    game_state:
        Current game state dict.

    Returns
    -------
    list[int]
        Indices into ``hand`` of valid target cards.
    """
    spec = get_consumable_target_spec(card, game_state)

    if not spec.needs_card_targets:
        return []

    if spec.target_filter == "no_edition":
        return [i for i, c in enumerate(hand) if not c.edition]

    # Default: all hand cards are valid
    return list(range(len(hand)))


# ---------------------------------------------------------------------------
# validate_card_targets
# ---------------------------------------------------------------------------


def validate_card_targets(
    card: Card,
    target_indices: tuple[int, ...],
    hand: list[Card],
    game_state: dict[str, Any] | None = None,
) -> bool:
    """Validate that a specific target selection is legal for this consumable.

    Parameters
    ----------
    card:
        The consumable card.
    target_indices:
        Tuple of hand card indices selected as targets.
    hand:
        Current hand cards.
    game_state:
        Current game state dict.

    Returns
    -------
    bool
        True if the selection is valid.
    """
    spec = get_consumable_target_spec(card, game_state)

    # No-target consumables: must have empty selection
    if not spec.needs_card_targets:
        return len(target_indices) == 0

    n = len(target_indices)

    # Count check
    if n < spec.min_targets or n > spec.max_targets:
        return False

    if spec.exact_targets is not None and n != spec.exact_targets:
        return False

    # Bounds check: all indices must be valid
    if any(i < 0 or i >= len(hand) for i in target_indices):
        return False

    # Duplicates check
    if len(set(target_indices)) != n:
        return False

    # Filter check
    valid = set(get_valid_target_cards(card, hand, game_state))
    if not all(i in valid for i in target_indices):
        return False

    return True
