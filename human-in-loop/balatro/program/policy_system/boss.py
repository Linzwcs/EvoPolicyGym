"""Pure constraints derived from the visible Boss Blind rule text."""

from __future__ import annotations

from typing import Any


def _rule(blind: dict[str, Any]) -> str:
    return str(blind.get("rule", "")).lower()


def requires_five_cards(blind: dict[str, Any]) -> bool:
    rule = _rule(blind)
    return "must play 5 cards" in rule or (
        "five cards" in rule and "must be played" in rule
    )


def only_one_hand(blind: dict[str, Any]) -> bool:
    rule = _rule(blind)
    return (
        "only one hand is available" in rule
        or "only 1 hand" in rule
    )


def restricts_hand_type(blind: dict[str, Any]) -> str:
    """Return ``unique`` or ``single`` for visible hand-type Boss rules."""

    rule = _rule(blind)
    if (
        "no repeat hand types" in rule
        or "no poker hand type may be played more than once" in rule
    ):
        return "unique"
    if (
        "only one hand type" in rule
        or "after the first hand, only that poker hand type" in rule
    ):
        return "single"
    return ""


def first_hand_hidden(blind: dict[str, Any], hand: list[dict[str, Any]]) -> bool:
    rule = _rule(blind)
    return (
        "first hand" in rule
        and "face-down" in rule
        and any(card.get("rank") is None for card in hand)
    )


def filter_candidates(
    ranked: list[tuple[float, str, list[int]]],
    *,
    blind: dict[str, Any],
    played_hand_types: set[str],
) -> list[tuple[float, str, list[int]]]:
    result = ranked
    if requires_five_cards(blind):
        result = [candidate for candidate in result if len(candidate[2]) == 5]

    restriction = restricts_hand_type(blind)
    if played_hand_types and restriction in {"unique", "single"}:
        if restriction == "unique":
            result = [
                candidate
                for candidate in result
                if candidate[1] not in played_hand_types
            ]
        else:
            result = [
                candidate
                for candidate in result
                if candidate[1] in played_hand_types
            ]

    # A public rule must not make the policy emit an empty/illegal intent.
    return result or ranked
