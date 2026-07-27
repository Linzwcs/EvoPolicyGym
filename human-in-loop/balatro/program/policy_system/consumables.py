"""Conservative selection of visible, explicitly-targeted consumables."""

from __future__ import annotations

from itertools import combinations
from typing import Any


def targeted_consumable(
    consumables: list[dict[str, Any]],
    descriptor: dict[str, Any],
    hand: list[dict[str, Any]],
    primary_hand: str,
) -> tuple[int, list[int]] | None:
    """Return a safe target only when the public rule describes its intent.

    This deliberately avoids guessing Tarot/Spectral effects: a card must say
    ``selected``/``enhance`` (or be a matching Planet) and the legal descriptor
    must provide the complete target bounds.
    """
    by_index = {int(card.get("index", -1)): card for card in consumables}
    hand_indices = [int(card.get("index", -1)) for card in hand]
    for raw in descriptor.get("targets") or []:
        if not isinstance(raw, dict):
            continue
        target = int(raw.get("target_index", -1))
        card = by_index.get(target)
        if card is None:
            continue
        summary = str((card.get("rule") or {}).get("summary", "")).lower()
        card_set = str(card.get("set", "")).lower()
        planet = "planet" in card_set and primary_hand.lower() in summary
        targeted = any(word in summary for word in ("selected", "enhance"))
        if not (planet or targeted):
            continue
        minimum = int(raw.get("min_cards", 0))
        maximum = int(raw.get("max_cards", 0))
        allowed = sorted({int(x) for x in raw.get("card_indices", [])})
        allowed = [index for index in allowed if index in hand_indices]
        if minimum == maximum == 0:
            return target, []
        if minimum < 1 or maximum < minimum or len(allowed) < minimum:
            continue
        # Prefer the largest legal target set; the ordering is deterministic
        # and does not depend on hidden card state.
        count = min(maximum, len(allowed))
        if count < minimum:
            continue
        return target, list(next(combinations(allowed, count)))
    return None
