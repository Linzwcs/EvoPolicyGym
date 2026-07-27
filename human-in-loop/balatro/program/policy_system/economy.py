"""Budgeted Voucher, Celestial pack, and reroll decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EconomyContext:
    money: int
    ante: int
    reroll_cost: int
    free_rerolls: int
    rerolls_this_shop: int
    celestials_this_shop: int
    joker_count: int
    joker_slots: int

    @property
    def reserve(self) -> int:
        return min(25, max(5, self.ante * 5))


@dataclass(frozen=True)
class EconomyIntent:
    kind: str
    target_index: int | None = None


def _parameters(card: dict[str, Any]) -> dict[str, Any]:
    rule = card.get("rule")
    if not isinstance(rule, dict):
        return {}
    parameters = rule.get("parameters")
    return parameters if isinstance(parameters, dict) else {}


def voucher_value(card: dict[str, Any], context: EconomyContext) -> float:
    """Value visible permanent effects without using Voucher identity."""

    summary = str((card.get("rule") or {}).get("summary", "")).lower()
    value = 0.0
    if "add 1 hand" in summary:
        value += 18
    if "hand size" in summary:
        value += 14
    if "cheaper" in summary or "% cheaper" in summary:
        value += 12
    if "card slot" in summary and "shop" in summary:
        value += 10
    if "interest cap" in summary and context.money >= 25:
        value += 8
    if "planet cards appear" in summary:
        value += 8
    if "consumable slot" in summary:
        value += 7
    if "edition" in summary:
        value += 6
    if "add 1 discard" in summary:
        value += 5
    if "prerequisite" in summary or "no immediate gameplay effect" in summary:
        value = 0
    if "lose 1 hand" in summary:
        value -= 18
    return value


def _affordable(
    card: dict[str, Any],
    context: EconomyContext,
) -> bool:
    cost = int(card.get("cost", 99))
    return cost <= context.money and context.money - cost >= context.reserve


def choose_voucher(
    vouchers: list[dict[str, Any]],
    *,
    legal_indices: set[int],
    context: EconomyContext,
) -> EconomyIntent | None:
    candidates = [
        card
        for card in vouchers
        if int(card.get("index", -1)) in legal_indices
        and _affordable(card, context)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda card: (
            voucher_value(card, context),
            -int(card.get("cost", 99)),
        ),
        reverse=True,
    )
    best = candidates[0]
    if voucher_value(best, context) < 9:
        return None
    return EconomyIntent(
        kind="redeem_voucher",
        target_index=int(best["index"]),
    )


def _celestial_value(card: dict[str, Any]) -> float:
    name = str(card.get("name", "")).lower()
    parameters = _parameters(card)
    choices = int(parameters.get("choose", 1))
    revealed = int(parameters.get("extra", 3))
    size_bonus = 0.0
    if "mega" in name:
        size_bonus = 5
    elif "jumbo" in name:
        size_bonus = 2
    return 8 + size_bonus + max(0, choices - 1) * 3 + max(0, revealed - 3)


def choose_celestial(
    boosters: list[dict[str, Any]],
    *,
    legal_indices: set[int],
    context: EconomyContext,
) -> EconomyIntent | None:
    if context.celestials_this_shop >= 1:
        return None
    candidates = [
        card
        for card in boosters
        if int(card.get("index", -1)) in legal_indices
        and "celestial" in str(card.get("name", "")).lower()
        and _affordable(card, context)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda card: (
            _celestial_value(card),
            -int(card.get("cost", 99)),
        ),
        reverse=True,
    )
    return EconomyIntent(
        kind="open_booster",
        target_index=int(candidates[0]["index"]),
    )


def choose_planet_purchase(
    cards: list[dict[str, Any]],
    *,
    legal_indices: set[int],
    context: EconomyContext,
    primary_hand: str,
    poker_hands: dict[str, dict[str, Any]],
    consumable_count: int,
    consumable_slots: int,
    planets_this_shop: int,
) -> EconomyIntent | None:
    if (
        planets_this_shop >= 1
        or consumable_count >= consumable_slots
        or context.ante < 3
    ):
        return None
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for card in cards:
        index = int(card.get("index", -1))
        summary = str((card.get("rule") or {}).get("summary", "")).lower()
        if (
            index not in legal_indices
            or not _affordable(card, context)
            or context.money - int(card.get("cost", 99))
            < context.reserve + 5
            or (
                str(card.get("set", "")).lower() != "planet"
                and "level up" not in summary
            )
        ):
            continue
        hand = next(
            (
                name
                for name in sorted(poker_hands, key=len, reverse=True)
                if f"level up {name.lower()}" in summary
            ),
            "",
        )
        if not hand:
            continue
        history = poker_hands[hand]
        played = int(history.get("played", 0))
        level = int(history.get("level", 1))
        if played < 5 and level < 2:
            continue
        alignment = (
            played * 2
            + max(0, level - 1) * 4
            + (4 if hand == primary_hand else 0)
        )
        ranked.append((alignment, -int(card.get("cost", 99)), card))
    if not ranked:
        return None
    _, _, best = max(ranked, key=lambda item: (item[0], item[1]))
    return EconomyIntent(
        kind="buy_card",
        target_index=int(best["index"]),
    )


def choose_reroll(
    *,
    reroll_legal: bool,
    context: EconomyContext,
) -> EconomyIntent | None:
    if not reroll_legal or context.rerolls_this_shop >= 1:
        return None
    if context.free_rerolls > 0:
        if context.money < context.reserve:
            return None
        if context.joker_count < context.joker_slots and context.ante < 3:
            return None
        return EconomyIntent(kind="reroll_shop")
    if (
        context.ante >= 3
        and context.joker_count >= context.joker_slots
        and context.money - context.reroll_cost
        >= context.reserve + 10
    ):
        return EconomyIntent(kind="reroll_shop")
    return None
