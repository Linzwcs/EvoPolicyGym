"""Build-aware Joker valuation and shop upgrade selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .effects import EffectContext, EffectProfile, profile_joker


@dataclass(frozen=True)
class JokerValue:
    card: dict[str, Any]
    profile: EffectProfile
    value: float


@dataclass(frozen=True)
class ShopUpgrade:
    candidate: JokerValue
    replaced: JokerValue | None
    gain: float


def acquisition_score(card: dict[str, Any]) -> float:
    """Preserve the baseline's open-slot acquisition behavior."""

    ability = card.get("ability") or {}
    summary = str((card.get("rule") or {}).get("summary", "")).lower()
    score = float(ability.get("t_chips") or ability.get("bonus") or 0) / 12
    score += float(ability.get("t_mult") or ability.get("mult") or 0)
    score += max(0.0, float(ability.get("x_mult") or 1) - 1) * 20
    for word, value in (
        ("chips", 3),
        ("mult", 4),
        ("retrigger", 5),
        ("copy", 9),
        ("scales", 5),
        ("each played", 3),
        ("contains", 2),
    ):
        if word in summary:
            score += value
    if any(
        phrase in summary for phrase in ("earn $", "sell value", "reroll")
    ):
        score -= 3
    if card.get("edition") in ("foil", "holo", "polychrome", "negative"):
        score += 8
    return score


def effect_context(
    *,
    primary_hand: str,
    money: int,
    jokers: list[dict[str, Any]],
    deck_remaining: int,
    ante: int,
) -> EffectContext:
    return EffectContext(
        primary_hand=primary_hand,
        money=money,
        joker_count=len(jokers),
        deck_remaining=deck_remaining,
        ante=ante,
    )


def _profile_value(profile: EffectProfile, *, owned: bool) -> float:
    value = (
        profile.chips / 10
        + profile.add_mult * 1.8
        + (profile.x_mult - 1) * 28
        + profile.retriggers * 12
        + profile.economy
        + profile.utility
    )
    value *= 0.55 + 0.45 * profile.confidence
    if owned and "unknown" in profile.roles:
        value += 6
    return value


def value_joker(
    card: dict[str, Any],
    context: EffectContext,
    *,
    owned: bool,
) -> JokerValue:
    profile = profile_joker(card, context)
    return JokerValue(
        card=card,
        profile=profile,
        value=_profile_value(profile, owned=owned),
    )


def _role_multiplier(
    candidate: JokerValue,
    owned: list[JokerValue],
) -> float:
    covered = {
        role
        for joker in owned
        for role in joker.profile.roles
        if role != "unknown"
    }
    missing = candidate.profile.roles - covered - {"unknown"}
    multiplier = 1.0 + min(0.3, len(missing) * 0.12)
    if "xmult" in missing and {"+mult", "chips"} <= covered:
        multiplier += 0.12
    return multiplier


def choose_joker_upgrade(
    *,
    shop_cards: list[dict[str, Any]],
    owned_cards: list[dict[str, Any]],
    sellable_indices: set[int],
    slots: int,
    money: int,
    context: EffectContext,
    reserve: int = 0,
) -> ShopUpgrade | None:
    """Return a worthwhile affordable addition or replacement."""

    owned = [
        value_joker(card, context, owned=True) for card in owned_cards
    ]
    candidates = []
    for card in shop_cards:
        if card.get("set") != "Joker":
            continue
        candidate = value_joker(card, context, owned=False)
        candidate = JokerValue(
            card=candidate.card,
            profile=candidate.profile,
            value=candidate.value * _role_multiplier(candidate, owned),
        )
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (item.value, -int(item.card.get("cost", 99))),
        reverse=True,
    )

    has_slot = len(owned_cards) < slots
    if has_slot:
        affordable = [
            item
            for item in candidates
            if int(item.card.get("cost", 99)) <= money
            and money - int(item.card.get("cost", 99)) >= reserve
        ]
        if not affordable:
            return None
        affordable.sort(
            key=lambda item: (
                acquisition_score(item.card),
                -int(item.card.get("cost", 0)),
            ),
            reverse=True,
        )
        best = affordable[0]
        if acquisition_score(best.card) <= 1 and owned_cards:
            return None
        return ShopUpgrade(candidate=best, replaced=None, gain=best.value)

    sellable = [
        item
        for item in owned
        if int(item.card.get("index", -1)) in sellable_indices
        and not item.card.get("eternal")
    ]
    if not sellable:
        return None
    weakest = min(sellable, key=lambda item: item.value)
    affordable_after_sale = [
        item
        for item in candidates
        if int(item.card.get("cost", 99))
        <= money + int(weakest.card.get("sell_value", 0))
        and money + int(weakest.card.get("sell_value", 0))
        - int(item.card.get("cost", 99)) >= reserve
    ]
    if not affordable_after_sale:
        return None
    best = affordable_after_sale[0]
    gain = best.value - weakest.value
    required_gain = max(4.0, weakest.value * 0.2)
    if gain < required_gain:
        return None
    return ShopUpgrade(candidate=best, replaced=weakest, gain=gain)
