"""Turn visible Joker rules into generic effect profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_NUMBER = re.compile(r"(x|\+)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class EffectContext:
    primary_hand: str
    money: int
    joker_count: int
    deck_remaining: int
    ante: int


@dataclass(frozen=True)
class EffectProfile:
    chips: float
    add_mult: float
    x_mult: float
    retriggers: float
    economy: float
    utility: float
    roles: frozenset[str]
    confidence: float


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _summary(card: dict[str, Any]) -> str:
    return str(_mapping(card.get("rule")).get("summary", "")).lower()


def _parameters(card: dict[str, Any]) -> dict[str, Any]:
    ability = _mapping(card.get("ability"))
    return _mapping(_mapping(card.get("rule")).get("parameters")) or ability


def _copy_joker(card: dict[str, Any]) -> bool:
    summary = _summary(card)
    return "copy" in summary and "joker" in summary


def _card_phase_effect(card: dict[str, Any]) -> bool:
    summary = _summary(card)
    return any(
        token in summary
        for token in (
            "scored card",
            "when scored",
            "held card",
            " held in hand",
            "face card",
            "per card",
        )
    )


def _main_xmult(card: dict[str, Any]) -> bool:
    if card.get("debuffed") or _copy_joker(card) or _card_phase_effect(card):
        return False
    parameters = _parameters(card)
    nested = _mapping(parameters.get("extra"))
    return (
        _number(parameters.get("x_mult"), 1.0) > 1
        or _number(nested.get("Xmult", nested.get("x_mult")), 1.0) > 1
        or "x mult" in _summary(card)
        or "xmult" in _summary(card)
    )


def _main_add_mult(card: dict[str, Any]) -> bool:
    if card.get("debuffed") or _copy_joker(card) or _card_phase_effect(card):
        return False
    parameters = _parameters(card)
    nested = _mapping(parameters.get("extra"))
    return any(
        _number(value) != 0
        for value in (
            parameters.get("t_mult"),
            parameters.get("mult"),
            nested.get("mult"),
            nested.get("s_mult"),
        )
    ) or "+mult" in _summary(card)


def next_main_xmult_swap(jokers: list[dict[str, Any]]) -> int | None:
    """Return a position whose main XMult Joker should move one step right."""

    if any(_copy_joker(joker) for joker in jokers):
        return None
    additive_positions = [
        position
        for position, joker in enumerate(jokers)
        if _main_add_mult(joker)
    ]
    for position in range(len(jokers) - 1, -1, -1):
        if (
            _main_xmult(jokers[position])
            and any(other > position for other in additive_positions)
        ):
            return position
    return None


def _hand_activation(required: str, primary: str) -> float:
    if not required:
        return 1.0
    if required == primary:
        return 1.0
    contained_by = {
        "Pair": {
            "Two Pair",
            "Three of a Kind",
            "Full House",
            "Four of a Kind",
            "Five of a Kind",
            "Flush House",
            "Flush Five",
        },
        "Two Pair": {"Full House", "Flush House"},
        "Three of a Kind": {
            "Full House",
            "Four of a Kind",
            "Five of a Kind",
            "Flush House",
            "Flush Five",
        },
        "Flush": {"Straight Flush", "Flush House", "Flush Five"},
        "Straight": {"Straight Flush"},
    }
    if primary in contained_by.get(required, set()):
        return 0.9
    # A shop purchase can justify pivoting toward a visible hand-specific
    # effect, so a mismatch is discounted without being treated as impossible.
    return 0.4


def _nested_extra(
    extra: dict[str, Any],
) -> tuple[float, float, float, float]:
    chips = 0.0
    mult = 0.0
    x_mult = 1.0
    economy = 0.0
    for key, value in extra.items():
        normalized = key.lower().replace("_", "")
        amount = _number(value)
        if "chip" in normalized:
            chips += amount
        elif normalized in {"mult", "smult", "hmult"}:
            mult += amount
        elif "xmult" in normalized and amount > 1:
            x_mult = max(x_mult, amount)
        elif any(word in normalized for word in ("dollar", "money", "payout")):
            economy += amount
    odds = _number(extra.get("odds"), 1.0)
    if odds > 1:
        probability = 1.0 / odds
        chips *= probability
        mult *= probability
        x_mult = 1.0 + (x_mult - 1.0) * probability
        economy *= probability
    return chips, mult, x_mult, economy


def profile_joker(
    card: dict[str, Any],
    context: EffectContext,
) -> EffectProfile:
    """Approximate one visible Joker without relying on its identity."""

    ability = _mapping(card.get("ability"))
    rule = _mapping(card.get("rule"))
    parameters = _mapping(rule.get("parameters")) or ability
    summary = str(rule.get("summary", "")).lower()
    effect = str(parameters.get("effect", "")).lower()
    required_hand = str(parameters.get("type") or "")
    activation = _hand_activation(required_hand, context.primary_hand)

    chips = (
        _number(parameters.get("t_chips"))
        + _number(parameters.get("bonus"))
        + _number(parameters.get("perma_bonus"))
    ) * activation
    add_mult = (
        _number(parameters.get("t_mult"))
        + _number(parameters.get("mult"))
        + _number(parameters.get("h_mult"))
    ) * activation
    x_mult = max(
        1.0,
        _number(parameters.get("x_mult"), 1.0),
        _number(parameters.get("h_x_mult"), 1.0),
    )
    retriggers = 0.0
    economy = (
        _number(parameters.get("h_dollars"))
        + _number(parameters.get("p_dollars"))
    )
    utility = (
        _number(parameters.get("d_size")) * 5
        + _number(parameters.get("h_size")) * 7
    )
    roles: set[str] = set()
    confidence = 0.8

    raw_extra = parameters.get("extra")
    extra = _number(raw_extra)
    if isinstance(raw_extra, dict):
        nested_chips, nested_mult, nested_x_mult, nested_economy = (
            _nested_extra(raw_extra)
        )
        every = _number(raw_extra.get("every"))
        if every > 0 and nested_x_mult > 1:
            nested_x_mult = (
                1.0 + (nested_x_mult - 1.0) / (every + 1.0)
            )
        elif (
            nested_x_mult > 1
            and any(token in summary for token in (" if ", " when "))
        ):
            nested_x_mult = 1.0 + (nested_x_mult - 1.0) * 0.4
        chips += nested_chips
        add_mult += nested_mult
        x_mult = max(x_mult, nested_x_mult)
        economy += nested_economy

    if required_hand and x_mult > 1:
        x_mult = 1.0 + (x_mult - 1.0) * activation

    if extra:
        if "per joker owned" in summary:
            add_mult += extra * max(1, context.joker_count)
        elif "per $1 held" in summary and "chip" in summary:
            chips += extra * context.money
        elif "per card remaining in deck" in summary and "chip" in summary:
            chips += extra * context.deck_remaining
        elif "total tarot uses" in summary and "mult" in summary:
            add_mult += extra
        elif "x mult" in summary or "xmult" in summary:
            x_activation = 1.0
            if any(token in summary for token in (" if ", " first ", " when ")):
                x_activation = 0.5
            if any(token in summary for token in ("≥", "at least")):
                x_activation = 0.2
            x_mult = max(x_mult, 1.0 + (extra - 1.0) * x_activation)
        elif "face card" in summary and "mult" in summary:
            add_mult += extra * 1.4
        elif "face card" in summary and "chip" in summary:
            chips += extra * 1.4
        elif "even" in summary and "mult" in summary:
            add_mult += extra * 1.5
        elif "odd" in summary and "chip" in summary:
            chips += extra * 1.5
        elif chips == 0 and "chip" in summary:
            chips += extra * activation
        elif add_mult == 0 and "mult" in summary:
            add_mult += extra * activation

    if "retrigger" in summary:
        retriggers = 1.0
    if "level up" in summary:
        utility += 9
        roles.add("scaling")
    if any(word in summary for word in ("scales", "increases", "gains")):
        utility += 5
        roles.add("scaling")
    if "free reroll" in summary or "bonus reroll" in effect:
        utility += 5
        roles.add("economy")
    if any(word in summary for word in ("create tarot", "create planet")):
        utility += 5
        roles.add("generation")
    if "all played cards score" in summary:
        utility += 8
        roles.add("enabler")
    if "copy" in summary and "joker" in summary:
        utility += 38
        roles.add("copy")
    if any(
        word in summary
        for word in ("earn $", "dollars", "money", "sell value")
    ):
        economy += max(1.0, extra)

    match = _NUMBER.search(summary)
    if x_mult == 1.0 and match and match.group(1).lower() == "x":
        x_mult = float(match.group(2))
        if " if " in summary:
            x_mult = 1.0 + (x_mult - 1.0) * 0.4
            confidence = min(confidence, 0.6)

    edition = str(card.get("edition") or "").lower()
    if edition == "foil":
        chips += 50
    elif edition in {"holo", "holographic"}:
        add_mult += 10
    elif edition == "polychrome":
        x_mult *= 1.5
    elif edition == "negative":
        utility += 8
        roles.add("slot")

    if card.get("debuffed"):
        chips = add_mult = retriggers = economy = utility = 0.0
        x_mult = 1.0
        confidence = 1.0

    if chips:
        roles.add("chips")
    if add_mult:
        roles.add("+mult")
    if x_mult > 1:
        roles.add("xmult")
    if retriggers:
        roles.add("retrigger")
    if economy:
        roles.add("economy")
    if utility and not roles:
        roles.add("utility")
    if not roles:
        roles.add("unknown")
        confidence = min(confidence, 0.35)

    return EffectProfile(
        chips=chips,
        add_mult=add_mult,
        x_mult=x_mult,
        retriggers=retriggers,
        economy=economy,
        utility=utility,
        roles=frozenset(roles),
        confidence=confidence,
    )
