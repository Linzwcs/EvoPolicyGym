"""Visible, order-aware hand score approximation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_FACE_RANKS = {"Jack", "Queen", "King"}
_ODD_RANKS = {"Ace", "3", "5", "7", "9"}
_EVEN_RANKS = {"2", "4", "6", "8", "10"}
RANK_VALUE = {
    **{str(value): value for value in range(2, 11)},
    "Jack": 11,
    "Queen": 12,
    "King": 13,
    "Ace": 14,
}
_X_MULT_TEXT = re.compile(r"\bx\s*\d+(?:\.\d+)?")
_LOWEST_HELD_MULT = re.compile(
    r"\+(\d+(?:\.\d+)?)× lowest held card"
)


@dataclass(frozen=True)
class ScoringContext:
    money: int = 0
    deck_remaining: int = 0
    discards_left: int = 0
    hands_left: int = 0
    round_hand_counts: dict[str, int] | None = None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _rank(card: dict[str, Any]) -> str:
    return str(card.get("rank") or "")


def _suit(card: dict[str, Any]) -> str:
    return str(card.get("suit") or "")


def _mentions_x_mult(summary: str) -> bool:
    return (
        "x mult" in summary
        or "xmult" in summary
        or _X_MULT_TEXT.search(summary) is not None
    )


def hand_contains(actual: str, required: str) -> bool:
    if not required or actual == required:
        return True
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
        "Straight": {"Straight Flush"},
        "Flush": {"Straight Flush", "Flush House", "Flush Five"},
    }
    return actual in contained_by.get(required, set())


def _card_values(card: dict[str, Any]) -> tuple[float, float, float]:
    if card.get("debuffed"):
        return 0.0, 0.0, 1.0
    ability = _mapping(card.get("ability"))
    chips = (
        _number(card.get("chips"))
        + _number(ability.get("bonus"))
        + _number(ability.get("perma_bonus"))
    )
    mult = _number(ability.get("mult"))
    x_mult = 1.0
    edition = str(card.get("edition") or "").lower()
    if edition == "foil":
        chips += 50
    elif edition in {"holo", "holographic"}:
        mult += 10
    elif edition == "polychrome":
        x_mult *= 1.5
    enhancement = str(card.get("enhancement") or "").lower()
    if enhancement in {"m_glass", "glass card"}:
        x_mult *= 2
    return chips, mult, x_mult


def _retriggered_scoring(
    scoring: list[dict[str, Any]],
    jokers: list[dict[str, Any]],
    context: ScoringContext,
) -> list[dict[str, Any]]:
    repetitions = [1] * len(scoring)
    for joker in jokers:
        if joker.get("debuffed"):
            continue
        parameters = _mapping(_mapping(joker.get("rule")).get("parameters"))
        if not parameters:
            parameters = _mapping(joker.get("ability"))
        summary = str(
            _mapping(joker.get("rule")).get("summary", "")
        ).lower()
        if "retrigger" not in summary:
            continue
        extra = max(1, int(_number(parameters.get("extra"), 1.0)))
        if "first scored card" in summary and repetitions:
            repetitions[0] += extra
        elif "2/3/4/5" in summary:
            for index, card in enumerate(scoring):
                if _rank(card) in {"2", "3", "4", "5"}:
                    repetitions[index] += extra
        elif "face card" in summary:
            for index, card in enumerate(scoring):
                if _rank(card) in _FACE_RANKS:
                    repetitions[index] += extra
        elif "final hand" in summary and context.hands_left == 1:
            repetitions = [count + extra for count in repetitions]

    expanded: list[dict[str, Any]] = []
    for card, count in zip(scoring, repetitions, strict=True):
        expanded.extend([card] * count)
    return expanded


def _effective_jokers(
    jokers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    effective: list[dict[str, Any]] = []
    for position, joker in enumerate(jokers):
        summary = str(
            _mapping(joker.get("rule")).get("summary", "")
        ).lower()
        if (
            not joker.get("debuffed")
            and "copy the joker to the right" in summary
            and position + 1 < len(jokers)
        ):
            copied = dict(jokers[position + 1])
            copied["edition"] = joker.get("edition")
            effective.append(copied)
        else:
            effective.append(joker)
    return effective


def _condition_applies(
    summary: str,
    *,
    hand_name: str,
    selected: list[dict[str, Any]],
    scoring: list[dict[str, Any]],
    held: list[dict[str, Any]],
    context: ScoringContext,
) -> bool:
    if "same hand type played twice this round" in summary:
        counts = context.round_hand_counts or {}
        return counts.get(hand_name, 0) >= 1
    if "3 or fewer" in summary:
        return len(selected) <= 3
    if "≤3" in summary:
        return len(selected) <= 3
    if "0 discard" in summary:
        return context.discards_left == 0
    if "final hand" in summary or "last hand" in summary:
        return context.hands_left == 1
    if "face card" in summary:
        return any(_rank(card) in _FACE_RANKS for card in scoring)
    if "all 4 suits" in summary:
        return len({_suit(card) for card in scoring}) == 4
    if "all held cards are spades or clubs" in summary:
        return bool(held) and all(
            _suit(card) in {"Spades", "Clubs"} for card in held
        )
    if "club + card of another suit" in summary:
        suits = {_suit(card) for card in scoring}
        return "Clubs" in suits and len(suits) >= 2
    if any(
        token in summary
        for token in (" if ", " when ", " only")
    ):
        return False
    return True


def _per_card_effect(
    *,
    parameters: dict[str, Any],
    summary: str,
    scoring: list[dict[str, Any]],
    held: list[dict[str, Any]],
) -> tuple[float, float, float]:
    raw_extra = parameters.get("extra")
    extra = _number(raw_extra)
    nested = _mapping(raw_extra)
    chips = 0.0
    mult = 0.0
    x_mult = 1.0

    suit = str(nested.get("suit") or "")
    if not suit and "scored" in summary:
        for plural, singular in (
            ("Hearts", "heart"),
            ("Diamonds", "diamond"),
            ("Spades", "spade"),
            ("Clubs", "club"),
        ):
            if singular in summary:
                suit = plural
                break
    suit_cards = [card for card in scoring if _suit(card) == suit]
    if suit and suit_cards:
        suit_mult = _number(nested.get("s_mult"), extra)
        suit_chips = _number(nested.get("s_chips"), extra)
        if "mult" in summary:
            mult += suit_mult * len(suit_cards)
        if "chip" in summary:
            chips += suit_chips * len(suit_cards)

    raw_ranks = nested.get("ranks")
    allowed_ranks: set[str] = set()
    if isinstance(raw_ranks, list):
        allowed_ranks = {str(rank) for rank in raw_ranks}
    elif "for ace" in summary:
        allowed_ranks = {"Ace"}
    elif "for 10 or 4" in summary:
        allowed_ranks = {"10", "4"}
    if allowed_ranks:
        count = sum(_rank(card) in allowed_ranks for card in scoring)
        chips += _number(nested.get("chips")) * count
        mult += _number(nested.get("mult")) * count

    face_count = sum(_rank(card) in _FACE_RANKS for card in scoring)
    if face_count:
        if "face card" in summary and "chip" in summary:
            chips += extra * face_count
        if (
            "face card" in summary
            and "mult" in summary
            and not _mentions_x_mult(summary)
        ):
            mult += extra * face_count
        if _mentions_x_mult(summary) and "first" in summary:
            first_face = next(
                card for card in scoring if _rank(card) in _FACE_RANKS
            )
            triggers = sum(card is first_face for card in scoring)
            x_mult *= max(1.0, extra) ** triggers

    if "even" in summary and "mult" in summary:
        mult += extra * sum(_rank(card) in _EVEN_RANKS for card in scoring)
    if "odd" in summary and "chip" in summary:
        chips += extra * sum(_rank(card) in _ODD_RANKS for card in scoring)
    lowest_held = _LOWEST_HELD_MULT.search(summary)
    if lowest_held and held:
        held_values = [
            value
            for card in held
            if (value := RANK_VALUE.get(_rank(card), 0)) > 0
            and not card.get("debuffed")
        ]
        if held_values:
            mult += float(lowest_held.group(1)) * min(held_values)

    for rank in ("Ace", "King", "Queen", "Jack"):
        if f"per {rank.lower()} held" not in summary:
            continue
        count = sum(
            _rank(card) == rank and not card.get("debuffed")
            for card in held
        )
        if "x mult" in summary:
            x_mult *= max(1.0, extra) ** count
        elif "mult" in summary:
            mult += extra * count
    return chips, mult, x_mult


def _dynamic_effect(
    *,
    parameters: dict[str, Any],
    summary: str,
    joker_count: int,
    context: ScoringContext,
    condition_active: bool,
    per_card_handled: bool,
) -> tuple[float, float, float]:
    raw_extra = parameters.get("extra")
    extra = _number(raw_extra)
    nested = _mapping(raw_extra)
    chips = 0.0
    mult = 0.0
    x_mult = 1.0

    if "per joker owned" in summary and "mult" in summary:
        mult += extra * joker_count
    elif (
        "per $" in summary
        and "mult" in summary
        and _number(nested.get("dollars")) > 0
    ):
        mult += (
            context.money // int(_number(nested.get("dollars")))
        ) * _number(nested.get("mult"))
    elif "per $1 held" in summary and "chip" in summary:
        chips += extra * context.money
    elif "per card remaining in deck" in summary and "chip" in summary:
        chips += extra * context.deck_remaining
    elif "per discard remaining" in summary and "chip" in summary:
        chips += extra * context.discards_left
    elif "total tarot uses" in summary and "mult" in summary:
        mult += extra
    elif "random mult between" in summary:
        mult += (
            _number(nested.get("min")) + _number(nested.get("max"))
        ) / 2
    elif (
        condition_active
        and not per_card_handled
        and ("chips" in nested or "chip_mod" in nested)
        and "ranks" not in nested
        and "suit" not in nested
    ):
        chips += _number(nested.get("chips"), _number(nested.get("chip_mod")))
    elif (
        condition_active
        and not per_card_handled
        and "mult" in nested
        and "ranks" not in nested
        and "suit" not in nested
    ):
        mult += _number(nested.get("mult"))

    nested_x = _number(
        nested.get("Xmult", nested.get("x_mult")),
        1.0,
    )
    recurring = _number(nested.get("every")) > 0
    remaining_value = nested.get("remaining")
    if isinstance(remaining_value, str):
        remaining_text = remaining_value.strip().split(maxsplit=1)[0]
        remaining = (
            float(remaining_text)
            if remaining_text.lstrip("-").isdigit()
            else None
        )
    elif isinstance(remaining_value, (int, float)) and not isinstance(
        remaining_value,
        bool,
    ):
        remaining = float(remaining_value)
    else:
        remaining = None
    recurring_ready = not recurring or remaining == 0
    if nested_x > 1 and recurring_ready and condition_active:
        odds = max(1.0, _number(nested.get("odds"), 1.0))
        x_mult *= 1.0 + (nested_x - 1.0) / odds
    return chips, mult, x_mult


def estimate_score(
    *,
    hand: list[dict[str, Any]],
    selected_indices: list[int],
    scoring_local: list[int],
    hand_name: str,
    poker_hand: dict[str, Any],
    jokers: list[dict[str, Any]],
    context: ScoringContext,
) -> float:
    jokers = _effective_jokers(jokers)
    selected = [hand[index] for index in selected_indices]
    if any(
        "all played cards score"
        in str((_mapping(joker.get("rule"))).get("summary", "")).lower()
        for joker in jokers
        if not joker.get("debuffed")
    ):
        scoring_local = list(range(len(selected)))
    scoring = [selected[index] for index in scoring_local]
    scoring = _retriggered_scoring(scoring, jokers, context)
    selected_set = set(selected_indices)
    held = [
        card for index, card in enumerate(hand) if index not in selected_set
    ]

    chips = _number(poker_hand.get("chips"), 5.0)
    mult = _number(poker_hand.get("mult"), 1.0)
    for card in scoring:
        card_chips, card_mult, card_x_mult = _card_values(card)
        chips += card_chips
        mult += card_mult
        mult *= card_x_mult

    for joker_position, joker in enumerate(jokers):
        if joker.get("debuffed"):
            continue
        parameters = _mapping(_mapping(joker.get("rule")).get("parameters"))
        if not parameters:
            parameters = _mapping(joker.get("ability"))
        summary = str(
            _mapping(joker.get("rule")).get("summary", "")
        ).lower()
        required = str(parameters.get("type") or "")
        active = hand_contains(hand_name, required)
        sell_value_mult = (
            "sum of all other jokers' sell values" in summary
        )
        condition_active = _condition_applies(
            summary,
            hand_name=hand_name,
            selected=selected,
            scoring=scoring,
            held=held,
            context=context,
        )

        edition = str(joker.get("edition") or "").lower()
        if edition == "foil":
            chips += 50
        elif edition in {"holo", "holographic"}:
            mult += 10
        elif edition == "polychrome":
            mult *= 1.5

        if active:
            chips += (
                _number(parameters.get("t_chips"))
                + _number(parameters.get("bonus"))
                + _number(parameters.get("perma_bonus"))
            )
            mult += (
                _number(parameters.get("t_mult"))
                + (
                    0.0
                    if sell_value_mult
                    else _number(parameters.get("mult"))
                )
                + _number(parameters.get("h_mult"))
            )
            if sell_value_mult:
                mult += sum(
                    _number(other.get("sell_value"))
                    for position, other in enumerate(jokers)
                    if position != joker_position
                )

        per_chips, per_mult, per_x = _per_card_effect(
            parameters=parameters,
            summary=summary,
            scoring=scoring,
            held=held,
        )
        chips += per_chips
        mult += per_mult
        mult *= per_x

        dynamic_chips, dynamic_mult, dynamic_x = _dynamic_effect(
            parameters=parameters,
            summary=summary,
            joker_count=len(jokers),
            context=context,
            condition_active=condition_active,
            per_card_handled=bool(
                per_chips or per_mult or per_x > 1
            ),
        )
        chips += dynamic_chips
        mult += dynamic_mult
        mult *= dynamic_x

        raw_extra = parameters.get("extra")
        extra = _number(raw_extra)
        history_handled = (
            "times this hand type played" in summary
            and isinstance(poker_hand.get("played"), int)
        )
        if history_handled:
            mult += max(1.0, extra) * (
                int(poker_hand.get("played", 0)) + 1
            )
        structurally_per_card = any(
            token in summary
            for token in (
                " per ",
                "face card",
                "even",
                "odd",
                " scored",
            )
        )
        if (
            extra
            and condition_active
            and not structurally_per_card
            and not history_handled
        ):
            if (
                "mult" in summary
                and not _mentions_x_mult(summary)
                and not per_mult
                and not dynamic_mult
                and not _number(parameters.get("t_mult"))
            ):
                mult += extra
            elif (
                "chip" in summary
                and not per_chips
                and not dynamic_chips
                and not _number(parameters.get("t_chips"))
            ):
                chips += extra
            elif _mentions_x_mult(summary) and not per_x > 1:
                mult *= max(1.0, extra)

        if active:
            x_mult = max(
                1.0,
                _number(parameters.get("x_mult"), 1.0),
                _number(parameters.get("h_x_mult"), 1.0),
            )
            mult *= x_mult

    return chips * mult
