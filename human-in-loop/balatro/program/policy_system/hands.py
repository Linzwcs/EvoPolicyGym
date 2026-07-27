"""Pure poker hand classification, enumeration, and visible score estimates."""

from __future__ import annotations

import itertools
from typing import Any

from .scoring import ScoringContext, estimate_score

RANK = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 11,
    "Queen": 12,
    "King": 13,
    "Ace": 14,
}


def _draw_keep_indices(
    hand: list[dict[str, Any]],
    best_indices: list[int],
) -> set[int]:
    keep = set(best_indices)
    suits: dict[str, list[int]] = {}
    ranks: dict[str, list[int]] = {}
    for index, card in enumerate(hand):
        suit = card.get("suit")
        rank = card.get("rank")
        if suit:
            suits.setdefault(str(suit), []).append(index)
        if rank:
            ranks.setdefault(str(rank), []).append(index)
    for group in suits.values():
        if len(group) >= 4:
            keep.update(group)
    for group in ranks.values():
        if len(group) >= 2:
            keep.update(group)
    return keep


def classify(cards: list[dict[str, Any]]) -> tuple[str, list[int]]:
    ranks = [RANK.get(str(c.get("rank")), 0) for c in cards]
    by_rank = {
        rank: [index for index, value in enumerate(ranks) if value == rank]
        for rank in set(ranks)
        if rank
    }
    groups = sorted(
        by_rank.values(),
        key=lambda indices: (len(indices), ranks[indices[0]]),
        reverse=True,
    )
    suits = [str(card.get("suit") or "") for card in cards]
    flush = (
        len(cards) == 5
        and all(suits)
        and len(set(suits)) == 1
    )
    unique = sorted(set(ranks))
    straight = len(cards) == 5 and len(unique) == 5 and (
        unique[-1] - unique[0] == 4 or unique == [2, 3, 4, 5, 14]
    )
    counts = sorted((len(indices) for indices in groups), reverse=True)
    if straight and flush:
        return "Straight Flush", list(range(5))
    if counts == [4, 1]:
        return "Four of a Kind", groups[0]
    if counts == [3, 2]:
        return "Full House", groups[0] + groups[1]
    if flush:
        return "Flush", list(range(5))
    if straight:
        return "Straight", list(range(5))
    if counts and counts[0] == 3:
        return "Three of a Kind", groups[0]
    if counts[:2] == [2, 2]:
        return "Two Pair", groups[0] + groups[1]
    if counts and counts[0] == 2:
        return "Pair", groups[0]
    high = max(range(len(cards)), key=lambda index: ranks[index])
    return "High Card", [high]


def candidates(
    hand: list[dict[str, Any]],
    poker: dict[str, dict[str, Any]],
    jokers: list[dict[str, Any]],
    *,
    money: int = 0,
    deck_remaining: int = 0,
    discards_left: int = 0,
    hands_left: int = 0,
    round_hand_counts: dict[str, int] | None = None,
) -> list[tuple[float, str, list[int]]]:
    result = []
    context = ScoringContext(
        money=money,
        deck_remaining=deck_remaining,
        discards_left=discards_left,
        hands_left=hands_left,
        round_hand_counts=round_hand_counts,
    )
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(range(len(hand)), size):
            cards = [hand[index] for index in combo]
            name, scoring_local = classify(cards)
            score = estimate_score(
                hand=hand,
                selected_indices=list(combo),
                scoring_local=scoring_local,
                hand_name=name,
                poker_hand=poker.get(name, {}),
                jokers=jokers,
                context=context,
            )
            result.append(
                (
                    score - 0.05 * (size - len(scoring_local)),
                    name,
                    list(combo),
                )
            )
    return sorted(result, reverse=True)


def discard_choice(
    hand: list[dict[str, Any]],
    best_indices: list[int],
) -> list[int]:
    keep = _draw_keep_indices(hand, best_indices)
    junk = [index for index in range(len(hand)) if index not in keep]
    if not junk:
        junk = sorted(
            range(len(hand)),
            key=lambda index: RANK.get(str(hand[index].get("rank")), 0),
        )[:3]
    return junk[:5]


def cycle_play_choice(
    hand: list[dict[str, Any]],
    ranked: list[tuple[float, str, list[int]]],
    *,
    score: float,
    hand_name: str,
    selected_indices: list[int],
) -> list[int]:
    """Add safe non-scoring cards to improve next-hand draw throughput."""

    capacity = 5 - len(selected_indices)
    if capacity <= 0:
        return selected_indices

    keep = _draw_keep_indices(hand, selected_indices)
    unmodified = {"", "base", "c_base", "default base"}
    extras = [
        index
        for index, card in enumerate(hand)
        if index not in keep
        and not card.get("edition")
        and not card.get("seal")
        and str(card.get("enhancement") or "").lower() in unmodified
    ]
    extras.sort(
        key=lambda index: (
            RANK.get(str(hand[index].get("rank")), 0),
            index,
        )
    )

    for count in range(min(capacity, len(extras)), 0, -1):
        desired = set(selected_indices)
        desired.update(extras[:count])
        minimum_score = score - 0.05 * count - 1e-9
        for candidate_score, candidate_name, candidate_indices in ranked:
            if (
                candidate_name == hand_name
                and candidate_score >= minimum_score
                and set(candidate_indices) == desired
            ):
                return candidate_indices
    return selected_indices
