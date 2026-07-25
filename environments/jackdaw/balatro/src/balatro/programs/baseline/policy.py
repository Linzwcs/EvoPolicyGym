"""Deterministic poker-aware starting Policy for the Balatro Benchmark."""

from __future__ import annotations

import itertools
from typing import Any, cast

from evopolicygym.policy import PolicyContext, PolicyValue

_RANKS = {
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


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        state = cast(dict[str, Any], observation)
        phase = state["phase"]

        if phase == "blind_select":
            return {"kind": "select_blind"}
        if phase == "round_eval":
            return {"kind": "cash_out"}
        if phase == "shop":
            return self._shop(state)
        if phase == "pack_opening":
            return self._pack(state)
        if phase == "selecting_hand":
            hand = cast(list[dict[str, Any]], state["hand"])
            return {
                "kind": "play_hand",
                "card_indices": list(_best_hand(hand)),
            }
        raise RuntimeError(f"unexpected phase: {phase!r}")

    def _shop(self, state: dict[str, Any]) -> PolicyValue:
        shop = cast(dict[str, Any], state["shop"])
        resources = cast(dict[str, Any], state["resources"])
        jokers = cast(list[dict[str, Any]], state["jokers"])
        if len(jokers) < int(resources["joker_slots"]):
            for card in cast(list[dict[str, Any]], shop["cards"]):
                if card["set"] == "Joker" and int(card["cost"]) <= int(resources["money"]):
                    return {
                        "kind": "buy_card",
                        "target_index": int(card["index"]),
                    }
        return {"kind": "next_round"}

    def _pack(self, state: dict[str, Any]) -> PolicyValue:
        pack = cast(dict[str, Any], state["pack"])
        for card in cast(list[dict[str, Any]], pack["cards"]):
            if card["set"] == "Joker":
                return {
                    "kind": "pick_pack_card",
                    "target_index": int(card["index"]),
                    "card_indices": [],
                }
        return {"kind": "skip_pack"}


def _best_hand(hand: list[dict[str, Any]]) -> tuple[int, ...]:
    if not hand:
        raise ValueError("hand must be non-empty")
    best: tuple[int, ...] = (0,)
    best_value = _hand_value(hand, best)
    for size in range(1, min(5, len(hand)) + 1):
        for indices in itertools.combinations(range(len(hand)), size):
            value = _hand_value(hand, indices)
            if value > best_value:
                best = indices
                best_value = value
    return best


def _hand_value(
    hand: list[dict[str, Any]],
    indices: tuple[int, ...],
) -> tuple[int, int, int]:
    cards = [hand[index] for index in indices]
    ranks = [_RANKS.get(str(card.get("rank")), 0) for card in cards]
    suits = [str(card.get("suit")) for card in cards]
    counts = sorted(
        (ranks.count(rank) for rank in set(ranks) if rank),
        reverse=True,
    )
    flush = len(cards) == 5 and len(set(suits)) == 1
    unique = sorted(set(ranks))
    straight = (
        len(cards) == 5
        and len(unique) == 5
        and (unique[-1] - unique[0] == 4 or unique == [2, 3, 4, 5, 14])
    )
    if straight and flush:
        category = 8
    elif counts == [4, 1]:
        category = 7
    elif counts == [3, 2]:
        category = 6
    elif flush:
        category = 5
    elif straight:
        category = 4
    elif counts and counts[0] == 3:
        category = 3
    elif counts[:2] == [2, 2]:
        category = 2
    elif counts and counts[0] == 2:
        category = 1
    else:
        category = 0
    return category, sum(ranks), -len(indices)


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()
