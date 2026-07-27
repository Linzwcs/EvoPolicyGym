"""Strict construction and admission of legal Actions."""

from __future__ import annotations

from typing import Any

_SIMPLE = {
    "select_blind",
    "skip_blind",
    "cash_out",
    "reroll_shop",
    "next_round",
    "skip_pack",
    "sort_hand_by_rank",
    "sort_hand_by_suit",
}
_ENTITY = {
    "buy_card",
    "sell_joker",
    "sell_consumable",
    "redeem_voucher",
    "open_booster",
    "swap_joker_left",
    "swap_joker_right",
    "swap_hand_left",
    "swap_hand_right",
}


class ActionGateway:
    def __init__(self, legal: dict[str, dict[str, Any]]) -> None:
        self.legal = legal

    def simple(self, kind: str) -> dict[str, Any] | None:
        if kind in _SIMPLE and kind in self.legal:
            return {"kind": kind}
        return None

    def entity(self, kind: str, target: int) -> dict[str, Any] | None:
        desc = self.legal.get(kind)
        if kind not in _ENTITY or not desc:
            return None
        if target not in [int(x) for x in desc.get("target_indices", [])]:
            return None
        return {"kind": kind, "target_index": target}

    def cards(self, kind: str, indices: list[int]) -> dict[str, Any] | None:
        desc = self.legal.get(kind)
        if kind not in {"play_hand", "discard"} or not desc:
            return None
        allowed = {int(x) for x in desc.get("card_indices", [])}
        if len(indices) != len(set(indices)) or not set(indices) <= allowed:
            return None
        if not int(desc.get("min_cards", 1)) <= len(indices) <= int(
            desc.get("max_cards", 5)
        ):
            return None
        return {"kind": kind, "card_indices": indices}

    def target_cards(
        self,
        kind: str,
        target: int,
        indices: list[int],
    ) -> dict[str, Any] | None:
        desc = self.legal.get(kind)
        if kind not in {"use_consumable", "pick_pack_card"} or not desc:
            return None
        targets = desc.get("targets") or []
        spec = next(
            (x for x in targets if int(x.get("target_index", -1)) == target),
            None,
        )
        if spec is None:
            if (
                target not in [int(x) for x in desc.get("target_indices", [])]
                or indices
            ):
                return None
        else:
            allowed = {int(x) for x in spec.get("card_indices", [])}
            if len(indices) != len(set(indices)) or not set(indices) <= allowed:
                return None
            if not int(spec.get("min_cards", 0)) <= len(indices) <= int(
                spec.get("max_cards", 0)
            ):
                return None
        return {
            "kind": kind,
            "target_index": target,
            "card_indices": indices,
        }
