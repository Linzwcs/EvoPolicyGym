"""Normalized observation access and episode-local planning state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodePlan:
    primary_hand: str = "Pair"
    secondary_hand: str = "High Card"
    confidence: float = 0.5
    used_discard_this_round: bool = False
    pending_purchase_key: str | None = None
    shop_round: int = -1
    rerolls_this_shop: int = 0
    celestials_this_shop: int = 0
    planets_this_shop: int = 0
    hand_round_id: int = -1
    played_hand_types: set[str] | None = None
    played_hand_counts: dict[str, int] | None = None
    consecutive_discards: int = 0

    def prepare_round(self, round_id: int) -> None:
        if self.hand_round_id != round_id:
            self.hand_round_id = round_id
            self.played_hand_types = set()
            self.played_hand_counts = {}
            self.consecutive_discards = 0


class StateView:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.phase = str(raw.get("phase", ""))
        self.hand = list(raw.get("hand") or [])
        self.jokers = list(raw.get("jokers") or [])
        self.consumables = list(raw.get("consumables") or [])
        self.resources = dict(raw.get("resources") or {})
        self.progress = dict(raw.get("progress") or {})
        self.blind = dict(raw.get("blind") or {})
        self.shop = dict(raw.get("shop") or {})
        self.pack = dict(raw.get("pack") or {})
        self.deck = dict(raw.get("deck") or {})
        self.poker_hands = {
            str(item.get("name")): item for item in (raw.get("poker_hands") or [])
        }
        self.legal = {
            str(item.get("kind")): item for item in (raw.get("legal_actions") or [])
        }

    def money(self) -> int:
        return int(self.resources.get("money", 0))

    def remaining(self) -> int:
        return max(
            0,
            int(self.blind.get("target_chips", 0))
            - int(self.resources.get("chips", 0)),
        )
