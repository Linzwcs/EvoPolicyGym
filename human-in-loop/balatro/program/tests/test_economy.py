"""Tests for budgeted long-horizon shop spending."""

from __future__ import annotations

import unittest
from typing import Any

from policy_system.economy import (
    EconomyContext,
    choose_celestial,
    choose_planet_purchase,
    choose_reroll,
    choose_voucher,
)
from policy_system.strategy import BalatroPolicy


def _context(**overrides: int) -> EconomyContext:
    values = {
        "money": 40,
        "ante": 3,
        "reroll_cost": 5,
        "free_rerolls": 0,
        "rerolls_this_shop": 0,
        "celestials_this_shop": 0,
        "joker_count": 5,
        "joker_slots": 5,
    }
    values.update(overrides)
    return EconomyContext(**values)


def _shop_card(
    *,
    index: int,
    name: str,
    cost: int,
    card_set: str,
    summary: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "key": name.lower().replace(" ", "_"),
        "name": name,
        "cost": cost,
        "set": card_set,
        "rule": {
            "summary": summary,
            "parameters": parameters or {},
        },
        "ability": parameters or {},
    }


class EconomyTest(unittest.TestCase):
    def test_direct_planet_purchase_prefers_established_hand(self) -> None:
        pair = _shop_card(
            index=0,
            name="Pair Planet",
            cost=3,
            card_set="Planet",
            summary="Level up Pair by 1.",
        )
        high_card = _shop_card(
            index=1,
            name="High Card Planet",
            cost=3,
            card_set="Planet",
            summary="Level up High Card by 1.",
        )

        intent = choose_planet_purchase(
            [pair, high_card],
            legal_indices={0, 1},
            context=_context(money=25, ante=3),
            primary_hand="High Card",
            poker_hands={
                "Pair": {"played": 2, "level": 1},
                "High Card": {"played": 8, "level": 2},
            },
            consumable_count=0,
            consumable_slots=2,
            planets_this_shop=0,
        )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.kind, "buy_card")
        self.assertEqual(intent.target_index, 1)

    def test_direct_planet_purchase_requires_slot_and_reserve(self) -> None:
        planet = _shop_card(
            index=0,
            name="Pair Planet",
            cost=3,
            card_set="Planet",
            summary="Level up Pair by 1.",
        )
        full = choose_planet_purchase(
            [planet],
            legal_indices={0},
            context=_context(),
            primary_hand="Pair",
            poker_hands={"Pair": {"played": 3, "level": 1}},
            consumable_count=2,
            consumable_slots=2,
            planets_this_shop=0,
        )
        poor = choose_planet_purchase(
            [planet],
            legal_indices={0},
            context=_context(money=17, ante=3),
            primary_hand="Pair",
            poker_hands={"Pair": {"played": 3, "level": 1}},
            consumable_count=0,
            consumable_slots=2,
            planets_this_shop=0,
        )

        self.assertIsNone(full)
        self.assertIsNone(poor)

    def test_high_value_voucher_beats_no_effect_prerequisite(self) -> None:
        blank = _shop_card(
            index=0,
            name="Blank",
            cost=10,
            card_set="Voucher",
            summary="No immediate gameplay effect; prerequisite.",
        )
        hand = _shop_card(
            index=1,
            name="More Hands",
            cost=10,
            card_set="Voucher",
            summary="Permanently add 1 hand per round.",
        )

        intent = choose_voucher(
            [blank, hand],
            legal_indices={0, 1},
            context=_context(),
        )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.kind, "redeem_voucher")
        self.assertEqual(intent.target_index, 1)

    def test_celestial_pack_preserves_interest_reserve(self) -> None:
        pack = _shop_card(
            index=0,
            name="Celestial Pack",
            cost=4,
            card_set="Booster",
            summary="Reveal 3 Planet cards and choose 1.",
            parameters={"choose": 1, "extra": 3},
        )

        affordable = choose_celestial(
            [pack],
            legal_indices={0},
            context=_context(money=20, ante=3),
        )
        too_expensive = choose_celestial(
            [pack],
            legal_indices={0},
            context=_context(money=18, ante=3),
        )
        already_opened = choose_celestial(
            [pack],
            legal_indices={0},
            context=_context(celestials_this_shop=1),
        )

        self.assertIsNotNone(affordable)
        self.assertIsNone(too_expensive)
        self.assertIsNone(already_opened)

    def test_reroll_is_limited_and_budgeted(self) -> None:
        self.assertIsNone(
            choose_reroll(
                reroll_legal=True,
                context=_context(money=29),
            )
        )
        self.assertIsNotNone(
            choose_reroll(
                reroll_legal=True,
                context=_context(free_rerolls=1),
            )
        )
        self.assertIsNone(
            choose_reroll(
                reroll_legal=True,
                context=_context(free_rerolls=1, rerolls_this_shop=1),
            )
        )
        self.assertIsNone(
            choose_reroll(
                reroll_legal=True,
                context=_context(money=14, free_rerolls=1),
            )
        )

    def test_paid_reroll_requires_full_build_and_large_surplus(self) -> None:
        paid = choose_reroll(
            reroll_legal=True,
            context=_context(money=35, ante=4),
        )
        short = choose_reroll(
            reroll_legal=True,
            context=_context(money=34, ante=4),
        )
        open_slot = choose_reroll(
            reroll_legal=True,
            context=_context(
                money=35,
                ante=4,
                joker_count=4,
                joker_slots=5,
            ),
        )

        self.assertIsNotNone(paid)
        self.assertIsNone(short)
        self.assertIsNone(open_slot)

    def test_planet_pack_prefers_established_hand_over_last_hand(self) -> None:
        pair = _shop_card(
            index=0,
            name="Pair Planet",
            cost=3,
            card_set="Planet",
            summary="Level up Pair (+15 Chips, +1 Mult).",
        )
        flush = _shop_card(
            index=1,
            name="Flush Planet",
            cost=3,
            card_set="Planet",
            summary="Level up Flush (+15 Chips, +2 Mult).",
        )
        policy = BalatroPolicy()
        policy.plan.primary_hand = "Flush"
        state = {
            "phase": "pack_opening",
            "pack": {
                "cards": [pair, flush],
                "choices_remaining": 1,
            },
            "poker_hands": [
                {
                    "name": "Pair",
                    "played": 8,
                    "level": 1,
                    "chips": 10,
                    "mult": 2,
                },
                {
                    "name": "Flush",
                    "played": 1,
                    "level": 1,
                    "chips": 35,
                    "mult": 4,
                },
            ],
            "legal_actions": [
                {
                    "kind": "pick_pack_card",
                    "targets": [
                        {
                            "target_index": 0,
                            "card_indices": [],
                            "min_cards": 0,
                            "max_cards": 0,
                        },
                        {
                            "target_index": 1,
                            "card_indices": [],
                            "min_cards": 0,
                            "max_cards": 0,
                        },
                    ],
                },
                {"kind": "skip_pack"},
            ],
        }

        self.assertEqual(
            policy.act(state),
            {
                "kind": "pick_pack_card",
                "target_index": 0,
                "card_indices": [],
            },
        )

    def test_policy_uses_zero_target_planet_consumable(self) -> None:
        policy = BalatroPolicy()
        state = {
            "phase": "selecting_hand",
            "hand": [],
            "jokers": [],
            "consumables": [
                {
                    "index": 0,
                    "set": "Planet",
                    "name": "Pluto",
                    "rule": {"summary": "Level up High Card."},
                }
            ],
            "poker_hands": [],
            "resources": {"money": 10, "discards_left": 0, "hands_left": 4},
            "progress": {"rounds_cleared": 0},
            "blind": {"rule": "No special rule.", "target_chips": 300},
            "deck": {"draw_pile": 44},
            "legal_actions": [
                {
                    "kind": "use_consumable",
                    "targets": [
                        {
                            "target_index": 0,
                            "card_indices": [],
                            "min_cards": 0,
                            "max_cards": 0,
                        }
                    ],
                },
                {"kind": "next_round"},
            ],
        }

        policy.plan.primary_hand = "High Card"

        self.assertEqual(
            policy.act(state),
            {
                "kind": "use_consumable",
                "target_index": 0,
                "card_indices": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
