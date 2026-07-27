"""Tests for visible effect parsing and build-aware replacement."""

from __future__ import annotations

import unittest
from typing import Any

from policy_system.build import choose_joker_upgrade, effect_context
from policy_system.effects import (
    EffectContext,
    next_main_xmult_swap,
    profile_joker,
)
from policy_system.strategy import BalatroPolicy


def _joker(
    *,
    index: int,
    key: str,
    summary: str,
    cost: int = 5,
    sell_value: int = 2,
    **parameters: Any,
) -> dict[str, Any]:
    return {
        "index": index,
        "key": key,
        "name": key,
        "set": "Joker",
        "cost": cost,
        "sell_value": sell_value,
        "edition": None,
        "debuffed": False,
        "eternal": False,
        "ability": parameters,
        "rule": {
            "summary": summary,
            "parameters": parameters,
            "rarity": {"level": 1, "name": "Common"},
        },
    }


class EffectProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = EffectContext(
            primary_hand="Pair",
            money=20,
            joker_count=5,
            deck_remaining=30,
            ante=3,
        )

    def test_dynamic_visible_parameters_use_current_context(self) -> None:
        abstract = _joker(
            index=0,
            key="abstract",
            summary="+3 mult per joker owned.",
            extra=3,
        )
        bull = _joker(
            index=1,
            key="bull",
            summary="+2 chips per $1 held.",
            extra=2,
        )

        self.assertEqual(profile_joker(abstract, self.context).add_mult, 15)
        self.assertEqual(profile_joker(bull, self.context).chips, 40)

    def test_hand_specific_effect_is_discounted_when_build_mismatches(self) -> None:
        droll = _joker(
            index=0,
            key="droll",
            summary="+conditional Mult if hand contains Flush.",
            t_mult=10,
            type="Flush",
        )
        flush_context = EffectContext(
            primary_hand="Flush",
            money=20,
            joker_count=3,
            deck_remaining=30,
            ante=3,
        )

        self.assertEqual(profile_joker(droll, flush_context).add_mult, 10)
        self.assertEqual(profile_joker(droll, self.context).add_mult, 4)

    def test_conditional_xmult_is_not_misclassified_as_additive_mult(self) -> None:
        photograph = _joker(
            index=0,
            key="photograph",
            summary="X Mult on FIRST face card in scoring hand only.",
            extra=2,
        )

        profile = profile_joker(photograph, self.context)

        self.assertEqual(profile.add_mult, 0)
        self.assertEqual(profile.x_mult, 1.5)
        self.assertIn("xmult", profile.roles)

    def test_recurring_xmult_is_valued_by_visible_trigger_frequency(
        self,
    ) -> None:
        loyalty = _joker(
            index=0,
            key="loyalty",
            summary="X Mult every N+1 hands.",
            extra={"Xmult": 4, "every": 5},
        )

        profile = profile_joker(loyalty, self.context)

        self.assertEqual(profile.x_mult, 1.5)

    def test_copy_rule_receives_build_utility(self) -> None:
        blueprint = _joker(
            index=0,
            key="blueprint",
            summary="Copy the Joker to the right.",
        )

        profile = profile_joker(blueprint, self.context)

        self.assertGreaterEqual(profile.utility, 38)
        self.assertIn("copy", profile.roles)


class BuildUpgradeTest(unittest.TestCase):
    def test_empty_build_accepts_affordable_unmodeled_first_joker(self) -> None:
        first = _joker(
            index=0,
            key="first",
            summary="Visible but not yet modeled scaling effect.",
        )
        context = effect_context(
            primary_hand="Pair",
            money=5,
            jokers=[],
            deck_remaining=30,
            ante=1,
        )

        upgrade = choose_joker_upgrade(
            shop_cards=[first],
            owned_cards=[],
            sellable_indices=set(),
            slots=5,
            money=5,
            context=context,
        )

        self.assertIsNotNone(upgrade)
        assert upgrade is not None
        self.assertEqual(upgrade.candidate.card["key"], "first")
        self.assertIsNone(upgrade.replaced)

    def test_open_build_preserves_baseline_non_scoring_rejection(self) -> None:
        owned = _joker(
            index=0,
            key="owned",
            summary="+10 Mult.",
            t_mult=10,
        )
        economy = _joker(
            index=0,
            key="economy",
            summary="Earn $4 at end of round.",
            extra=4,
        )
        context = effect_context(
            primary_hand="Pair",
            money=10,
            jokers=[owned],
            deck_remaining=30,
            ante=2,
        )

        upgrade = choose_joker_upgrade(
            shop_cards=[economy],
            owned_cards=[owned],
            sellable_indices={0},
            slots=5,
            money=10,
            context=context,
        )

        self.assertIsNone(upgrade)

    def test_full_build_replaces_weak_sellable_joker(self) -> None:
        weak = _joker(
            index=0,
            key="weak",
            summary="No modeled scoring effect.",
        )
        strong = _joker(
            index=0,
            key="strong",
            summary="+20 Mult.",
            t_mult=20,
        )
        context = effect_context(
            primary_hand="Pair",
            money=20,
            jokers=[weak],
            deck_remaining=30,
            ante=3,
        )

        upgrade = choose_joker_upgrade(
            shop_cards=[strong],
            owned_cards=[weak],
            sellable_indices={0},
            slots=1,
            money=20,
            context=context,
        )

        self.assertIsNotNone(upgrade)
        assert upgrade is not None
        assert upgrade.replaced is not None
        self.assertEqual(upgrade.replaced.card["key"], "weak")
        self.assertEqual(upgrade.candidate.card["key"], "strong")

    def test_full_build_accepts_moderate_positive_upgrade(self) -> None:
        weak = _joker(
            index=0,
            key="weak",
            summary="+10 Mult.",
            t_mult=10,
        )
        better = _joker(
            index=1,
            key="better",
            summary="+13 Mult.",
            t_mult=13,
        )
        context = effect_context(
            primary_hand="Pair",
            money=20,
            jokers=[weak],
            deck_remaining=30,
            ante=3,
        )

        upgrade = choose_joker_upgrade(
            shop_cards=[better],
            owned_cards=[weak],
            sellable_indices={0},
            slots=1,
            money=20,
            context=context,
        )

        self.assertIsNotNone(upgrade)
        assert upgrade is not None
        self.assertEqual(upgrade.candidate.card["key"], "better")

    def test_policy_sells_then_buys_same_visible_upgrade(self) -> None:
        weak = _joker(
            index=0,
            key="weak",
            summary="No modeled scoring effect.",
        )
        strong = _joker(
            index=0,
            key="strong",
            summary="+20 Mult.",
            t_mult=20,
        )
        policy = BalatroPolicy()
        full_shop = _shop_state(
            jokers=[weak],
            shop_cards=[strong],
            slots=1,
            legal=[
                {
                    "kind": "sell_joker",
                    "target_indices": [0],
                },
                {"kind": "next_round"},
            ],
        )

        self.assertEqual(
            policy.act(full_shop),
            {"kind": "sell_joker", "target_index": 0},
        )

        open_shop = _shop_state(
            jokers=[],
            shop_cards=[strong],
            slots=1,
            legal=[
                {
                    "kind": "buy_card",
                    "target_indices": [0],
                },
                {"kind": "next_round"},
            ],
        )
        self.assertEqual(
            policy.act(open_shop),
            {"kind": "buy_card", "target_index": 0},
        )

    def test_policy_places_right_copy_before_strongest_joker(self) -> None:
        strong = _joker(
            index=0,
            key="strong",
            summary="+20 Mult.",
            t_mult=20,
        )
        weak = _joker(
            index=1,
            key="weak",
            summary="+10 Chips.",
            t_chips=10,
        )
        blueprint = _joker(
            index=2,
            key="blueprint",
            summary="Copy the Joker to the right.",
        )
        state = {
            "phase": "selecting_hand",
            "jokers": [strong, weak, blueprint],
            "resources": {"money": 20},
            "progress": {"ante": 3},
            "deck": {"draw_pile": 30},
            "legal_actions": [
                {
                    "kind": "swap_joker_left",
                    "target_indices": [1, 2],
                },
            ],
        }

        self.assertEqual(
            BalatroPolicy().act(state),
            {"kind": "swap_joker_left", "target_index": 2},
        )

    def test_policy_moves_main_xmult_after_additive_mult(self) -> None:
        trio = _joker(
            index=0,
            key="trio",
            summary="X3 Mult if hand contains Three of a Kind.",
            x_mult=3,
            type="Three of a Kind",
        )
        chips = _joker(
            index=1,
            key="chips",
            summary="+50 Chips.",
            t_chips=50,
        )
        zany = _joker(
            index=2,
            key="zany",
            summary="+12 Mult if hand contains Three of a Kind.",
            t_mult=12,
            type="Three of a Kind",
        )
        state = {
            "phase": "selecting_hand",
            "jokers": [trio, chips, zany],
            "resources": {"money": 20},
            "progress": {"ante": 3},
            "deck": {"draw_pile": 30},
            "legal_actions": [
                {
                    "kind": "swap_joker_right",
                    "target_indices": [0, 1],
                },
            ],
        }

        self.assertEqual(next_main_xmult_swap([trio, chips, zany]), 0)
        self.assertEqual(
            BalatroPolicy().act(state),
            {"kind": "swap_joker_right", "target_index": 0},
        )

    def test_main_order_leaves_card_phase_and_copy_jokers_alone(
        self,
    ) -> None:
        photograph = _joker(
            index=0,
            key="photograph",
            summary="X2 Mult on FIRST face card when scored.",
            extra=2,
        )
        zany = _joker(
            index=1,
            key="zany",
            summary="+12 Mult if hand contains Three of a Kind.",
            t_mult=12,
            type="Three of a Kind",
        )
        blueprint = _joker(
            index=2,
            key="blueprint",
            summary="Copy the Joker to the right.",
        )

        self.assertIsNone(next_main_xmult_swap([photograph, zany]))
        self.assertIsNone(next_main_xmult_swap([photograph, zany, blueprint]))

    def test_copy_target_uses_current_visible_hand_context(self) -> None:
        blueprint = _joker(
            index=0,
            key="blueprint",
            summary="Copy the Joker to the right.",
        )
        half = _joker(
            index=1,
            key="half",
            summary="+20 Mult if ≤3 cards played.",
            extra={"mult": 20, "size": 3},
        )
        blackboard = _joker(
            index=2,
            key="blackboard",
            summary="X3 if all held cards are Spades or Clubs.",
            extra=3,
        )
        state = {
            "phase": "selecting_hand",
            "hand": [
                {
                    "rank": "10",
                    "suit": "Hearts",
                    "chips": 10,
                    "debuffed": False,
                    "ability": {},
                },
                {
                    "rank": "10",
                    "suit": "Diamonds",
                    "chips": 10,
                    "debuffed": False,
                    "ability": {},
                },
                {
                    "rank": "King",
                    "suit": "Spades",
                    "chips": 10,
                    "debuffed": False,
                    "ability": {},
                },
                {
                    "rank": "Queen",
                    "suit": "Clubs",
                    "chips": 10,
                    "debuffed": False,
                    "ability": {},
                },
            ],
            "jokers": [blueprint, half, blackboard],
            "poker_hands": [
                {"name": "Pair", "chips": 10, "mult": 2},
                {"name": "High Card", "chips": 5, "mult": 1},
            ],
            "resources": {
                "money": 20,
                "discards_left": 0,
                "hands_left": 4,
            },
            "progress": {"ante": 3},
            "blind": {
                "rule": "No special rule.",
                "target_chips": 300,
            },
            "deck": {"draw_pile": 30},
            "legal_actions": [
                {
                    "kind": "swap_joker_right",
                    "target_indices": [0, 1],
                },
            ],
        }

        self.assertEqual(
            BalatroPolicy().act(state),
            {"kind": "swap_joker_right", "target_index": 0},
        )


def _shop_state(
    *,
    jokers: list[dict[str, Any]],
    shop_cards: list[dict[str, Any]],
    slots: int,
    legal: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": "shop",
        "jokers": jokers,
        "shop": {"cards": shop_cards, "boosters": [], "vouchers": []},
        "resources": {
            "money": 20,
            "joker_slots": slots,
        },
        "progress": {"ante": 3},
        "deck": {"draw_pile": 30},
        "legal_actions": legal,
    }


if __name__ == "__main__":
    unittest.main()
