from __future__ import annotations

import unittest

from policy_system.consumables import targeted_consumable


class ConsumableTest(unittest.TestCase):
    def test_targeted_visible_effect_uses_legal_cards(self) -> None:
        result = targeted_consumable(
            [{"index": 7, "set": "Tarot", "rule": {"summary": "Enhance 2 selected cards."}}],
            {"targets": [{"target_index": 7, "card_indices": [1, 2, 9], "min_cards": 2, "max_cards": 2}]},
            [{"index": 1}, {"index": 2}, {"index": 3}],
            "Pair",
        )
        self.assertEqual(result, (7, [1, 2]))

    def test_unrelated_target_is_ignored(self) -> None:
        self.assertIsNone(
            targeted_consumable(
                [{"index": 7, "set": "Spectral", "rule": {"summary": "Create a random card."}}],
                {"targets": [{"target_index": 7, "card_indices": [1], "min_cards": 1, "max_cards": 1}]},
                [{"index": 1}],
                "Pair",
            )
        )
