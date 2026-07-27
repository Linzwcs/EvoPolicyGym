"""Unit tests for exact Action admission."""

from __future__ import annotations

import unittest

from policy_system.actions import ActionGateway


class ActionGatewayTest(unittest.TestCase):
    def test_card_action_requires_current_membership_and_cardinality(self) -> None:
        gate = ActionGateway(
            {
                "play_hand": {
                    "kind": "play_hand",
                    "card_indices": [0, 1, 2],
                    "min_cards": 1,
                    "max_cards": 2,
                }
            }
        )

        self.assertEqual(
            gate.cards("play_hand", [2, 0]),
            {"kind": "play_hand", "card_indices": [2, 0]},
        )
        self.assertIsNone(gate.cards("play_hand", []))
        self.assertIsNone(gate.cards("play_hand", [0, 0]))
        self.assertIsNone(gate.cards("play_hand", [3]))
        self.assertIsNone(gate.cards("play_hand", [0, 1, 2]))

    def test_nested_target_card_constraints_are_exact(self) -> None:
        gate = ActionGateway(
            {
                "use_consumable": {
                    "kind": "use_consumable",
                    "targets": [
                        {
                            "target_index": 1,
                            "card_indices": [0, 2, 4],
                            "min_cards": 2,
                            "max_cards": 2,
                        }
                    ],
                }
            }
        )

        self.assertEqual(
            gate.target_cards("use_consumable", 1, [4, 0]),
            {
                "kind": "use_consumable",
                "target_index": 1,
                "card_indices": [4, 0],
            },
        )
        self.assertIsNone(gate.target_cards("use_consumable", 1, [0]))
        self.assertIsNone(gate.target_cards("use_consumable", 1, [0, 3]))
        self.assertIsNone(gate.target_cards("use_consumable", 2, [0, 2]))


if __name__ == "__main__":
    unittest.main()
