"""Tests for visible Boss Blind constraints."""

from __future__ import annotations

import unittest

from policy_system.boss import (
    filter_candidates,
    only_one_hand,
    restricts_hand_type,
)
from policy_system.state import EpisodePlan


class BossConstraintTest(unittest.TestCase):
    def test_needle_allows_discard_planning_for_its_only_hand(self) -> None:
        self.assertTrue(
            only_one_hand({"rule": "Only one hand is available."})
        )

    def test_round_history_resets_together(self) -> None:
        plan = EpisodePlan()
        plan.prepare_round(3)
        assert plan.played_hand_types is not None
        assert plan.played_hand_counts is not None
        plan.played_hand_types.add("Pair")
        plan.played_hand_counts["Pair"] = 1

        plan.prepare_round(4)

        self.assertEqual(plan.played_hand_types, set())
        self.assertEqual(plan.played_hand_counts, {})

    def test_eye_rule_forbids_repeating_a_hand_type(self) -> None:
        blind = {
            "rule": "No poker hand type may be played more than once this round."
        }
        ranked = [(20.0, "Pair", [0, 1]), (15.0, "Flush", [0, 1, 2, 3, 4])]

        self.assertEqual(restricts_hand_type(blind), "unique")
        self.assertEqual(
            filter_candidates(
                ranked,
                blind=blind,
                played_hand_types={"Pair"},
            ),
            [ranked[1]],
        )

    def test_mouth_rule_reuses_the_first_hand_type(self) -> None:
        blind = {
            "rule": "After the first hand, only that poker hand type may be played this round."
        }
        ranked = [(20.0, "Flush", [0, 1, 2, 3, 4]), (15.0, "Pair", [0, 1])]

        self.assertEqual(restricts_hand_type(blind), "single")
        self.assertEqual(
            filter_candidates(
                ranked,
                blind=blind,
                played_hand_types={"Pair"},
            ),
            [ranked[1]],
        )


if __name__ == "__main__":
    unittest.main()
