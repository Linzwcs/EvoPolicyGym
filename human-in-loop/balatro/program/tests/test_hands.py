"""Tests for public-information poker hand classification."""

from __future__ import annotations

import unittest
from typing import Any

from policy_system.hands import classify, discard_choice
from policy_system.strategy import BalatroPolicy


class HandClassificationTest(unittest.TestCase):
    def test_face_down_cards_are_not_preserved_as_a_pair_or_flush(
        self,
    ) -> None:
        hand: list[dict[str, Any]] = [
            {"rank": None, "suit": None} for _ in range(5)
        ] + [
            {"rank": "7", "suit": "Hearts"},
            {"rank": "7", "suit": "Clubs"},
            {"rank": "Ace", "suit": "Spades"},
        ]

        self.assertEqual(discard_choice(hand, [5, 6]), [0, 1, 2, 3, 4])

    def test_face_down_cards_do_not_form_an_invented_flush(self) -> None:
        hidden: list[dict[str, Any]] = [
            {"rank": None, "suit": None} for _ in range(5)
        ]

        name, scoring = classify(hidden)

        self.assertEqual(name, "High Card")
        self.assertEqual(scoring, [0])

    def test_psychic_rule_never_plays_fewer_than_five_cards(self) -> None:
        ranks = ["Ace", "King", "Queen", "Jack", "10", "9", "8", "7"]
        hand: list[dict[str, Any]] = [
            {
                "rank": rank,
                "suit": ("Hearts", "Clubs", "Spades", "Diamonds")[index % 4],
                "chips": 11 if rank == "Ace" else min(10, 14 - index),
                "debuffed": False,
                "ability": {},
            }
            for index, rank in enumerate(ranks)
        ]
        observation: dict[str, Any] = {
            "phase": "selecting_hand",
            "hand": hand,
            "jokers": [],
            "poker_hands": [],
            "resources": {
                "chips": 0,
                "discards_left": 0,
                "hands_left": 4,
                "money": 0,
            },
            "progress": {},
            "blind": {
                "rule": "Exactly five cards must be played in every hand.",
                "target_chips": 300,
            },
            "deck": {"draw_pile": 44},
            "legal_actions": [
                {
                    "kind": "play_hand",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
            ],
        }

        action = BalatroPolicy().act(observation)

        self.assertEqual(action["kind"], "play_hand")
        self.assertEqual(len(action["card_indices"]), 5)

    def test_early_discard_seeks_margin_before_spending_a_hand(self) -> None:
        ranks = ["10", "10", "Ace", "King", "Queen", "9", "7", "3"]
        hand: list[dict[str, Any]] = [
            {
                "rank": rank,
                "suit": "Clubs" if index < 2 else "Hearts",
                "chips": 11 if rank == "Ace" else 10,
                "debuffed": False,
                "ability": {},
            }
            for index, rank in enumerate(ranks)
        ]
        onyx_agate = {
            "name": "Onyx Agate",
            "debuffed": False,
            "rule": {
                "summary": "+7 Mult per Club scored.",
                "parameters": {"extra": 7},
            },
        }
        observation: dict[str, Any] = {
            "phase": "selecting_hand",
            "hand": hand,
            "jokers": [onyx_agate],
            "poker_hands": [
                {"name": "Pair", "chips": 10, "mult": 2},
                {"name": "High Card", "chips": 5, "mult": 1},
            ],
            "resources": {
                "chips": 0,
                "discards_left": 4,
                "hands_left": 4,
                "money": 0,
            },
            "progress": {},
            "blind": {
                "rule": "No special rule.",
                "target_chips": 1600,
            },
            "deck": {"draw_pile": 44},
            "legal_actions": [
                {
                    "kind": "play_hand",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
                {
                    "kind": "discard",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
            ],
        }

        action = BalatroPolicy().act(observation)

        self.assertEqual(action["kind"], "discard")

    def test_early_build_stops_chasing_after_one_failed_discard(
        self,
    ) -> None:
        ranks = ["5", "5", "4", "4", "Ace", "7", "9", "3"]
        hand: list[dict[str, Any]] = [
            {
                "rank": rank,
                "suit": ("Hearts", "Clubs", "Spades", "Diamonds")[index % 4],
                "chips": (
                    11
                    if rank == "Ace"
                    else min(10, int(rank))
                ),
                "debuffed": False,
                "ability": {},
            }
            for index, rank in enumerate(ranks)
        ]
        observation: dict[str, Any] = {
            "phase": "selecting_hand",
            "hand": hand,
            "jokers": [],
            "poker_hands": [
                {"name": "Two Pair", "chips": 20, "mult": 2},
                {"name": "Pair", "chips": 10, "mult": 2},
            ],
            "resources": {
                "chips": 0,
                "discards_left": 3,
                "hands_left": 4,
                "money": 0,
            },
            "progress": {"ante": 1, "rounds_cleared": 0},
            "blind": {"rule": "No special rule.", "target_chips": 300},
            "deck": {"draw_pile": 39},
            "legal_actions": [
                {
                    "kind": "play_hand",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
                {
                    "kind": "discard",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
            ],
        }
        policy = BalatroPolicy()
        policy.plan.prepare_round(0)
        policy.plan.consecutive_discards = 1

        self.assertEqual(policy.act(observation)["kind"], "play_hand")

    def test_nonlethal_play_cycles_safe_low_cards(self) -> None:
        cards = [
            ("10", "Hearts"),
            ("10", "Clubs"),
            ("Ace", "Diamonds"),
            ("King", "Spades"),
            ("Queen", "Diamonds"),
            ("9", "Spades"),
            ("7", "Diamonds"),
            ("3", "Spades"),
        ]
        hand: list[dict[str, Any]] = [
            {
                "rank": rank,
                "suit": suit,
                "chips": (
                    11
                    if rank == "Ace"
                    else 10
                    if rank in {"10", "Jack", "Queen", "King"}
                    else int(rank)
                ),
                "debuffed": False,
                "edition": None,
                "enhancement": "c_base",
                "seal": None,
                "ability": {},
            }
            for rank, suit in cards
        ]
        observation: dict[str, Any] = {
            "phase": "selecting_hand",
            "hand": hand,
            "jokers": [],
            "poker_hands": [
                {"name": "Pair", "chips": 10, "mult": 2},
                {"name": "High Card", "chips": 5, "mult": 1},
            ],
            "resources": {
                "chips": 0,
                "discards_left": 0,
                "hands_left": 3,
                "money": 0,
            },
            "progress": {"ante": 2, "rounds_cleared": 3},
            "blind": {"rule": "No special rule.", "target_chips": 1000},
            "deck": {"draw_pile": 44},
            "legal_actions": [
                {
                    "kind": "play_hand",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
            ],
        }

        action = BalatroPolicy().act(observation)

        self.assertEqual(action["kind"], "play_hand")
        self.assertEqual(
            set(action["card_indices"]),
            {0, 1, 5, 6, 7},
        )

        observation["blind"]["target_chips"] = 100
        observation["resources"]["discards_left"] = 1
        action_with_discard = BalatroPolicy().act(observation)

        self.assertEqual(action_with_discard["kind"], "play_hand")
        self.assertEqual(
            set(action_with_discard["card_indices"]),
            {0, 1},
        )

        observation["progress"]["ante"] = 1
        observation["resources"]["discards_left"] = 0
        observation["blind"]["target_chips"] = 1000
        early_action = BalatroPolicy().act(observation)

        self.assertEqual(early_action["kind"], "play_hand")
        self.assertEqual(
            set(early_action["card_indices"]),
            {0, 1},
        )

    def test_first_hidden_hand_is_played_without_spending_a_discard(self) -> None:
        observation: dict[str, Any] = {
            "phase": "selecting_hand",
            "hand": [
                {
                    "rank": None,
                    "suit": None,
                    "chips": None,
                    "debuffed": False,
                    "ability": {},
                }
            ],
            "jokers": [],
            "poker_hands": [],
            "resources": {
                "chips": 0,
                "discards_left": 4,
                "hands_left": 4,
                "money": 0,
            },
            "progress": {"rounds_cleared": 0},
            "blind": {
                "rule": "The first hand is drawn face-down.",
                "target_chips": 300,
            },
            "deck": {"draw_pile": 44},
            "legal_actions": [
                {
                    "kind": "play_hand",
                    "card_indices": [0],
                    "min_cards": 1,
                    "max_cards": 1,
                },
                {
                    "kind": "discard",
                    "card_indices": [0],
                    "min_cards": 1,
                    "max_cards": 1,
                },
            ],
        }

        self.assertEqual(
            BalatroPolicy().act(observation)["kind"],
            "play_hand",
        )

    def test_needle_uses_discard_before_its_only_weak_hand(self) -> None:
        hand: list[dict[str, Any]] = [
            {
                "rank": rank,
                "suit": ("Hearts", "Clubs", "Spades", "Diamonds")[index % 4],
                "chips": int(rank),
                "debuffed": False,
                "ability": {},
            }
            for index, rank in enumerate(("2", "2", "3", "5", "7", "8", "9", "10"))
        ]
        observation: dict[str, Any] = {
            "phase": "selecting_hand",
            "hand": hand,
            "jokers": [],
            "poker_hands": [
                {"name": "Pair", "chips": 10, "mult": 2},
                {"name": "High Card", "chips": 5, "mult": 1},
            ],
            "resources": {
                "chips": 0,
                "discards_left": 4,
                "hands_left": 1,
                "money": 0,
            },
            "progress": {"ante": 2, "rounds_cleared": 5},
            "blind": {
                "rule": "Only one hand is available.",
                "target_chips": 600,
            },
            "deck": {"draw_pile": 44},
            "legal_actions": [
                {
                    "kind": "play_hand",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
                {
                    "kind": "discard",
                    "card_indices": list(range(8)),
                    "min_cards": 1,
                    "max_cards": 5,
                },
            ],
        }

        self.assertEqual(BalatroPolicy().act(observation)["kind"], "discard")


if __name__ == "__main__":
    unittest.main()
