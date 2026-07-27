"""Tests for visible hand and Joker score modeling."""

from __future__ import annotations

import unittest
from typing import Any

from policy_system.scoring import ScoringContext, estimate_score


def _card(rank: str, suit: str = "Hearts") -> dict[str, Any]:
    if rank == "Ace":
        chips = 11
    elif rank in {"10", "Jack", "Queen", "King"}:
        chips = 10
    else:
        chips = int(rank)
    return {
        "rank": rank,
        "suit": suit,
        "chips": chips,
        "debuffed": False,
        "edition": None,
        "enhancement": "c_base",
        "ability": {},
    }


def _joker(
    summary: str,
    **parameters: Any,
) -> dict[str, Any]:
    return {
        "debuffed": False,
        "edition": None,
        "ability": parameters,
        "rule": {
            "summary": summary,
            "parameters": parameters,
        },
    }


def _pair_score(
    jokers: list[dict[str, Any]],
    *,
    context: ScoringContext | None = None,
) -> float:
    hand = [_card("10", "Diamonds"), _card("10", "Diamonds")]
    return estimate_score(
        hand=hand,
        selected_indices=[0, 1],
        scoring_local=[0, 1],
        hand_name="Pair",
        poker_hand={"chips": 10, "mult": 2},
        jokers=jokers,
        context=context or ScoringContext(),
    )


class ScoringTest(unittest.TestCase):
    def test_visible_default_chip_mod_is_counted(self) -> None:
        stuntman = _joker(
            "+Chips chips (default 250).",
            extra={"chip_mod": 250, "h_size": 2},
        )

        self.assertEqual(_pair_score([stuntman]), 560)

    def test_money_bucket_mult_uses_current_visible_money(self) -> None:
        bootstraps = _joker(
            "+2 mult per $5 held.",
            extra={"dollars": 5, "mult": 2},
        )

        self.assertEqual(
            _pair_score([bootstraps], context=ScoringContext(money=17)),
            240,
        )

    def test_lowest_held_rank_mult_uses_unplayed_card(self) -> None:
        raised_fist = _joker("+2× lowest held card's rank as mult.")
        hand = [_card("10"), _card("10"), _card("3")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[raised_fist],
            context=ScoringContext(),
        )

        self.assertEqual(score, 240)

    def test_hand_specific_bonus_does_not_leak_to_wrong_hand(self) -> None:
        zany = _joker(
            "+12 Mult if hand contains Three of a Kind.",
            t_mult=12,
            type="Three of a Kind",
        )

        self.assertEqual(_pair_score([zany]), 60)

    def test_hand_specific_xmult_does_not_leak_to_wrong_hand(self) -> None:
        tribe = _joker(
            "X Mult if hand contains Flush.",
            x_mult=2,
            type="Flush",
        )

        self.assertEqual(_pair_score([tribe]), 60)

    def test_recurring_xmult_requires_visible_ready_state(self) -> None:
        waiting = _joker(
            "X Mult every N+1 hands.",
            extra={"Xmult": 4, "every": 5, "remaining": "5 remaining"},
        )
        ready = _joker(
            "X Mult every N+1 hands.",
            extra={"Xmult": 4, "every": 5, "remaining": "0 remaining"},
        )

        self.assertEqual(_pair_score([waiting]), 60)
        self.assertEqual(_pair_score([ready]), 240)

    def test_unobservable_conditional_nested_xmult_stays_inactive(self) -> None:
        card_sharp = _joker(
            "X Mult if same hand type played twice this round.",
            extra={"Xmult": 3},
        )

        self.assertEqual(_pair_score([card_sharp]), 60)

    def test_final_hand_xmult_uses_visible_hands_left(self) -> None:
        acrobat = _joker(
            "X Mult on last hand of round.",
            extra=3,
        )

        self.assertEqual(
            _pair_score(
                [acrobat],
                context=ScoringContext(hands_left=3),
            ),
            60,
        )
        self.assertEqual(
            _pair_score(
                [acrobat],
                context=ScoringContext(hands_left=1),
            ),
            180,
        )

    def test_unmatched_per_card_effect_does_not_become_flat_bonus(self) -> None:
        even_steven = _joker(
            "+4 Mult for even cards.",
            extra=4,
        )
        odd_pair = [_card("5"), _card("5")]

        score = estimate_score(
            hand=odd_pair,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[even_steven],
            context=ScoringContext(),
        )

        self.assertEqual(score, 40)

    def test_nested_mult_respects_visible_hand_size_condition(self) -> None:
        half_joker = _joker(
            "+20 Mult if ≤3 cards played.",
            extra={"mult": 20, "size": 3},
        )
        short_score = _pair_score([half_joker])
        hand = [
            _card("10", "Hearts"),
            _card("9", "Hearts"),
            _card("8", "Hearts"),
            _card("7", "Hearts"),
            _card("6", "Hearts"),
        ]
        long_score = estimate_score(
            hand=hand,
            selected_indices=[0, 1, 2, 3, 4],
            scoring_local=[0, 1, 2, 3, 4],
            hand_name="Straight Flush",
            poker_hand={"chips": 100, "mult": 8},
            jokers=[half_joker],
            context=ScoringContext(),
        )

        self.assertEqual(short_score, 660)
        self.assertEqual(long_score, 1120)

    def test_visible_current_chips_are_not_disabled_by_decay_wording(
        self,
    ) -> None:
        ice_cream = _joker(
            "Starts +100 Chips, -5 per hand.",
            extra={"chip_mod": 5, "chips": 100},
        )

        self.assertEqual(_pair_score([ice_cream]), 260)

    def test_visible_rank_filter_applies_nested_values_per_card(self) -> None:
        scholar = _joker(
            "+20 Chips and +4 Mult for Ace.",
            extra={"chips": 20, "mult": 4},
        )
        hand = [_card("Ace"), _card("Ace")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[scholar],
            context=ScoringContext(),
        )

        self.assertEqual(score, 720)

    def test_suit_mult_counts_matching_scoring_cards(self) -> None:
        greedy = _joker(
            "+3 Mult per Diamond scored.",
            effect="Suit Mult",
            extra={"s_mult": 3, "suit": "Diamonds"},
        )

        self.assertEqual(_pair_score([greedy]), 240)

    def test_visible_suit_name_can_supply_per_card_mult(self) -> None:
        onyx = _joker(
            "+7 Mult per Club scored.",
            extra=7,
        )
        hand = [_card("10", "Clubs"), _card("10", "Clubs")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[onyx],
            context=ScoringContext(),
        )

        self.assertEqual(score, 480)

    def test_held_rank_mult_counts_only_unplayed_cards(self) -> None:
        shoot_the_moon = _joker(
            "+13 Mult per Queen held.",
            extra=13,
        )
        hand = [
            _card("10", "Diamonds"),
            _card("10", "Clubs"),
            _card("Queen"),
        ]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[shoot_the_moon],
            context=ScoringContext(),
        )

        self.assertEqual(score, 450)

    def test_all_four_suits_condition_uses_scoring_cards(self) -> None:
        flower_pot = _joker(
            "X Mult if scoring hand contains all 4 suits.",
            extra=3,
        )
        hand = [
            _card("10", "Hearts"),
            _card("10", "Diamonds"),
            _card("10", "Spades"),
            _card("10", "Clubs"),
        ]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1, 2, 3],
            scoring_local=[0, 1, 2, 3],
            hand_name="Four of a Kind",
            poker_hand={"chips": 60, "mult": 7},
            jokers=[flower_pot],
            context=ScoringContext(),
        )

        self.assertEqual(score, 2100)

    def test_all_black_held_condition_applies_visible_xmult(self) -> None:
        blackboard = _joker(
            "X3 if all held cards are Spades or Clubs.",
            extra=3,
        )
        hand = [
            _card("10", "Hearts"),
            _card("10", "Diamonds"),
            _card("King", "Spades"),
            _card("Queen", "Clubs"),
        ]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[blackboard],
            context=ScoringContext(),
        )

        self.assertEqual(score, 180)

    def test_visible_dynamic_mult_uses_current_joker_count(self) -> None:
        abstract = _joker(
            "+3 mult per joker owned.",
            extra=3,
        )
        inert = _joker("No scoring effect.")

        self.assertEqual(_pair_score([abstract, inert]), 240)

    def test_hand_history_mult_includes_the_current_play(self) -> None:
        supernova = _joker(
            "+mult = times this hand type played in the run.",
            extra=1,
        )
        hand = [_card("10"), _card("10")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2, "played": 4},
            jokers=[supernova],
            context=ScoringContext(),
        )

        self.assertEqual(score, 210)

    def test_same_round_repeat_xmult_uses_episode_history(self) -> None:
        card_sharp = _joker(
            "X Mult if same hand type played twice this round.",
            extra={"Xmult": 3},
        )

        first = _pair_score(
            [card_sharp],
            context=ScoringContext(round_hand_counts={}),
        )
        repeat = _pair_score(
            [card_sharp],
            context=ScoringContext(round_hand_counts={"Pair": 1}),
        )

        self.assertEqual(first, 60)
        self.assertEqual(repeat, 180)

    def test_sell_value_mult_uses_other_visible_jokers(self) -> None:
        swashbuckler = _joker(
            "+mult = sum of all other jokers' sell values.",
            mult=1,
        )
        swashbuckler["sell_value"] = 3
        first = _joker("No scoring effect.")
        first["sell_value"] = 4
        second = _joker("No scoring effect.")
        second["sell_value"] = 6

        self.assertEqual(
            _pair_score([swashbuckler, first, second]),
            360,
        )

    def test_face_additive_and_xmult_apply_in_joker_order(self) -> None:
        face_mult = _joker("+5 Mult for face cards.", extra=5)
        photograph = _joker(
            "X Mult on FIRST face card in scoring hand only.",
            extra=2,
        )
        hand = [_card("King"), _card("King")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[face_mult, photograph],
            context=ScoringContext(),
        )

        self.assertEqual(score, 720)

    def test_visible_retrigger_repeats_card_and_per_card_effects(self) -> None:
        hack = _joker(
            "Retrigger 2/3/4/5 cards.",
            extra=1,
        )
        greedy = _joker(
            "+3 Mult per Heart scored.",
            effect="Suit Mult",
            extra={"s_mult": 3, "suit": "Hearts"},
        )
        hand = [_card("5"), _card("5")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[hack, greedy],
            context=ScoringContext(),
        )

        self.assertEqual(score, 420)

    def test_visible_right_copy_repeats_neighbor_scoring_effect(self) -> None:
        blueprint = _joker("Copy the Joker to the right.")
        strong = _joker("+20 Mult.", t_mult=20)

        self.assertEqual(_pair_score([blueprint, strong]), 1260)

    def test_first_card_retrigger_repeats_first_face_xmult(self) -> None:
        hanging_chad = _joker(
            "+2 retriggers for FIRST scored card.",
            extra=2,
        )
        photograph = _joker(
            "X Mult on FIRST face card in scoring hand only.",
            extra=2,
        )
        hand = [_card("King"), _card("King")]

        score = estimate_score(
            hand=hand,
            selected_indices=[0, 1],
            scoring_local=[0, 1],
            hand_name="Pair",
            poker_hand={"chips": 10, "mult": 2},
            jokers=[hanging_chad, photograph],
            context=ScoringContext(),
        )

        self.assertEqual(score, 800)


if __name__ == "__main__":
    unittest.main()
