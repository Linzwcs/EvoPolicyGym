"""Performance benchmarks for hand evaluation.

Run with: uv run pytest -m benchmark
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_eval import evaluate_poker_hand


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _card(suit: str, rank: str) -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "T",
        "Jack": "J",
        "Queen": "Q",
        "King": "K",
        "Ace": "A",
    }
    c = Card()
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability("c_base")
    return c


@pytest.mark.benchmark
class TestSingleHandPerformance:
    def test_evaluate_poker_hand_single(self):
        """evaluate_poker_hand on a single 5-card hand."""
        hand = [
            _card("Hearts", "5"),
            _card("Spades", "5"),
            _card("Clubs", "King"),
            _card("Diamonds", "King"),
            _card("Hearts", "Ace"),
        ]

        # Warm up
        for _ in range(100):
            evaluate_poker_hand(hand)

        n = 10_000
        start = time.perf_counter()
        for _ in range(n):
            evaluate_poker_hand(hand)
        elapsed = time.perf_counter() - start

        us_per_call = elapsed / n * 1_000_000
        print(f"\n  evaluate_poker_hand: {us_per_call:.1f} us/call ({n:,} calls)")
        assert us_per_call < 100, f"Too slow: {us_per_call:.1f} us (target < 100 us)"
