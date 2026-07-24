"""Performance benchmarks for the env layer.

Run with: uv run pytest tests/benchmarks/test_env_bench.py -m benchmark
"""

from __future__ import annotations

import random
import time

import pytest

from jackdaw.engine.actions import Discard, PlayHand
from jackdaw.env.action_space import get_action_mask
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.observation import encode_observation


def _resolve_action(action, gs: dict):
    """Fill in card indices for marker PlayHand/Discard actions."""
    hand = gs.get("hand", [])
    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        return PlayHand(card_indices=indices)
    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        return Discard(card_indices=indices)
    return action


def _make_adapter_at_hand_phase() -> tuple[DirectAdapter, dict]:
    """Create an adapter and advance to SELECTING_HAND phase."""
    from jackdaw.engine.actions import GamePhase, SelectBlind

    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "BENCH_SEED_42")

    # Advance to selecting_hand
    for _ in range(10):
        gs = adapter.raw_state
        if gs.get("phase") == GamePhase.SELECTING_HAND:
            return adapter, gs
        legal = adapter.get_legal_actions()
        if not legal:
            break
        # Prefer SelectBlind to get to hand phase quickly
        select = [a for a in legal if isinstance(a, SelectBlind)]
        adapter.step(select[0] if select else legal[0])

    return adapter, adapter.raw_state


def _collect_game_states(n: int) -> list[dict]:
    """Collect diverse game states by playing random actions."""
    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "BENCH_COLLECT_0")
    states = []
    seed_counter = 0

    while len(states) < n:
        if adapter.done:
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")

        states.append(adapter.raw_state)

        legal = adapter.get_legal_actions()
        if not legal:
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")
            continue
        adapter.step(_resolve_action(random.choice(legal), adapter.raw_state))

    return states[:n]


@pytest.mark.benchmark
class TestEnvStepsPerSecond:
    """Assert that DirectAdapter env steps run fast enough for training."""

    def test_env_steps_per_second(self):
        """Full env step loop (engine + encode + mask) > 500 steps/sec."""
        adapter = DirectAdapter()
        adapter.reset("b_red", 1, "BENCH_SPS_0")
        n = 500
        done_count = 0

        t0 = time.perf_counter()
        for i in range(n):
            if adapter.done:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")

            gs = adapter.raw_state
            encode_observation(gs)
            get_action_mask(gs)

            legal = adapter.get_legal_actions()
            if not legal:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")
                continue
            adapter.step(_resolve_action(random.choice(legal), adapter.raw_state))

        elapsed = time.perf_counter() - t0
        sps = n / elapsed
        assert sps > 500, f"Only {sps:.0f} steps/sec (target: >500)"


@pytest.mark.benchmark
class TestEncodingLatency:
    """Assert encode_observation is fast enough."""

    def test_encoding_latency(self):
        """encode_observation < 500us mean across diverse game states."""
        states = _collect_game_states(500)

        times = []
        for gs in states:
            t0 = time.perf_counter()
            encode_observation(gs)
            times.append(time.perf_counter() - t0)

        mean_us = sum(times) / len(times) * 1e6
        assert mean_us < 500, f"encode_observation mean {mean_us:.1f}us (target: <500us)"


@pytest.mark.benchmark
class TestActionMaskLatency:
    """Assert get_action_mask is fast enough."""

    def test_action_mask_latency(self):
        """get_action_mask < 100us mean across diverse game states."""
        states = _collect_game_states(500)

        times = []
        for gs in states:
            t0 = time.perf_counter()
            get_action_mask(gs)
            times.append(time.perf_counter() - t0)

        mean_us = sum(times) / len(times) * 1e6
        assert mean_us < 100, f"get_action_mask mean {mean_us:.1f}us (target: <100us)"
