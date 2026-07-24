"""Performance benchmarks for the game runner.

Run with: uv run pytest -m benchmark
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.runner import greedy_play_agent, simulate_run


@pytest.mark.benchmark
class TestRunnerPerformance:
    def test_over_500_runs_per_sec(self):
        n = 100
        t0 = time.time()
        for i in range(n):
            simulate_run("b_red", 1, f"PERF_{i}", greedy_play_agent, max_actions=500)
        elapsed = time.time() - t0
        assert n / elapsed > 500, f"Only {n / elapsed:.0f} runs/sec (target: >500)"
