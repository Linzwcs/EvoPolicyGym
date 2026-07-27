"""The public MiniGrid GoToObject Benchmark distribution."""

from .baseline import baseline_program
from .benchmark import GoToObjectBenchmark
from .config import GoToObjectConfig

__all__ = [
    "GoToObjectBenchmark",
    "GoToObjectConfig",
    "baseline_program",
]

