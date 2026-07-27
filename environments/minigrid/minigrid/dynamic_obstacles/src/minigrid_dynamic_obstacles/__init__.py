"""The public MiniGrid DynamicObstacles Benchmark distribution."""

from .baseline import baseline_program
from .benchmark import DynamicObstaclesBenchmark
from .config import DynamicObstaclesConfig

__all__ = [
    "DynamicObstaclesBenchmark",
    "DynamicObstaclesConfig",
    "baseline_program",
]

