"""Generic harness orchestration."""

from hlbench.harness.epoch_runner import EpochResult, run_epoch
from hlbench.harness.loop_runner import LoopResult, run_loop

__all__ = ["EpochResult", "LoopResult", "run_epoch", "run_loop"]
