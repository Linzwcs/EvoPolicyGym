"""Run repeated harness epochs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.harness.epoch_runner import EpochResult, default_loop_root, run_epoch
from hlbench.reports import RunReport, build_run_report


@dataclass(frozen=True)
class LoopResult:
    epochs: list[EpochResult]
    run_dir: Path
    report: RunReport

    def to_record(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "epochs": [epoch.to_record() for epoch in self.epochs],
            "report": self.report.to_record(),
        }


def run_loop(
    *,
    scenario_name: str,
    epochs: int = 1,
    run_id: str | None = None,
    command: list[str] | None = None,
    agent_backend: str = "command",
    agent_preset: str = "none",
    agent_command: list[str] | None = None,
    model_name: str = "local",
    train_episodes: int | None = 2,
    timeout_seconds: int = 1800,
    base_root: Path | None = None,
) -> LoopResult:
    results: list[EpochResult] = []
    base_run_id = run_id or "loop"
    base_root = base_root or default_loop_root(
        scenario_name=scenario_name,
        model_name=model_name,
        run_id=base_run_id,
    )
    workspace_root = base_root / "workspace"
    for index in range(epochs):
        epoch_run_id = f"{base_run_id}-epoch-{index:03d}"
        prior_feedback = results[-1].evaluation if results else None
        results.append(
            run_epoch(
                scenario_name=scenario_name,
                run_id=epoch_run_id,
                command=command,
                agent_backend=agent_backend,
                agent_preset=agent_preset,
                agent_command=agent_command,
                model_name=model_name,
                train_episodes=train_episodes,
                timeout_seconds=timeout_seconds,
                workspace_root=workspace_root,
                epoch_dir=base_root / "epochs" / f"epoch_{index:03d}",
                reset_workspace=(index == 0),
                prior_feedback=prior_feedback,
            )
        )
    report = build_run_report(
        run_dir=base_root,
        scenario_name=scenario_name,
        model_name=model_name,
        run_id=base_run_id,
        epochs=results,
    )
    return LoopResult(epochs=results, run_dir=base_root, report=report)
