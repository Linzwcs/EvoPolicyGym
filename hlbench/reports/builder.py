"""Build human-facing reports from run artifacts."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.core.artifacts import write_json, write_jsonl
from hlbench.core.scenario import load_scenario
from hlbench.harness.epoch_runner import EpochResult


@dataclass(frozen=True)
class RunReport:
    run_dir: Path
    learning_curve: list[dict[str, Any]]
    metrics: dict[str, Any]
    report_dir: Path

    def to_record(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "report_dir": str(self.report_dir),
            "learning_curve_path": str(self.run_dir / "learning_curve.json"),
            "metrics_path": str(self.run_dir / "metrics.json"),
            "html_path": str(self.report_dir / "index.html"),
            "metrics": self.metrics,
        }


def build_run_report(
    *,
    run_dir: Path,
    scenario_name: str,
    model_name: str,
    run_id: str,
    epochs: list[EpochResult],
) -> RunReport:
    scenario = load_scenario(scenario_name)
    records = [epoch.to_record() for epoch in epochs]
    learning_curve = [_curve_row(index, record, run_dir=run_dir) for index, record in enumerate(records)]
    metrics = _metrics(
        run_id=run_id,
        model_name=model_name,
        scenario_name=scenario_name,
        scenario_id=scenario.scenario_id,
        records=records,
        learning_curve=learning_curve,
    )
    report_dir = run_dir / "report"
    write_jsonl(run_dir / "transitions.jsonl", records)
    write_json(run_dir / "learning_curve.json", learning_curve)
    write_json(run_dir / "metrics.json", metrics)
    write_json(report_dir / "learning_curve.json", learning_curve)
    write_json(report_dir / "metrics.json", metrics)
    _write_html(report_dir / "index.html", metrics=metrics, learning_curve=learning_curve)
    _write_transitions_html(report_dir / "transitions.html", records=records, run_dir=run_dir)
    _write_curve_svg(report_dir / "learning_curve.svg", learning_curve=learning_curve, metric_suffix="mean_score")
    return RunReport(
        run_dir=run_dir,
        learning_curve=learning_curve,
        metrics=metrics,
        report_dir=report_dir,
    )


def _curve_row(index: int, record: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    evaluation = record["evaluation"]
    comparison = record["comparison"]
    reward = comparison["reward"]
    submission = record["submission"]
    epoch_dir = Path(str(record["run_dir"]))
    row: dict[str, Any] = {
        "epoch": index,
        "epoch_dir": _relative_or_string(epoch_dir, run_dir),
        "transition": _relative_or_string(epoch_dir / "transition.json", run_dir),
        "policy_sha256": submission.get("policy_sha256"),
        "invalid_transition": bool(reward.get("invalid", False)),
        "minimum_score_applied": bool(reward.get("minimum_score_applied", False)),
        "reward": reward.get("reward"),
        "validation_score_delta": reward.get("validation_score_delta"),
        "heldout_score_delta": reward.get("heldout_score_delta"),
        "agent_returncode": submission.get("agent", {}).get("returncode"),
        "agent_duration_seconds": submission.get("agent", {}).get("duration_seconds"),
        "compile_ok": bool(submission.get("compile", {}).get("ok", False)),
        "protected_changed": bool(submission.get("protected_changed", False)),
    }
    for split in ("train", "validation", "heldout"):
        summary = evaluation[split]["summary"]
        row[f"{split}_mean_score"] = summary.get("mean_score")
        row[f"{split}_success_rate"] = summary.get("success_rate")
        row[f"{split}_mean_steps"] = summary.get("mean_steps")
        row[f"{split}_episodes"] = summary.get("episodes")
        row[f"{split}_minimum_score_episodes"] = summary.get("minimum_score_episodes")
    return row


def _metrics(
    *,
    run_id: str,
    model_name: str,
    scenario_name: str,
    scenario_id: str,
    records: list[dict[str, Any]],
    learning_curve: list[dict[str, Any]],
) -> dict[str, Any]:
    final = learning_curve[-1] if learning_curve else {}
    heldout_scores = [_float(row.get("heldout_mean_score")) for row in learning_curve]
    validation_scores = [_float(row.get("validation_mean_score")) for row in learning_curve]
    train_scores = [_float(row.get("train_mean_score")) for row in learning_curve]
    invalid_count = sum(1 for row in learning_curve if row["invalid_transition"])
    minimum_count = sum(1 for row in learning_curve if row["minimum_score_applied"])
    agent_failure_count = sum(1 for record in records if record["submission"]["agent"].get("returncode") != 0)
    compile_failure_count = sum(1 for record in records if not record["submission"]["compile"].get("ok"))
    protected_violation_count = sum(1 for record in records if record["submission"].get("protected_changed"))
    runtime_failure_count = sum(1 for row in learning_curve if _has_runtime_minimum_score(row))
    return {
        "run_id": run_id,
        "model_name": model_name,
        "env_name": scenario_name,
        "scenario_id": scenario_id,
        "epochs": len(learning_curve),
        "primary": {
            "heldout_return_auc": _normalized_auc(heldout_scores),
            "final_heldout_mean_return": final.get("heldout_mean_score"),
            "best_heldout_mean_return": max(heldout_scores) if heldout_scores else None,
            "final_validation_mean_return": final.get("validation_mean_score"),
            "final_train_mean_return": final.get("train_mean_score"),
        },
        "quality": {
            "invalid_transition_rate": invalid_count / len(learning_curve) if learning_curve else 0.0,
            "minimum_score_count": minimum_count,
            "agent_failure_count": agent_failure_count,
            "compile_failure_count": compile_failure_count,
            "runtime_failure_count": runtime_failure_count,
            "contract_violation_count": protected_violation_count,
        },
        "cost": {
            "train_episodes": sum(int(row.get("train_episodes") or 0) for row in learning_curve),
            "agent_wall_time_seconds": sum(_float(record["submission"]["agent"].get("duration_seconds")) for record in records),
        },
        "curves": {
            "train_mean_return": train_scores,
            "validation_mean_return": validation_scores,
            "heldout_mean_return": heldout_scores,
        },
    }


def _write_html(path: Path, *, metrics: dict[str, Any], learning_curve: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{row['epoch']}</td>"
        f"<td>{_fmt(row.get('train_mean_score'))}</td>"
        f"<td>{_fmt(row.get('validation_mean_score'))}</td>"
        f"<td>{_fmt(row.get('heldout_mean_score'))}</td>"
        f"<td>{_fmt(row.get('reward'))}</td>"
        f"<td>{html.escape(str(row.get('invalid_transition')))}</td>"
        f"<td><a href=\"../{html.escape(str(row['transition']))}\">transition</a></td>"
        "</tr>"
        for row in learning_curve
    )
    primary = metrics["primary"]
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HLBench Report - {html.escape(str(metrics['run_id']))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #d5d7da; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d5d7da; padding: 12px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 0.85rem; color: #5f6368; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>HLBench Report</h1>
  <p>{html.escape(str(metrics['model_name']))} / {html.escape(str(metrics['env_name']))} / {html.escape(str(metrics['run_id']))}</p>
  <section class="summary">
    <div class="metric"><strong>Final heldout return</strong>{_fmt(primary.get('final_heldout_mean_return'))}</div>
    <div class="metric"><strong>Best heldout return</strong>{_fmt(primary.get('best_heldout_mean_return'))}</div>
    <div class="metric"><strong>Heldout return AUC</strong>{_fmt(primary.get('heldout_return_auc'))}</div>
    <div class="metric"><strong>Invalid transition rate</strong>{_fmt(metrics['quality'].get('invalid_transition_rate'))}</div>
  </section>
  <h2>Learning Curve</h2>
  <img src="learning_curve.svg" alt="Learning curve">
  <h2>Epochs</h2>
  <table>
    <thead><tr><th>Epoch</th><th>Train</th><th>Validation</th><th>Heldout</th><th>Reward</th><th>Invalid</th><th>Artifact</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _write_transitions_html(path: Path, *, records: list[dict[str, Any]], run_dir: Path) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(_relative_or_string(Path(str(record['run_dir'])) / 'transition.json', run_dir))}</td>"
        f"<td>{html.escape(str(record['submission']['compile'].get('ok')))}</td>"
        f"<td>{html.escape(str(record['comparison']['reward'].get('invalid')))}</td>"
        "</tr>"
        for index, record in enumerate(records)
    )
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>HLBench Transitions</title></head>
<body>
<h1>Transitions</h1>
<table>
<thead><tr><th>Epoch</th><th>Transition</th><th>Compile OK</th><th>Invalid</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _write_curve_svg(path: Path, *, learning_curve: list[dict[str, Any]], metric_suffix: str) -> None:
    width, height = 720, 280
    padding = 36
    series = [
        ("train", "#1a73e8"),
        ("validation", "#188038"),
        ("heldout", "#d93025"),
    ]
    values = [
        _float(row.get(f"{split}_{metric_suffix}"))
        for row in learning_curve
        for split, _color in series
        if row.get(f"{split}_{metric_suffix}") is not None
    ]
    if not learning_curve or not values:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    x_span = max(1, len(learning_curve) - 1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = padding + (width - 2 * padding) * index / x_span
        y = height - padding - (height - 2 * padding) * (value - lo) / (hi - lo)
        return x, y

    lines: list[str] = []
    for split, color in series:
        coords = [
            point(index, _float(row.get(f"{split}_{metric_suffix}")))
            for index, row in enumerate(learning_curve)
            if row.get(f"{split}_{metric_suffix}") is not None
        ]
        if coords:
            path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path_data}" />')
            for x, y in coords:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" />')
    labels = "\n".join(
        f'<text x="{padding + index * 120}" y="22" fill="{color}" font-size="12">{split}</text>'
        for index, (split, color) in enumerate(series)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#9aa0a6" />
<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#9aa0a6" />
<text x="{padding}" y="{padding - 10}" fill="#5f6368" font-size="12">{_fmt(hi)}</text>
<text x="{padding}" y="{height - 8}" fill="#5f6368" font-size="12">{_fmt(lo)}</text>
{labels}
{chr(10).join(lines)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _normalized_auc(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    area = sum((values[index - 1] + values[index]) / 2 for index in range(1, len(values)))
    return area / (len(values) - 1)


def _has_runtime_minimum_score(row: dict[str, Any]) -> bool:
    return any(
        int(row.get(f"{split}_minimum_score_episodes") or 0) > 0
        for split in ("train", "validation", "heldout")
    )


def _relative_or_string(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape(str(value))
