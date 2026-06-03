"""Protocol artifact schema builders.

These functions translate core model objects into stable JSON-shaped
dicts. They do not read or write files; filesystem adapters own that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..core import (
    Eval,
    Feed,
    OutcomeStatus,
    PoolKind,
    Report,
    Run,
    Score,
    Submit,
    Verdict,
)

SUMMARY_SCHEMA = "0.1"
RUN_SCHEMA = "0.1"

Json = Any


def summary(
    run: Run,
    submit: Submit,
    feed: Feed,
    *,
    started: str,
    completed: str,
    wall: float,
    first: int | None = None,
    lengths: Sequence[int] | None = None,
    timeouts: Sequence[int] = (),
    errors: Sequence[int] = (),
    reward: Mapping[str, float] | None = None,
    reward_episodes: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Json]:
    """Build a `feedback/submit_NNN/summary.json` object."""

    _check_feed(submit, feed)
    data = _episode_data(feed)
    charged = 0 if feed.verdict.rejected else feed.cost

    if data is None:
        returns = None
        episode_lengths = None
        mean_length = None
        local_timeouts = None
        local_errors = None
        rewards = None
        rewards_by_episode = None
    else:
        returns = data
        episode_lengths = _lengths(lengths, len(returns), feed.verdict)
        mean_length = _mean(episode_lengths)
        local_timeouts = list(timeouts)
        local_errors = list(errors)
        rewards = dict(reward) if reward is not None else None
        rewards_by_episode = _reward_episodes(reward_episodes, len(returns))

    return {
        "schema_version": SUMMARY_SCHEMA,
        "submit_index": submit.index,
        "env": run.env,
        "status": feed.verdict.value,
        "n_episodes": charged,
        "first_global_episode": first if data is not None else None,
        "env_instances": list(submit.cases),
        "remaining_budget": run.budget.left,
        "submit_started_at": started,
        "submit_completed_at": completed,
        "wall_time_seconds": wall,
        "returns": returns,
        "mean_return": feed.score.mean if data is not None else None,
        "std_return": feed.score.std if data is not None else None,
        "min_return": min(returns) if returns else None,
        "max_return": max(returns) if returns else None,
        "episode_lengths": episode_lengths,
        "mean_episode_length": mean_length,
        "timeouts": local_timeouts,
        "errors": local_errors,
        "reward_components_mean": rewards,
        "reward_components_per_episode": rewards_by_episode,
    }


def feedback(run: Run, submit: Submit, report: Report) -> dict[str, Json]:
    """Build the agent-visible summary for one persisted submit report."""

    return summary(
        run,
        submit,
        report.feed,
        started=_stamp(report.started),
        completed=_stamp(report.completed),
        wall=report.wall,
        first=report.first,
        lengths=report.feed.lengths or None,
        timeouts=_trace_indexes(report.traces, "act_timeout"),
        errors=_trace_errors(report.traces),
    )


def outcome(
    run: Run,
    final: Eval | None,
    *,
    error: Mapping[str, Json] | None = None,
    auxiliary: Mapping[str, Json] | None = None,
) -> dict[str, Json]:
    """Build the `run.json:outcome` object."""

    if run.outcome is None:
        raise ValueError("run outcome is not set")

    aux = dict(auxiliary or {})
    if run.outcome == OutcomeStatus.completed:
        if run.pick is None or run.pick.empty:
            raise ValueError("completed run requires a non-empty pick")
        if final is None or final.kind != PoolKind.final:
            raise ValueError("completed run requires a final eval")

        score = final.score
        return {
            "status": run.outcome.value,
            "error": None,
            "final_score": _require(score.value, "final score"),
            "best_submit_index": run.pick.best,
            "val_scores": _val_scores(run),
            "heldout_mean_return": _require(score.mean, "heldout mean"),
            "heldout_std_return": _require(score.std, "heldout std"),
            "heldout_returns": _returns(score, "heldout returns"),
            "auxiliary": aux,
        }

    if run.outcome == OutcomeStatus.no_ok_submit:
        return {
            "status": run.outcome.value,
            "error": None,
            "final_score": 0.0,
            "best_submit_index": None,
            "val_scores": None,
            "heldout_mean_return": None,
            "heldout_std_return": None,
            "heldout_returns": None,
            "auxiliary": aux,
        }

    if error is None:
        raise ValueError("error outcome requires error details")

    return {
        "status": run.outcome.value,
        "error": dict(error),
        "final_score": None,
        "best_submit_index": None,
        "val_scores": None,
        "heldout_mean_return": None,
        "heldout_std_return": None,
        "heldout_returns": None,
        "auxiliary": aux,
    }


def record(
    run: Run,
    result: Mapping[str, Json],
    *,
    dimensions: Mapping[str, Json],
    timing: Mapping[str, Json],
    artifacts: Mapping[str, str] | None = None,
    versions: Mapping[str, Json] | None = None,
) -> dict[str, Json]:
    """Build the top-level `run.json` object."""

    return {
        "schema_version": RUN_SCHEMA,
        "protocol_version": run.protocol,
        "model": run.model,
        "env": run.env,
        "exp_id": run.exp,
        "experiment_dimensions": dict(dimensions),
        "timing": dict(timing),
        "outcome": dict(result),
        "artifacts": _artifacts(artifacts),
        "versions": dict(versions or {}),
    }


_ARTIFACTS = {
    "workspace": "workspace/",
    "feedback": "workspace/feedback/",
    "checkpoints": "checkpoints/",
    "logs_harness": "logs/harness.log",
    "logs_agent": "logs/agent.jsonl",
    "logs_env": "logs/env.log",
}


def _check_feed(submit: Submit, feed: Feed) -> None:
    if feed.submit != submit.index:
        raise ValueError("feed submit index does not match submit")
    if feed.cost != submit.cost:
        raise ValueError("feed cost does not match submit cases")


def _artifacts(artifacts: Mapping[str, str] | None) -> dict[str, str]:
    values = dict(artifacts or _ARTIFACTS)
    for key, path in values.items():
        if path.startswith("/"):
            raise ValueError(f"artifact path {key!r} must be relative")
    return values


def _episode_data(feed: Feed) -> list[float] | None:
    if feed.verdict == Verdict.ok:
        returns = _returns(feed.score, "submit returns")
        if len(returns) != feed.cost:
            raise ValueError("ok summary returns must match submit cost")
        _require(feed.score.mean, "submit mean")
        _require(feed.score.std, "submit std")
        return returns

    if feed.verdict.partial and feed.score.returns:
        returns = list(feed.score.returns)
        if len(returns) > feed.cost:
            raise ValueError("partial returns cannot exceed submit cost")
        return returns

    return None


def _lengths(
    lengths: Sequence[int] | None,
    count: int,
    verdict: Verdict,
) -> list[int] | None:
    if lengths is None:
        if verdict == Verdict.ok:
            raise ValueError("ok summary requires episode lengths")
        return None
    values = list(lengths)
    if len(values) != count:
        raise ValueError("episode lengths must match returns")
    return values


def _reward_episodes(
    reward_episodes: Mapping[str, Sequence[float]] | None,
    count: int,
) -> dict[str, list[float]] | None:
    if reward_episodes is None:
        return None
    values = {key: list(series) for key, series in reward_episodes.items()}
    for key, series in values.items():
        if len(series) != count:
            raise ValueError(f"reward component {key!r} length must match returns")
    return values


def _val_scores(run: Run) -> dict[str, float]:
    if run.pick is None:
        raise ValueError("run pick is not set")
    return {str(index): value for index, value in sorted(run.pick.scores.items())}


def _returns(score: Score, label: str) -> list[float]:
    if not score.returns:
        raise ValueError(f"{label} are required")
    return list(score.returns)


def _require(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _mean(values: Sequence[int] | None) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _trace_indexes(traces: Sequence[Any], category: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, trace in enumerate(traces)
        if _category(getattr(trace, "error", None)) == category
    )


def _trace_errors(traces: Sequence[Any]) -> tuple[int, ...]:
    return tuple(
        index
        for index, trace in enumerate(traces)
        if (kind := _category(getattr(trace, "error", None))) is not None
        and kind != "act_timeout"
    )


def _category(error: str | None) -> str | None:
    if not error:
        return None
    return error.split(":", 1)[0]


def _stamp(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
