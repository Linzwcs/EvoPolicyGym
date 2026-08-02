"""Crafter scoring with complete compressed public training evidence."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import statistics
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import cast

import numpy as np
from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue, TensorValue
from numpy.typing import NDArray

from .config import CrafterConfig
from .constants import ACHIEVEMENTS, ACTIONS
from .environment import CrafterEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-crafter/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_OBSERVATION_SHAPE = (64, 64, 3)
_OBSERVATION_BYTES = 64 * 64 * 3
_OBSERVATION_CHUNK_FRAMES = 1_024
_AGENT_SKILL_NAME = "optimize-crafter-policy"
_MOVEMENT_ACTIONS = frozenset({1, 2, 3, 4})
_OPPOSITE_MOVEMENT = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
}
_LONG_REVERSE_RUN = 8
_SHORT_ACTION_CYCLE_MIN_PERIOD = 2
_SHORT_ACTION_CYCLE_MAX_PERIOD = 8
_LONG_SHORT_ACTION_CYCLE_RUN = 16
_LONG_SAME_ACTION_RUN = 16
_SURVIVAL_THRESHOLDS = (300, 600, 900)
_SURVIVAL_BAND_WEIGHTS = (1.0, 2.0, 4.0)
_SURVIVAL_WEIGHT = 0.70
_MAINTENANCE_WEIGHT = 0.15
_PRODUCTIVITY_WEIGHT = 0.10
_INNOVATION_WEIGHT = 0.05
_MAINTENANCE_VITALS = ("health", "food", "drink")
_MAINTENANCE_WARNING_THRESHOLD = 5
_MAINTENANCE_RECOVERY_CAP = 3
_PRODUCTIVITY_EVENT_CAPS = {
    "collect_wood": 8,
    "collect_sapling": 4,
    "collect_stone": 8,
    "collect_coal": 4,
    "collect_iron": 3,
    "collect_diamond": 2,
    "defeat_zombie": 4,
    "defeat_skeleton": 2,
    "place_plant": 4,
    "place_stone": 8,
}


def _agent_skill() -> str:
    packaged = files("crafter_benchmarks").joinpath(
        "skills",
        _AGENT_SKILL_NAME,
        "SKILL.md",
    )
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    source = (
        Path(__file__).parents[2]
        / "skills"
        / _AGENT_SKILL_NAME
        / "SKILL.md"
    )
    return source.read_text(encoding="utf-8")


class CrafterBenchmark:
    """Official shifted-geometric achievement score over seeded Episodes."""

    def __init__(self, config: CrafterConfig | None = None) -> None:
        selected = CrafterConfig() if config is None else config
        if type(selected) is not CrafterConfig:
            raise TypeError("config must be CrafterConfig or None")
        self._config = selected
        self._spec = _spec(selected)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        if type(split) is not str or split not in _SPLITS:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(
            EpisodeSpec(environment_seed=_episode_seed(split, seed, index))
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return CrafterEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        achievement_sets = tuple(_scored_achievements(record) for record in records)
        success_rates: dict[str, PolicyValue] = {
            name: 100.0
            * sum(name in achievements for achievements in achievement_sets)
            / len(records)
            for name in ACHIEVEMENTS
        }
        score = _crafter_score(
            tuple(cast(float, success_rates[name]) for name in ACHIEVEMENTS)
        )
        returns = tuple(
            record.total_reward if record.policy_failure is None else 0.0
            for record in records
        )
        action_diagnostics = _action_diagnostics(records)
        artifacts, artifact_summary = _complete_feedback_artifacts(records)

        feedback = Feedback(
            score=score,
            content={
                "summary": (
                    f"Crafter score {score:.3f}% across "
                    f"{len(records)} Episodes."
                ),
                "crafter_score_percent": score,
                "achievement_success_percent": success_rates,
                "mean_return": statistics.fmean(returns),
                "mean_steps": statistics.fmean(record.steps for record in records),
                "action_diagnostics": action_diagnostics,
                "episodes": len(records),
                "terminated_episodes": sum(_terminated(record) for record in records),
                "truncated_episodes": sum(_truncated(record) for record in records),
                "policy_failures": sum(
                    record.policy_failure is not None for record in records
                ),
                "failure_achievement_credit": 0.0,
                "detailed_feedback": artifact_summary,
            },
            artifacts=artifacts,
        )
        return feedback


class CrafterLongHorizonBenchmark(CrafterBenchmark):
    """Long-horizon survival gated by productive and novel development."""

    def __init__(self, config: CrafterConfig | None = None) -> None:
        super().__init__(config)
        if self._config.max_episode_steps < _SURVIVAL_THRESHOLDS[-1]:
            raise ValueError(
                "long-horizon Crafter requires max_episode_steps of at least 900"
            )
        self._spec = _long_horizon_spec(self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        canonical = super().feedback(records)
        if type(canonical.content) is not dict:
            raise RuntimeError("canonical Crafter Feedback content is invalid")
        content = dict(canonical.content)
        profile = _long_horizon_profile(records)
        score_value = profile["long_horizon_development_score"]
        if not isinstance(score_value, float):
            raise RuntimeError("long-horizon Crafter score is invalid")
        content.update(profile)
        content["summary"] = (
            f"Long-horizon development score {score_value:.3f}% across "
            f"{len(records)} Episodes; canonical Crafter score "
            f"{canonical.score:.3f}%."
        )
        return Feedback(
            score=score_value,
            content=content,
            artifacts=canonical.artifacts,
        )


def _spec(config: CrafterConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="crafter/CrafterReward-v1/achievement-score-v1",
        description=(
            "Survive and unlock Crafter's 22 achievements from canonical "
            "64x64 RGB observations. Maximize the official shifted-geometric "
            "achievement success score."
        ),
        observation_space={
            "type": "tensor",
            "dtype": "uint8",
            "shape": [64, 64, 3],
            "color_space": "RGB",
        },
        action_space={
            "type": "discrete",
            "values": list(range(len(ACTIONS))),
            "meaning": {
                str(index): name for index, name in enumerate(ACTIONS)
            },
        },
        metadata={
            "environment": "CrafterReward-v1",
            "provider": "danijar/crafter",
            "upstream_version": "1.8.3",
            "upstream_url": "https://github.com/danijar/crafter",
            "upstream_license": "MIT",
            "achievements": list(ACHIEVEMENTS),
            "official_score_formula": (
                "exp(mean(log(1 + success_percent))) - 1"
            ),
            "privileged_information_exposed": False,
        },
        environment_parameters={
            "area": [64, 64],
            "view": [9, 9],
            "image_size": [64, 64],
            "reward": True,
            "max_episode_steps": config.max_episode_steps,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="crafter_score_percent",
        score_direction="maximize",
        agent_skill=_agent_skill(),
    )


def _long_horizon_spec(config: CrafterConfig) -> BenchmarkSpec:
    canonical = _spec(config)
    metadata = dict(canonical.metadata)
    productivity_caps: dict[str, PolicyValue] = {
        name: cap for name, cap in _PRODUCTIVITY_EVENT_CAPS.items()
    }
    metadata.update(
        {
            "objective_profile": "long-horizon-development-v2",
            "canonical_comparison_metric": "crafter_score_percent",
            "survival_threshold_steps": list(_SURVIVAL_THRESHOLDS),
            "survival_score_formula": (
                "100 * (min(L, 300) + 2 * clamp(L - 300, 0, 300) "
                "+ 4 * clamp(L - 600, 0, 300)) / 2100"
            ),
            "episode_score_formula": (
                "survival * (0.70 + 0.15 * maintenance / 100 "
                "+ 0.10 * productivity / 100 "
                "+ 0.05 * innovation / 100)"
            ),
            "score_component_weights": {
                "survival": _SURVIVAL_WEIGHT,
                "maintenance": _MAINTENANCE_WEIGHT,
                "productivity": _PRODUCTIVITY_WEIGHT,
                "innovation": _INNOVATION_WEIGHT,
            },
            "maintenance": {
                "vitals": list(_MAINTENANCE_VITALS),
                "warning_threshold": _MAINTENANCE_WARNING_THRESHOLD,
                "recovery_cap_per_vital": _MAINTENANCE_RECOVERY_CAP,
                "credit": (
                    "low-state increases only; log1p(min(recoveries, cap)) "
                    "/ log1p(cap)"
                ),
            },
            "productivity": {
                "repeats_only": True,
                "credit": "log1p(min(repeats, cap)) / log1p(cap)",
                "event_caps": productivity_caps,
            },
        }
    )
    return BenchmarkSpec(
        id="crafter/CrafterReward-v1/long-horizon-development-v2",
        description=(
            "Develop one RGB Crafter Policy that survives repeated day-night "
            "cycles while sustaining useful production and unlocking new "
            "capabilities. Survival gates every scored Episode."
        ),
        observation_space=canonical.observation_space,
        action_space=canonical.action_space,
        metadata=metadata,
        environment_parameters=dict(canonical.environment_parameters),
        max_episode_steps=canonical.max_episode_steps,
        primary_metric="long_horizon_development_score",
        score_direction="maximize",
        agent_skill=None,
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _scored_achievements(record: EpisodeRecord) -> frozenset[str]:
    if record.policy_failure is not None:
        return frozenset()
    unlocked: set[str] = set()
    for transition in record.transitions:
        unlocked.update(_transition_achievements(transition.step.metrics))
    return frozenset(unlocked)


def _transition_achievements(metrics: PolicyValue) -> tuple[str, ...]:
    achievements, _ = _transition_metrics(metrics)
    return achievements


def _transition_event_counts(metrics: PolicyValue) -> dict[str, int]:
    _, event_counts = _transition_metrics(metrics)
    return event_counts


def _transition_metrics(
    metrics: PolicyValue,
) -> tuple[tuple[str, ...], dict[str, int]]:
    required_keys = {"achievements_unlocked"}
    allowed_keys = {
        "achievements_unlocked",
        "achievement_event_counts",
        "maintenance_vitals",
    }
    if (
        type(metrics) is not dict
        or not required_keys.issubset(metrics)
        or not set(metrics).issubset(allowed_keys)
    ):
        raise ValueError("Crafter transition metrics are invalid")
    _transition_maintenance_vitals(metrics)
    value = metrics["achievements_unlocked"]
    if type(value) is not list:
        raise ValueError("Crafter transition achievements are invalid")
    if any(type(name) is not str or name not in ACHIEVEMENTS for name in value):
        raise ValueError("Crafter transition achievements are invalid")
    names = cast(list[str], value)
    if len(names) != len(set(names)):
        raise ValueError("Crafter transition achievements contain duplicates")
    if "achievement_event_counts" not in metrics:
        return tuple(names), {name: 1 for name in names}
    event_value = metrics["achievement_event_counts"]
    if type(event_value) is not dict:
        raise ValueError("Crafter transition achievement events are invalid")
    event_counts: dict[str, int] = {}
    for name, count in event_value.items():
        if (
            type(name) is not str
            or name not in ACHIEVEMENTS
            or type(count) is not int
            or count <= 0
        ):
            raise ValueError("Crafter transition achievement events are invalid")
        event_counts[name] = count
    if not set(names).issubset(event_counts):
        raise ValueError("unlocked Crafter achievements must be successful events")
    return tuple(names), event_counts


def _transition_maintenance_vitals(
    metrics: PolicyValue,
) -> dict[str, int] | None:
    if type(metrics) is not dict:
        raise ValueError("Crafter transition metrics are invalid")
    if "maintenance_vitals" not in metrics:
        return None
    value = metrics["maintenance_vitals"]
    if type(value) is not dict or set(value) != set(_MAINTENANCE_VITALS):
        raise ValueError("Crafter maintenance vitals are invalid")
    vitals: dict[str, int] = {}
    for name in _MAINTENANCE_VITALS:
        amount = value[name]
        if type(amount) is not int or not 0 <= amount <= 9:
            raise ValueError("Crafter maintenance vitals are invalid")
        vitals[name] = amount
    return vitals


def _crafter_score(success_rates: Sequence[float]) -> float:
    if len(success_rates) != len(ACHIEVEMENTS):
        raise ValueError("Crafter score requires every achievement")
    if any(not math.isfinite(rate) or not 0.0 <= rate <= 100.0 for rate in success_rates):
        raise ValueError("Crafter success rates are invalid")
    return math.expm1(statistics.fmean(math.log1p(rate) for rate in success_rates))


def _long_horizon_profile(
    records: Sequence[EpisodeRecord],
) -> dict[str, PolicyValue]:
    episode_components = tuple(
        _long_horizon_episode_components(record) for record in records
    )
    effective_steps = tuple(component[0] for component in episode_components)
    survival_scores = tuple(component[1] for component in episode_components)
    maintenance_scores = tuple(component[2] for component in episode_components)
    productivity_scores = tuple(component[3] for component in episode_components)
    innovation_scores = tuple(component[4] for component in episode_components)
    episode_scores = tuple(component[5] for component in episode_components)
    event_totals = {name: 0 for name in ACHIEVEMENTS}
    for record in records:
        if record.policy_failure is not None:
            continue
        for name, count in _episode_event_counts(record).items():
            event_totals[name] += count
    maintenance_totals = {name: 0 for name in _MAINTENANCE_VITALS}
    for record in records:
        if record.policy_failure is None:
            for name, count in _maintenance_recovery_counts(record).items():
                maintenance_totals[name] += count
    survival_at_steps: dict[str, PolicyValue] = {}
    for threshold in _SURVIVAL_THRESHOLDS:
        count = sum(steps >= threshold for steps in effective_steps)
        survival_at_steps[str(threshold)] = {
            "count": count,
            "percent": 100.0 * count / len(records),
        }
    score = statistics.fmean(episode_scores)
    public_event_totals: dict[str, PolicyValue] = {
        name: count for name, count in event_totals.items()
    }
    public_productivity_caps: dict[str, PolicyValue] = {
        name: cap for name, cap in _PRODUCTIVITY_EVENT_CAPS.items()
    }
    public_maintenance_totals: dict[str, PolicyValue] = {
        name: count for name, count in maintenance_totals.items()
    }
    return {
        "long_horizon_development_score": score,
        "score_profile": "long-horizon-development-v2",
        "score_components": {
            "survival_score_percent": statistics.fmean(survival_scores),
            "maintenance_score_percent": statistics.fmean(maintenance_scores),
            "productivity_score_percent": statistics.fmean(productivity_scores),
            "innovation_score_percent": statistics.fmean(innovation_scores),
            "weights": {
                "survival": _SURVIVAL_WEIGHT,
                "maintenance": _MAINTENANCE_WEIGHT,
                "productivity": _PRODUCTIVITY_WEIGHT,
                "innovation": _INNOVATION_WEIGHT,
            },
            "aggregation": (
                "mean(survival * (0.70 + 0.15 * maintenance / 100 "
                "+ 0.10 * productivity / 100 "
                "+ 0.05 * innovation / 100))"
            ),
        },
        "survival_at_steps": survival_at_steps,
        "survival_steps": {
            "mean": statistics.fmean(effective_steps),
            "median": statistics.median(effective_steps),
            "p10": _nearest_rank(effective_steps, 0.10),
            "p90": _nearest_rank(effective_steps, 0.90),
            "max": max(effective_steps),
        },
        "achievement_event_counts": public_event_totals,
        "maintenance_recovery_counts": public_maintenance_totals,
        "maintenance_warning_threshold": _MAINTENANCE_WARNING_THRESHOLD,
        "maintenance_recovery_cap_per_vital": _MAINTENANCE_RECOVERY_CAP,
        "productivity_event_caps": public_productivity_caps,
        "productivity_repeats_only": True,
    }


def _long_horizon_episode_components(
    record: EpisodeRecord,
) -> tuple[int, float, float, float, float, float]:
    if record.policy_failure is not None:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    effective_steps = record.steps - int(_terminated(record))
    survival = _survival_score(effective_steps)
    event_counts = _episode_event_counts(record)
    maintenance = _maintenance_score(_maintenance_recovery_counts(record))
    productivity = _productivity_score(event_counts)
    innovation = 100.0 * len(_scored_achievements(record)) / len(ACHIEVEMENTS)
    combined = survival * (
        _SURVIVAL_WEIGHT
        + _MAINTENANCE_WEIGHT * maintenance / 100.0
        + _PRODUCTIVITY_WEIGHT * productivity / 100.0
        + _INNOVATION_WEIGHT * innovation / 100.0
    )
    return (
        effective_steps,
        survival,
        maintenance,
        productivity,
        innovation,
        combined,
    )


def _survival_score(effective_steps: int) -> float:
    starts = (0, 300, 600)
    weighted_steps = sum(
        weight * max(min(effective_steps - start, 300), 0)
        for start, weight in zip(starts, _SURVIVAL_BAND_WEIGHTS, strict=True)
    )
    maximum = 300 * sum(_SURVIVAL_BAND_WEIGHTS)
    return 100.0 * weighted_steps / maximum


def _episode_event_counts(record: EpisodeRecord) -> dict[str, int]:
    counts = {name: 0 for name in ACHIEVEMENTS}
    for transition in record.transitions:
        for name, count in _transition_event_counts(
            transition.step.metrics
        ).items():
            counts[name] += count
    return counts


def _maintenance_recovery_counts(record: EpisodeRecord) -> dict[str, int]:
    counts = {name: 0 for name in _MAINTENANCE_VITALS}
    previous: dict[str, int] | None = None
    for transition in record.transitions:
        current = _transition_maintenance_vitals(transition.step.metrics)
        if current is None:
            continue
        if previous is not None:
            for name in _MAINTENANCE_VITALS:
                if (
                    previous[name] <= _MAINTENANCE_WARNING_THRESHOLD
                    and current[name] > previous[name]
                ):
                    counts[name] += 1
        previous = current
    return counts


def _maintenance_score(recovery_counts: dict[str, int]) -> float:
    credits = [
        math.log1p(min(recovery_counts[name], _MAINTENANCE_RECOVERY_CAP))
        / math.log1p(_MAINTENANCE_RECOVERY_CAP)
        for name in _MAINTENANCE_VITALS
    ]
    return 100.0 * statistics.fmean(credits)


def _productivity_score(event_counts: dict[str, int]) -> float:
    credits = []
    for name, cap in _PRODUCTIVITY_EVENT_CAPS.items():
        repeats = max(event_counts[name] - 1, 0)
        credits.append(
            math.log1p(min(repeats, cap)) / math.log1p(cap)
        )
    return 100.0 * statistics.fmean(credits)


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _action_diagnostics(records: Sequence[EpisodeRecord]) -> dict[str, PolicyValue]:
    """Summarize public Actions without changing the official Crafter score."""

    counts = [0] * len(ACTIONS)
    total_actions = 0
    movement_actions = 0
    adjacent_movement_pairs = 0
    immediate_reverse_pairs = 0
    longest_reverse_run = 0
    episodes_with_long_reverse_run = 0
    longest_short_cycle_run = 0
    longest_short_cycle_period = 0
    episodes_with_long_short_cycle_run = 0
    longest_same_action_run = 0
    episodes_with_long_same_action_run = 0

    for record in records:
        actions: list[int] = []
        for transition in record.transitions:
            action = transition.action
            if type(action) is not int or action not in range(len(ACTIONS)):
                raise ValueError("Crafter Action diagnostics contain an invalid Action")
            actions.append(action)
            counts[action] += 1
            total_actions += 1
            movement_actions += action in _MOVEMENT_ACTIONS

        episode_longest = 0
        current_reverse_run = 1
        for previous, action in zip(actions, actions[1:], strict=False):
            if previous in _MOVEMENT_ACTIONS and action in _MOVEMENT_ACTIONS:
                adjacent_movement_pairs += 1
            if (
                previous in _MOVEMENT_ACTIONS
                and _OPPOSITE_MOVEMENT[previous] == action
            ):
                immediate_reverse_pairs += 1
                current_reverse_run += 1
                episode_longest = max(episode_longest, current_reverse_run)
            else:
                current_reverse_run = 1
        longest_reverse_run = max(longest_reverse_run, episode_longest)
        episodes_with_long_reverse_run += episode_longest >= _LONG_REVERSE_RUN
        cycle_run, cycle_period = _longest_short_action_cycle(actions)
        if cycle_run > longest_short_cycle_run or (
            cycle_run == longest_short_cycle_run
            and cycle_period
            and (
                not longest_short_cycle_period
                or cycle_period < longest_short_cycle_period
            )
        ):
            longest_short_cycle_run = cycle_run
            longest_short_cycle_period = cycle_period
        episodes_with_long_short_cycle_run += (
            cycle_run >= _LONG_SHORT_ACTION_CYCLE_RUN
        )
        same_action_run = _longest_same_action_run(actions)
        longest_same_action_run = max(
            longest_same_action_run,
            same_action_run,
        )
        episodes_with_long_same_action_run += (
            same_action_run >= _LONG_SAME_ACTION_RUN
        )

    return {
        "scored": False,
        "action_counts": {
            name: counts[index] for index, name in enumerate(ACTIONS)
        },
        "total_actions": total_actions,
        "movement_actions": movement_actions,
        "movement_action_percent": (
            100.0 * movement_actions / total_actions if total_actions else 0.0
        ),
        "adjacent_movement_pairs": adjacent_movement_pairs,
        "immediate_reverse_movement_pairs": immediate_reverse_pairs,
        "immediate_reverse_movement_percent": (
            100.0 * immediate_reverse_pairs / adjacent_movement_pairs
            if adjacent_movement_pairs
            else 0.0
        ),
        "longest_immediate_reverse_action_run": longest_reverse_run,
        "episodes_with_immediate_reverse_run_at_least_8": (
            episodes_with_long_reverse_run
        ),
        "longest_repeated_short_action_cycle_run": longest_short_cycle_run,
        "longest_repeated_short_action_cycle_period": (
            longest_short_cycle_period
        ),
        "short_action_cycle_min_period": _SHORT_ACTION_CYCLE_MIN_PERIOD,
        "short_action_cycle_max_period": _SHORT_ACTION_CYCLE_MAX_PERIOD,
        "episodes_with_repeated_short_action_cycle_run_at_least_16": (
            episodes_with_long_short_cycle_run
        ),
        "longest_same_action_run": longest_same_action_run,
        "episodes_with_same_action_run_at_least_16": (
            episodes_with_long_same_action_run
        ),
    }


def _longest_short_action_cycle(actions: Sequence[int]) -> tuple[int, int]:
    best_run = 0
    best_period = 0
    maximum_period = min(_SHORT_ACTION_CYCLE_MAX_PERIOD, len(actions) // 2)
    for period in range(_SHORT_ACTION_CYCLE_MIN_PERIOD, maximum_period + 1):
        matching = 0
        for index in range(period, len(actions)):
            if actions[index] == actions[index - period]:
                matching += 1
                run = matching + period
                period_block = actions[index - period + 1 : index + 1]
                if (
                    matching >= period
                    and len(set(period_block)) >= 2
                    and (
                    run > best_run
                    or (run == best_run and period < best_period)
                    )
                ):
                    best_run = run
                    best_period = period
            else:
                matching = 0
    return best_run, best_period


def _longest_same_action_run(actions: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for action in actions:
        current = current + 1 if action == previous else 1
        longest = max(longest, current)
        previous = action
    return longest


def _terminated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
    )


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _complete_feedback_artifacts(
    records: Sequence[EpisodeRecord],
) -> tuple[tuple[Artifact, ...], dict[str, PolicyValue]]:
    artifacts: list[Artifact] = []
    trajectory_entries: list[dict[str, object]] = []
    observation_entries: list[dict[str, object]] = []
    total_observations = 0
    total_transitions = 0

    frames: list[NDArray[np.uint8]] = []
    episode_indices: list[int] = []
    observation_indices: list[int] = []
    chunk_index = 0

    def flush_observations() -> None:
        nonlocal chunk_index
        if not frames:
            return
        output = io.BytesIO()
        np.savez_compressed(
            output,
            observations=np.stack(frames),
            episode_indices=np.asarray(episode_indices, dtype=np.uint32),
            observation_indices=np.asarray(observation_indices, dtype=np.uint32),
        )
        name = f"bulk/observations-{chunk_index:06d}.npz"
        content = output.getvalue()
        artifacts.append(
            Artifact(
                name=name,
                media_type="application/x-npz",
                content=content,
                retention="bulk",
            )
        )
        observation_entries.append(
            {
                "artifact": name,
                "frames": len(frames),
                "compressed_bytes": len(content),
                "first": {
                    "episode_index": episode_indices[0],
                    "observation_index": observation_indices[0],
                },
                "last": {
                    "episode_index": episode_indices[-1],
                    "observation_index": observation_indices[-1],
                },
            }
        )
        frames.clear()
        episode_indices.clear()
        observation_indices.clear()
        chunk_index += 1

    for episode_index, record in enumerate(records):
        trajectory = _trajectory_artifact(record, episode_index=episode_index)
        artifacts.append(trajectory)
        trajectory_entries.append(
            {
                "episode_index": episode_index,
                "artifact": trajectory.name,
                "steps": record.steps,
                "compressed_bytes": trajectory.size,
            }
        )
        total_transitions += record.steps
        for observation_index, observation in enumerate(_observations(record)):
            frames.append(_observation_array(observation))
            episode_indices.append(episode_index)
            observation_indices.append(observation_index)
            total_observations += 1
            if len(frames) == _OBSERVATION_CHUNK_FRAMES:
                flush_observations()
    flush_observations()

    bulk_bytes = sum(
        artifact.size for artifact in artifacts if artifact.retention == "bulk"
    )
    manifest = {
        "schema": "crafter/complete-feedback-manifest/v1",
        "complete": True,
        "source_observation": {
            "color_space": "RGB",
            "dtype": "uint8",
            "shape": list(_OBSERVATION_SHAPE),
            "layout": "HWC",
        },
        "alignment": (
            "observation[t] -> action[t] -> observation[t + 1]"
        ),
        "episodes": len(records),
        "transitions": total_transitions,
        "observations": total_observations,
        "trajectory_artifacts": trajectory_entries,
        "observation_artifacts": observation_entries,
        "bulk_compressed_bytes": bulk_bytes,
        "retention": {
            "class": "bulk",
            "policy": (
                "complete for the newest submission; oldest bulk Artifacts "
                "are evicted first from Agent and Host records"
            ),
        },
    }
    artifacts.append(
        Artifact(
            name="artifact-manifest.json",
            media_type="application/json",
            content=(
                json.dumps(
                    manifest,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8", errors="strict"),
        )
    )
    return tuple(artifacts), {
        "schema": "crafter/complete-feedback-summary/v1",
        "complete": True,
        "episodes": len(records),
        "transitions": total_transitions,
        "observations": total_observations,
        "bulk_compressed_bytes": bulk_bytes,
        "observation_chunks": len(observation_entries),
        "trajectory_artifacts": len(trajectory_entries),
    }


def _trajectory_artifact(
    record: EpisodeRecord,
    *,
    episode_index: int,
) -> Artifact:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": record.total_reward,
                    "failure": record.policy_failure,
                    "unlocked_achievements": sorted(
                        _scored_achievements(record)
                    ),
                    "initial_observation_index": 0,
                    "final_observation_index": record.steps,
                }
            )
        )
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int or transition.action not in range(
                len(ACTIONS)
            ):
                raise ValueError("Crafter trajectory Action is invalid")
            stream.write(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation_index": step_index,
                        "next_observation_index": step_index + 1,
                        "action": transition.action,
                        "action_name": ACTIONS[transition.action],
                        "reward": transition.step.reward,
                        "achievements_unlocked": list(
                            _transition_achievements(transition.step.metrics)
                        ),
                        "achievement_event_counts": (
                            _transition_event_counts(transition.step.metrics)
                        ),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
    return Artifact(
        name=(
            f"bulk/episodes/episode-{episode_index:06d}/"
            "trajectory-000000.jsonl.gz"
        ),
        media_type="application/gzip",
        content=output.getvalue(),
        retention="bulk",
    )


def _observations(record: EpisodeRecord) -> Sequence[PolicyValue]:
    return (record.initial_observation,) + tuple(
        transition.step.observation for transition in record.transitions
    )


def _observation_array(
    value: PolicyValue,
) -> NDArray[np.uint8]:
    if (
        type(value) is not TensorValue
        or value.dtype != "uint8"
        or value.shape != _OBSERVATION_SHAPE
        or len(value.data) != _OBSERVATION_BYTES
    ):
        raise ValueError("Crafter Feedback observation is invalid")
    return np.frombuffer(value.data, dtype=np.uint8).reshape(_OBSERVATION_SHAPE)


def _json_line(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


__all__ = ["CrafterBenchmark", "CrafterLongHorizonBenchmark"]
