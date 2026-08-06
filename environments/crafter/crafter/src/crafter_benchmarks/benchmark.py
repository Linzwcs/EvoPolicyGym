"""Crafter scoring with complete compressed public training evidence."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import statistics
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import imageio_ffmpeg
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
from PIL import Image

from .config import CrafterConfig
from .constants import ACHIEVEMENTS, ACTIONS
from .environment import CrafterEnvironment
from .scoring import (
    FIRST_UNLOCK_REWARDS,
    PRODUCTIVITY_CREDIT_MAX,
    PROGRESS_CREDIT_MAX,
    REPEAT_EVENT_CAPS,
    REPEAT_EVENT_WEIGHTS,
    SURVIVAL_CREDIT_PER_ALIVE_STEP,
    VITAL_CREDIT_SCALE,
    repeat_event_credit,
    score_delta,
    transition_score_components,
    vital_quality,
)

_EPISODE_SEED_DOMAIN = b"evopolicygym-crafter/episode-seed/v1\0"
_EPISODE_ARTIFACT_SCENARIO_KEY = "publish_detailed_artifacts"
_SPLITS = frozenset({"train", "validation", "test"})
_OBSERVATION_SHAPE = (64, 64, 3)
_OBSERVATION_BYTES = 64 * 64 * 3
_OBSERVATION_CHUNK_FRAMES = 1_024
_REPLAY_SEGMENT_FRAMES = 2_048
_DETAILED_FEEDBACK_MAX_EPISODES = 16
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
_V3_SCORE_TOLERANCE = 1e-12


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
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
                scenario={
                    _EPISODE_ARTIFACT_SCENARIO_KEY: split == "train",
                },
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return CrafterEnvironment(
            _environment_episode(episode),
            config=self._config,
        )

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
        artifacts, artifact_summary = _complete_feedback_artifacts(
            records,
            replay_fps=self._config.replay_fps,
            replay_size=self._config.replay_size,
            detailed_artifacts=_detailed_artifacts_enabled(records),
        )

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


class CrafterSurvivalDevelopmentBenchmark(CrafterBenchmark):
    """Additive survival, maintenance, and development reward profile."""

    def __init__(self, config: CrafterConfig | None = None) -> None:
        super().__init__(config)
        self._spec = _survival_development_spec(self._config)

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return CrafterEnvironment(
            _environment_episode(episode),
            config=self._config,
            reward_profile="survival-development-v3",
        )

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        analyses = tuple(
            _survival_development_episode_analysis(
                record,
                max_episode_steps=self._config.max_episode_steps,
            )
            for record in records
        )
        content = _survival_development_profile(
            records,
            analyses,
            max_episode_steps=self._config.max_episode_steps,
        )
        score_value = content["mean_survival_development_return"]
        if not isinstance(score_value, float):
            raise RuntimeError("survival-development Crafter score is invalid")
        artifacts, artifact_summary = _complete_feedback_artifacts(
            records,
            score_profile="survival-development-v3",
            failure_return=-float(self._config.max_episode_steps),
            replay_fps=self._config.replay_fps,
            replay_size=self._config.replay_size,
            detailed_artifacts=_detailed_artifacts_enabled(records),
        )
        content["detailed_feedback"] = artifact_summary
        return Feedback(
            score=score_value,
            content=content,
            artifacts=artifacts,
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
            "public_replays": {
                "format": "H.264 MP4 without audio or overlays",
                "frames_per_second": config.replay_fps,
                "frame_size": [config.replay_size, config.replay_size],
                "source_alignment": "video frame i is observation i",
                "complete_artifact_episode_limit": (
                    _DETAILED_FEEDBACK_MAX_EPISODES
                ),
                "detailed_artifact_splits": ["train"],
            },
            "privileged_information_exposed": False,
        },
        environment_parameters={
            "area": [64, 64],
            "view": [9, 9],
            "image_size": [64, 64],
            "reward": True,
            "max_episode_steps": config.max_episode_steps,
            "replay_fps": config.replay_fps,
            "replay_size": config.replay_size,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="crafter_score_percent",
        score_direction="maximize",
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
    )


def _survival_development_spec(config: CrafterConfig) -> BenchmarkSpec:
    canonical = _spec(config)
    metadata = dict(canonical.metadata)
    first_rewards: dict[str, PolicyValue] = {
        name: reward for name, reward in FIRST_UNLOCK_REWARDS.items()
    }
    repeat_weights: dict[str, PolicyValue] = {
        name: weight for name, weight in REPEAT_EVENT_WEIGHTS.items()
    }
    repeat_caps: dict[str, PolicyValue] = {
        name: cap for name, cap in REPEAT_EVENT_CAPS.items()
    }
    metadata.update(
        {
            "objective_profile": "mean-survival-development-return-v3",
            "canonical_comparison_metric": "crafter_score_percent",
            "step_reward_formula": (
                "alive + 0.1 * alive * min(health, food, drink) / 9 "
                "+ first_unlock_delta + repeated_productivity_delta"
            ),
            "episode_score_formula": (
                "sum(step_reward), except Policy failure returns "
                "-max_episode_steps"
            ),
            "survival_credit_per_alive_step": SURVIVAL_CREDIT_PER_ALIVE_STEP,
            "vital_credit_scale": VITAL_CREDIT_SCALE,
            "vital_quality_formula": "min(health, food, drink) / 9",
            "energy_scored": False,
            "first_unlock_absolute_rewards": first_rewards,
            "progress_credit_max": PROGRESS_CREDIT_MAX,
            "productivity": {
                "maximum_credit_per_episode": PRODUCTIVITY_CREDIT_MAX,
                "repeats_only": True,
                "credit": (
                    "25 * weight / 40 * log1p(min(repeats, cap)) "
                    "/ log1p(cap)"
                ),
                "event_weights": repeat_weights,
                "event_caps": repeat_caps,
            },
            "policy_failure_return": -float(config.max_episode_steps),
            "trajectory_schema": "crafter/complete-feedback-manifest/v3",
            "upstream_reward_field": "upstream_reward",
        }
    )
    environment_parameters = dict(canonical.environment_parameters)
    environment_parameters["reward_profile"] = "survival-development-v3"
    return BenchmarkSpec(
        id=(
            "crafter/CrafterReward-v1/"
            "mean-survival-development-return-v3"
        ),
        description=(
            "Survive across the full RGB Crafter horizon while maintaining "
            "health, food, and drink, unlocking exponentially weighted later "
            "capabilities, and repeatedly completing bounded productive events."
        ),
        observation_space=canonical.observation_space,
        action_space=canonical.action_space,
        metadata=metadata,
        environment_parameters=environment_parameters,
        max_episode_steps=canonical.max_episode_steps,
        primary_metric="mean_survival_development_return",
        score_direction="maximize",
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _environment_episode(episode: EpisodeSpec) -> EpisodeSpec:
    _episode_detailed_artifacts(episode)
    return EpisodeSpec(environment_seed=episode.environment_seed)


def _detailed_artifacts_enabled(records: Sequence[EpisodeRecord]) -> bool:
    enabled = {
        _episode_detailed_artifacts(record.episode) for record in records
    }
    if len(enabled) != 1:
        raise ValueError("Crafter Feedback cannot mix Artifact modes")
    return enabled.pop()


def _episode_detailed_artifacts(episode: EpisodeSpec) -> bool:
    scenario = episode.scenario
    if scenario is None:
        return True
    if (
        type(scenario) is not dict
        or set(scenario) != {_EPISODE_ARTIFACT_SCENARIO_KEY}
    ):
        raise ValueError("Crafter Episode scenario is invalid")
    enabled = scenario[_EPISODE_ARTIFACT_SCENARIO_KEY]
    if type(enabled) is not bool:
        raise ValueError("Crafter Episode scenario is invalid")
    return enabled


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
        "energy",
        "maintenance_vitals",
        "score_delta_components",
        "upstream_reward",
    }
    if (
        type(metrics) is not dict
        or not required_keys.issubset(metrics)
        or not set(metrics).issubset(allowed_keys)
    ):
        raise ValueError("Crafter transition metrics are invalid")
    _transition_maintenance_vitals(metrics)
    _transition_energy(metrics)
    _transition_upstream_reward(metrics)
    _transition_score_delta_components(metrics)
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


def _transition_energy(metrics: PolicyValue) -> int | None:
    if type(metrics) is not dict:
        raise ValueError("Crafter transition metrics are invalid")
    if "energy" not in metrics:
        return None
    amount = metrics["energy"]
    if type(amount) is not int or not 0 <= amount <= 9:
        raise ValueError("Crafter energy is invalid")
    return amount


def _transition_upstream_reward(metrics: PolicyValue) -> float | None:
    if type(metrics) is not dict:
        raise ValueError("Crafter transition metrics are invalid")
    if "upstream_reward" not in metrics:
        return None
    value = metrics["upstream_reward"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("Crafter upstream reward is invalid")
    return float(value)


def _transition_score_delta_components(
    metrics: PolicyValue,
) -> dict[str, float] | None:
    if type(metrics) is not dict:
        raise ValueError("Crafter transition metrics are invalid")
    if "score_delta_components" not in metrics:
        return None
    value = metrics["score_delta_components"]
    expected = {"survival", "vital", "progress", "productivity"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("Crafter score delta components are invalid")
    components: dict[str, float] = {}
    for name in expected:
        amount = value[name]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or float(amount) < 0.0
        ):
            raise ValueError("Crafter score delta components are invalid")
        components[name] = float(amount)
    return components


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


@dataclass(frozen=True, slots=True)
class _SurvivalDevelopmentEpisodeAnalysis:
    status: str
    terminated: bool
    truncated: bool
    failure: str | None
    steps: int
    effective_survival_steps: int
    scored_return: float
    partial_return: float
    survival_credit: float
    vital_credit: float
    progress_credit: float
    productivity_credit: float
    failure_adjustment: float
    upstream_return: float
    unlocked: frozenset[str]
    event_totals: dict[str, int]
    progress_by_achievement: dict[str, float]
    productivity_by_event: dict[str, float]
    alive_vital_steps: int
    vital_quality_sum: float
    minimum_vital_sum: int
    zero_min_vital_steps: int
    low_vital_steps: dict[int, dict[str, int]]
    terminal_vitals: dict[str, int] | None


def _survival_development_episode_analysis(
    record: EpisodeRecord,
    *,
    max_episode_steps: int,
) -> _SurvivalDevelopmentEpisodeAnalysis:
    event_totals = {name: 0 for name in ACHIEVEMENTS}
    unlocked_seen: set[str] = set()
    progress_by_achievement = {name: 0.0 for name in ACHIEVEMENTS}
    productivity_by_event = {
        name: 0.0 for name in REPEAT_EVENT_WEIGHTS
    }
    component_values: dict[str, list[float]] = {
        "survival": [],
        "vital": [],
        "progress": [],
        "productivity": [],
    }
    upstream_rewards: list[float] = []
    transition_rewards: list[float] = []
    alive_vital_steps = 0
    vital_quality_sum = 0.0
    minimum_vital_sum = 0
    zero_min_vital_steps = 0
    low_vital_steps = {
        threshold: {
            "health": 0,
            "food": 0,
            "drink": 0,
            "energy": 0,
        }
        for threshold in (2, 5)
    }
    terminal_vitals: dict[str, int] | None = None

    for transition in record.transitions:
        metrics = transition.step.metrics
        unlocked, event_counts = _transition_metrics(metrics)
        vitals = _transition_maintenance_vitals(metrics)
        energy = _transition_energy(metrics)
        upstream_reward = _transition_upstream_reward(metrics)
        reported_components = _transition_score_delta_components(metrics)
        if (
            vitals is None
            or energy is None
            or upstream_reward is None
            or reported_components is None
        ):
            raise ValueError(
                "survival-development transitions require v3 scoring metrics"
            )
        if unlocked_seen.intersection(unlocked):
            raise ValueError("Crafter v3 achievement unlocked more than once")
        unlocked_seen.update(unlocked)

        before_totals = event_totals
        expected_components, event_totals = transition_score_components(
            terminated=transition.step.terminated,
            unlocked=unlocked,
            event_counts=event_counts,
            event_totals=before_totals,
            vitals=vitals,
        )
        for name, expected in expected_components.items():
            if not math.isclose(
                reported_components[name],
                expected,
                rel_tol=0.0,
                abs_tol=_V3_SCORE_TOLERANCE,
            ):
                raise ValueError("Crafter v3 reported score component drifted")
            component_values[name].append(expected)
        expected_reward = score_delta(expected_components)
        if not math.isclose(
            transition.step.reward,
            expected_reward,
            rel_tol=0.0,
            abs_tol=_V3_SCORE_TOLERANCE,
        ):
            raise ValueError("Crafter v3 Step.reward does not match its components")
        transition_rewards.append(transition.step.reward)
        upstream_rewards.append(upstream_reward)

        for name in unlocked:
            progress_by_achievement[name] += FIRST_UNLOCK_REWARDS[name]
        for name in REPEAT_EVENT_WEIGHTS:
            productivity_by_event[name] += repeat_event_credit(
                name,
                event_totals[name],
            ) - repeat_event_credit(name, before_totals[name])

        if not transition.step.terminated:
            alive_vital_steps += 1
            quality = vital_quality(vitals)
            minimum = min(vitals.values())
            vital_quality_sum += quality
            minimum_vital_sum += minimum
            zero_min_vital_steps += minimum == 0
            all_vitals = {**vitals, "energy": energy}
            for threshold in low_vital_steps:
                for name, amount in all_vitals.items():
                    low_vital_steps[threshold][name] += amount <= threshold
        if transition.step.terminated:
            terminal_vitals = {**vitals, "energy": energy}

    partial_return = sum(transition_rewards)
    if record.policy_failure is not None:
        zeros = {name: 0 for name in ACHIEVEMENTS}
        return _SurvivalDevelopmentEpisodeAnalysis(
            status="policy_failed",
            terminated=False,
            truncated=False,
            failure=record.policy_failure,
            steps=record.steps,
            effective_survival_steps=0,
            scored_return=-float(max_episode_steps),
            partial_return=partial_return,
            survival_credit=0.0,
            vital_credit=0.0,
            progress_credit=0.0,
            productivity_credit=0.0,
            failure_adjustment=-float(max_episode_steps),
            upstream_return=0.0,
            unlocked=frozenset(),
            event_totals=zeros,
            progress_by_achievement={name: 0.0 for name in ACHIEVEMENTS},
            productivity_by_event={
                name: 0.0 for name in REPEAT_EVENT_WEIGHTS
            },
            alive_vital_steps=0,
            vital_quality_sum=0.0,
            minimum_vital_sum=0,
            zero_min_vital_steps=0,
            low_vital_steps={
                threshold: {name: 0 for name in counts}
                for threshold, counts in low_vital_steps.items()
            },
            terminal_vitals=None,
        )

    survival_credit = math.fsum(component_values["survival"])
    vital_credit = math.fsum(component_values["vital"])
    progress_credit = math.fsum(component_values["progress"])
    productivity_credit = math.fsum(component_values["productivity"])
    scored_return = sum(transition_rewards)
    reconstructed = math.fsum(
        (survival_credit, vital_credit, progress_credit, productivity_credit)
    )
    if not math.isclose(
        scored_return,
        reconstructed,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Crafter v3 Episode return does not reconstruct")
    return _SurvivalDevelopmentEpisodeAnalysis(
        status="completed",
        terminated=_terminated(record),
        truncated=_truncated(record),
        failure=None,
        steps=record.steps,
        effective_survival_steps=record.steps - int(_terminated(record)),
        scored_return=scored_return,
        partial_return=partial_return,
        survival_credit=survival_credit,
        vital_credit=vital_credit,
        progress_credit=progress_credit,
        productivity_credit=productivity_credit,
        failure_adjustment=0.0,
        upstream_return=sum(upstream_rewards),
        unlocked=frozenset(unlocked_seen),
        event_totals=event_totals,
        progress_by_achievement=progress_by_achievement,
        productivity_by_event=productivity_by_event,
        alive_vital_steps=alive_vital_steps,
        vital_quality_sum=vital_quality_sum,
        minimum_vital_sum=minimum_vital_sum,
        zero_min_vital_steps=zero_min_vital_steps,
        low_vital_steps=low_vital_steps,
        terminal_vitals=terminal_vitals,
    )


def _survival_development_profile(
    records: Sequence[EpisodeRecord],
    analyses: Sequence[_SurvivalDevelopmentEpisodeAnalysis],
    *,
    max_episode_steps: int,
) -> dict[str, PolicyValue]:
    if len(records) != len(analyses) or not analyses:
        raise ValueError("Crafter v3 analyses do not align with Episodes")
    episodes = len(analyses)
    returns = tuple(item.scored_return for item in analyses)
    effective_steps = tuple(item.effective_survival_steps for item in analyses)
    score = statistics.fmean(returns)
    mean_survival = statistics.fmean(item.survival_credit for item in analyses)
    mean_vital = statistics.fmean(item.vital_credit for item in analyses)
    mean_progress = statistics.fmean(item.progress_credit for item in analyses)
    mean_productivity = statistics.fmean(
        item.productivity_credit for item in analyses
    )
    mean_failure = statistics.fmean(
        item.failure_adjustment for item in analyses
    )
    reconstructed = math.fsum(
        (
            mean_survival,
            mean_vital,
            mean_progress,
            mean_productivity,
            mean_failure,
        )
    )
    reconstruction_error = score - reconstructed

    success_rates: dict[str, PolicyValue] = {
        name: 100.0 * sum(name in item.unlocked for item in analyses) / episodes
        for name in ACHIEVEMENTS
    }
    canonical_score = _crafter_score(
        tuple(cast(float, success_rates[name]) for name in ACHIEVEMENTS)
    )
    event_totals = {
        name: sum(item.event_totals[name] for item in analyses)
        for name in ACHIEVEMENTS
    }
    achievement_mean_credit: dict[str, PolicyValue] = {
        name: statistics.fmean(
            item.progress_by_achievement[name] for item in analyses
        )
        for name in ACHIEVEMENTS
    }
    event_mean_credit: dict[str, PolicyValue] = {
        name: statistics.fmean(
            item.productivity_by_event[name] for item in analyses
        )
        for name in REPEAT_EVENT_WEIGHTS
    }
    event_saturation_percent: dict[str, PolicyValue] = {
        name: 100.0
        * sum(
            max(item.event_totals[name] - 1, 0) >= REPEAT_EVENT_CAPS[name]
            for item in analyses
        )
        / episodes
        for name in REPEAT_EVENT_WEIGHTS
    }
    survival_at_steps: dict[str, PolicyValue] = {}
    for threshold in _SURVIVAL_THRESHOLDS:
        count = sum(value >= threshold for value in effective_steps)
        survival_at_steps[str(threshold)] = {
            "count": count,
            "percent": 100.0 * count / episodes,
        }

    alive_vital_steps = sum(item.alive_vital_steps for item in analyses)
    vital_quality_sum = math.fsum(item.vital_quality_sum for item in analyses)
    minimum_vital_sum = sum(item.minimum_vital_sum for item in analyses)
    zero_min_vital_steps = sum(
        item.zero_min_vital_steps for item in analyses
    )
    low_vital_steps: dict[str, PolicyValue] = {}
    for threshold in (2, 5):
        low_vital_steps[str(threshold)] = {
            name: sum(item.low_vital_steps[threshold][name] for item in analyses)
            for name in ("health", "food", "drink", "energy")
        }

    deaths = tuple(item for item in analyses if item.terminated)
    terminal_vital_means: dict[str, PolicyValue] = {
        name: (
            statistics.fmean(
                cast(dict[str, int], item.terminal_vitals)[name]
                for item in deaths
            )
            if deaths
            else None
        )
        for name in ("health", "food", "drink", "energy")
    }
    episode_summaries: list[PolicyValue] = [
        {
            "episode_index": index,
            "status": item.status,
            "terminated": item.terminated,
            "truncated": item.truncated,
            "failure": item.failure,
            "steps": item.steps,
            "effective_survival_steps": item.effective_survival_steps,
            "return": item.scored_return,
            "partial_return": item.partial_return,
            "components": {
                "survival_credit": item.survival_credit,
                "vital_credit": item.vital_credit,
                "progress_credit": item.progress_credit,
                "productivity_credit": item.productivity_credit,
                "failure_adjustment": item.failure_adjustment,
            },
        }
        for index, item in enumerate(analyses)
    ]
    first_rewards: dict[str, PolicyValue] = {
        name: value for name, value in FIRST_UNLOCK_REWARDS.items()
    }
    repeat_weights: dict[str, PolicyValue] = {
        name: value for name, value in REPEAT_EVENT_WEIGHTS.items()
    }
    repeat_caps: dict[str, PolicyValue] = {
        name: value for name, value in REPEAT_EVENT_CAPS.items()
    }
    return {
        "schema": "crafter/mean-survival-development-feedback/v3",
        "score_profile": "mean-survival-development-return-v3",
        "summary": (
            f"Mean survival-development return {score:.3f} across "
            f"{episodes} Episodes; canonical Crafter score "
            f"{canonical_score:.3f}%."
        ),
        "mean_survival_development_return": score,
        "scoring_parameters": {
            "survival_credit_per_alive_step": SURVIVAL_CREDIT_PER_ALIVE_STEP,
            "vital_credit_scale": VITAL_CREDIT_SCALE,
            "vital_quality_formula": "min(health, food, drink) / 9",
            "progress_credit_max": PROGRESS_CREDIT_MAX,
            "productivity_credit_max": PRODUCTIVITY_CREDIT_MAX,
        },
        "score_components": {
            "mean_survival_credit": mean_survival,
            "mean_vital_credit": mean_vital,
            "mean_progress_credit": mean_progress,
            "mean_productivity_credit": mean_productivity,
            "mean_failure_adjustment": mean_failure,
            "reconstructed_mean_return": reconstructed,
            "reconstruction_error": reconstruction_error,
            "formula": (
                "mean(survival + vital + progress + productivity + failure)"
            ),
        },
        "episode_returns": {
            "mean": score,
            "median": statistics.median(returns),
            "p10": _nearest_rank_number(returns, 0.10),
            "p90": _nearest_rank_number(returns, 0.90),
            "min": min(returns),
            "max": max(returns),
        },
        "episodes": episodes,
        "terminated_episodes": sum(item.terminated for item in analyses),
        "truncated_episodes": sum(item.truncated for item in analyses),
        "policy_failures": sum(item.failure is not None for item in analyses),
        "failure_return": -float(max_episode_steps),
        "survival_steps": {
            "mean": statistics.fmean(effective_steps),
            "median": statistics.median(effective_steps),
            "p10": _nearest_rank(effective_steps, 0.10),
            "p90": _nearest_rank(effective_steps, 0.90),
            "min": min(effective_steps),
            "max": max(effective_steps),
        },
        "survival_at_steps": survival_at_steps,
        "vital_quality": {
            "mean": (
                vital_quality_sum / alive_vital_steps
                if alive_vital_steps
                else 0.0
            ),
            "mean_min_vital": (
                minimum_vital_sum / alive_vital_steps
                if alive_vital_steps
                else 0.0
            ),
            "zero_min_vital_steps": zero_min_vital_steps,
            "zero_min_vital_step_fraction": (
                zero_min_vital_steps / alive_vital_steps
                if alive_vital_steps
                else 0.0
            ),
            "steps_at_or_below": low_vital_steps,
            "energy_scored": False,
        },
        "terminal_profile": {
            "natural_deaths": len(deaths),
            "deaths_with_food_zero": sum(
                cast(dict[str, int], item.terminal_vitals)["food"] == 0
                for item in deaths
            ),
            "deaths_with_drink_zero": sum(
                cast(dict[str, int], item.terminal_vitals)["drink"] == 0
                for item in deaths
            ),
            "deaths_with_food_and_drink_positive": sum(
                cast(dict[str, int], item.terminal_vitals)["food"] > 0
                and cast(dict[str, int], item.terminal_vitals)["drink"] > 0
                for item in deaths
            ),
            "terminal_vital_means": terminal_vital_means,
        },
        "health_change_diagnostics": _health_change_diagnostics(records),
        "progress": {
            "maximum_credit_per_episode": PROGRESS_CREDIT_MAX,
            "absolute_reward_sum": int(PROGRESS_CREDIT_MAX),
            "achievement_absolute_rewards": first_rewards,
            "achievement_success_percent": success_rates,
            "achievement_mean_credit": achievement_mean_credit,
        },
        "productivity": {
            "maximum_credit_per_episode": PRODUCTIVITY_CREDIT_MAX,
            "weight_sum": int(sum(REPEAT_EVENT_WEIGHTS.values())),
            "repeats_only": True,
            "event_weights": repeat_weights,
            "event_caps": repeat_caps,
            "event_counts": {
                name: event_totals[name] for name in REPEAT_EVENT_WEIGHTS
            },
            "event_mean_credit": event_mean_credit,
            "event_saturation_percent": event_saturation_percent,
        },
        "canonical_comparison": {
            "crafter_score_percent": canonical_score,
            "mean_upstream_return": statistics.fmean(
                item.upstream_return for item in analyses
            ),
        },
        "world_development_diagnostics": {
            "cultivation": {
                "plants_placed": event_totals["place_plant"],
                "ripe_plant_harvests": event_totals["eat_plant"],
            },
            "facilities": {
                "tables_placed": event_totals["place_table"],
                "furnaces_placed": event_totals["place_furnace"],
                "dependent_tools_made": sum(
                    event_totals[name]
                    for name in ACHIEVEMENTS
                    if name.startswith("make_")
                ),
            },
            "construction": {
                "stone_blocks_placed": event_totals["place_stone"],
                "enclosure_geometry_verified": False,
            },
            "animal_breeding_supported": False,
            "scored_beyond_published_event_components": False,
        },
        "episode_score_summaries": episode_summaries,
        "action_diagnostics": _action_diagnostics(records),
        "detailed_feedback": {},
    }


def _health_change_diagnostics(
    records: Sequence[EpisodeRecord],
) -> dict[str, PolicyValue]:
    loss_events = 0
    health_lost = 0
    recovery_events = 0
    health_recovered = 0
    by_action = {
        name: {"events": 0, "amount": 0, "terminal_events": 0}
        for name in ACTIONS
    }
    for record in records:
        previous_health = 9
        for transition in record.transitions:
            if type(transition.action) is not int or transition.action not in range(
                len(ACTIONS)
            ):
                raise ValueError("Crafter health diagnostics contain invalid Action")
            vitals = _transition_maintenance_vitals(transition.step.metrics)
            if vitals is None:
                raise ValueError("Crafter v3 health diagnostics require vitals")
            health = vitals["health"]
            delta = health - previous_health
            if delta < 0:
                amount = -delta
                loss_events += 1
                health_lost += amount
                action_entry = by_action[ACTIONS[transition.action]]
                action_entry["events"] += 1
                action_entry["amount"] += amount
                action_entry["terminal_events"] += transition.step.terminated
            elif delta > 0:
                recovery_events += 1
                health_recovered += delta
            previous_health = health
    public_by_action: dict[str, PolicyValue] = {
        name: {key: value for key, value in values.items()}
        for name, values in by_action.items()
    }
    return {
        "loss_events": loss_events,
        "health_lost": health_lost,
        "recovery_events": recovery_events,
        "health_recovered": health_recovered,
        "loss_by_action": public_by_action,
        "scored": False,
    }


def _nearest_rank_number(
    values: Sequence[float],
    percentile: float,
) -> float:
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


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
    *,
    score_profile: str = "upstream",
    failure_return: float | None = None,
    replay_fps: int = 10,
    replay_size: int = 256,
    detailed_artifacts: bool = True,
) -> tuple[tuple[Artifact, ...], dict[str, PolicyValue]]:
    if score_profile not in {"upstream", "survival-development-v3"}:
        raise ValueError("Crafter Artifact score profile is invalid")
    if score_profile == "survival-development-v3" and failure_return is None:
        raise ValueError("Crafter v3 Artifacts require a failure return")
    if type(detailed_artifacts) is not bool:
        raise TypeError("detailed_artifacts must be bool")
    if not detailed_artifacts:
        return (), _aggregate_only_artifact_summary(
            records,
            score_profile=score_profile,
            reason="split_disables_detailed_artifacts",
        )
    if len(records) > _DETAILED_FEEDBACK_MAX_EPISODES:
        return (), _aggregate_only_artifact_summary(
            records,
            score_profile=score_profile,
            reason="episode_count_exceeds_detailed_artifact_limit",
        )
    artifacts: list[Artifact] = []
    trajectory_entries: list[dict[str, object]] = []
    observation_entries: list[dict[str, object]] = []
    replay_entries: list[dict[str, object]] = []
    replay_cache: dict[tuple[int, int, tuple[bytes, ...]], bytes] = {}
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
        trajectory = _trajectory_artifact(
            record,
            episode_index=episode_index,
            score_profile=score_profile,
            failure_return=failure_return,
        )
        artifacts.append(trajectory)
        trajectory_entries.append(
            {
                "episode_index": episode_index,
                "artifact": trajectory.name,
                "steps": record.steps,
                "compressed_bytes": trajectory.size,
            }
        )
        episode_replays, episode_replay_entries = _replay_artifacts(
            record,
            episode_index=episode_index,
            frames_per_second=replay_fps,
            frame_size=replay_size,
            cache=replay_cache,
        )
        artifacts.extend(episode_replays)
        replay_entries.extend(episode_replay_entries)
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
    replay_bytes = sum(
        artifact.size
        for artifact in artifacts
        if artifact.media_type == "video/mp4"
    )
    trajectory_bytes = sum(
        artifact.size
        for artifact in artifacts
        if artifact.media_type == "application/gzip"
    )
    manifest = {
        "schema": (
            "crafter/complete-feedback-manifest/v3"
            if score_profile == "survival-development-v3"
            else "crafter/complete-feedback-manifest/v2"
        ),
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
        "replay_artifacts": replay_entries,
        "bulk_compressed_bytes": bulk_bytes,
        "trajectory_compressed_bytes": trajectory_bytes,
        "replay_compressed_bytes": replay_bytes,
        "retention": {
            "observations": (
                "bulk capacity; newest submission is protected"
            ),
            "trajectories": "permanent",
            "replays": (
                "bulk capacity; newest submission is protected"
            ),
        },
    }
    if score_profile == "survival-development-v3":
        manifest["reward_semantics"] = {
            "reward": "survival-development-v3 shaped score delta",
            "upstream_reward": "pinned Crafter 1.8.3 reward",
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
        "schema": (
            "crafter/complete-feedback-summary/v2"
            if score_profile == "survival-development-v3"
            else "crafter/complete-feedback-summary/v1"
        ),
        "complete": True,
        "episodes": len(records),
        "transitions": total_transitions,
        "observations": total_observations,
        "bulk_compressed_bytes": bulk_bytes,
        "trajectory_compressed_bytes": trajectory_bytes,
        "replay_compressed_bytes": replay_bytes,
        "observation_chunks": len(observation_entries),
        "trajectory_artifacts": len(trajectory_entries),
        "replay_artifacts": len(replay_entries),
    }


def _aggregate_only_artifact_summary(
    records: Sequence[EpisodeRecord],
    *,
    score_profile: str,
    reason: str,
) -> dict[str, PolicyValue]:
    return {
        "schema": (
            "crafter/complete-feedback-summary/v2"
            if score_profile == "survival-development-v3"
            else "crafter/complete-feedback-summary/v1"
        ),
        "complete": False,
        "detail_scope": "aggregate-only",
        "reason": reason,
        "detailed_artifact_episode_limit": (
            _DETAILED_FEEDBACK_MAX_EPISODES
        ),
        "episodes": len(records),
        "transitions": sum(record.steps for record in records),
        "observations": sum(record.steps + 1 for record in records),
        "bulk_compressed_bytes": 0,
        "trajectory_compressed_bytes": 0,
        "replay_compressed_bytes": 0,
        "observation_chunks": 0,
        "trajectory_artifacts": 0,
        "replay_artifacts": 0,
    }


def _trajectory_artifact(
    record: EpisodeRecord,
    *,
    episode_index: int,
    score_profile: str = "upstream",
    failure_return: float | None = None,
) -> Artifact:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as stream:
        episode_header: dict[str, object] = {
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
            "unlocked_achievements": sorted(_scored_achievements(record)),
            "initial_observation_index": 0,
            "final_observation_index": record.steps,
        }
        if score_profile == "survival-development-v3":
            if failure_return is None:
                raise ValueError("Crafter v3 trajectory requires failure return")
            scored_return = (
                record.total_reward
                if record.policy_failure is None
                else failure_return
            )
            episode_header.update(
                {
                    "scored": record.policy_failure is None,
                    "partial_return": record.total_reward,
                    "scored_return": scored_return,
                    "return": scored_return,
                    "reward_profile": "survival-development-v3",
                }
            )
        stream.write(_json_line(episode_header))
        for step_index, transition in enumerate(record.transitions):
            if type(transition.action) is not int or transition.action not in range(
                len(ACTIONS)
            ):
                raise ValueError("Crafter trajectory Action is invalid")
            transition_line: dict[str, object] = {
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
            if score_profile == "survival-development-v3":
                upstream_reward = _transition_upstream_reward(
                    transition.step.metrics
                )
                components = _transition_score_delta_components(
                    transition.step.metrics
                )
                if upstream_reward is None or components is None:
                    raise ValueError("Crafter v3 trajectory metrics are incomplete")
                transition_line.update(
                    {
                        "upstream_reward": upstream_reward,
                        "score_delta_components": components,
                    }
                )
            stream.write(_json_line(transition_line))
    return Artifact(
        name=(
            f"trajectories/episode-{episode_index:06d}/"
            "trajectory-000000.jsonl.gz"
        ),
        media_type="application/gzip",
        content=output.getvalue(),
        retention="permanent",
    )


def _replay_artifacts(
    record: EpisodeRecord,
    *,
    episode_index: int,
    frames_per_second: int,
    frame_size: int,
    cache: dict[tuple[int, int, tuple[bytes, ...]], bytes],
) -> tuple[list[Artifact], list[dict[str, object]]]:
    observations = _observations(record)
    artifacts: list[Artifact] = []
    entries: list[dict[str, object]] = []
    for segment_index, start in enumerate(
        range(0, len(observations), _REPLAY_SEGMENT_FRAMES)
    ):
        stop = min(start + _REPLAY_SEGMENT_FRAMES, len(observations))
        segment = observations[start:stop]
        cache_key = None
        if len(segment) <= 4:
            cache_key = (
                frames_per_second,
                frame_size,
                tuple(_observation_array(item).tobytes() for item in segment),
            )
        content = None if cache_key is None else cache.get(cache_key)
        if content is None:
            content = _encode_replay(
                segment,
                frames_per_second=frames_per_second,
                frame_size=frame_size,
            )
            if cache_key is not None:
                cache[cache_key] = content
        name = (
            f"replays/episode-{episode_index:06d}/"
            f"replay-{segment_index:06d}.mp4"
        )
        artifact = Artifact(
            name=name,
            media_type="video/mp4",
            content=content,
            retention="bulk",
        )
        artifacts.append(artifact)
        entries.append(
            {
                "episode_index": episode_index,
                "artifact": name,
                "segment_index": segment_index,
                "video_frames": stop - start,
                "first_observation_index": start,
                "last_observation_index": stop - 1,
                "frames_per_second": frames_per_second,
                "frame_size": [frame_size, frame_size],
                "codec": "h264",
                "audio": False,
                "compressed_bytes": artifact.size,
            }
        )
    return artifacts, entries


def _encode_replay(
    observations: Sequence[PolicyValue],
    *,
    frames_per_second: int,
    frame_size: int,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="evopolicygym-crafter-replay-") as temporary:
        path = Path(temporary) / "replay.mp4"
        writer = imageio_ffmpeg.write_frames(
            path,
            (frame_size, frame_size),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=frames_per_second,
            quality=7,
            codec="libx264",
            macro_block_size=16,
            ffmpeg_log_level="error",
            output_params=["-an", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            for observation in observations:
                frame = _observation_array(observation)
                if frame_size != _OBSERVATION_SHAPE[0]:
                    frame = np.asarray(
                        Image.fromarray(frame).resize(
                            (frame_size, frame_size),
                            resample=Image.Resampling.NEAREST,
                        ),
                        dtype=np.uint8,
                    )
                writer.send(np.ascontiguousarray(frame))
        finally:
            writer.close()
        return path.read_bytes()


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


__all__ = [
    "CrafterBenchmark",
    "CrafterLongHorizonBenchmark",
    "CrafterSurvivalDevelopmentBenchmark",
]
