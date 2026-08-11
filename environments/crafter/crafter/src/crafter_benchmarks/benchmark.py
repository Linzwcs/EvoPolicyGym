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

from .config import CrafterConfig, ObservationProfile
from .constants import (
    ACHIEVEMENTS,
    ACTIONS,
    SYMBOLIC_ENTITY_NAMES,
    SYMBOLIC_FACING_NAMES,
    SYMBOLIC_INVENTORY_KEYS,
    SYMBOLIC_PLAYER_CENTER,
    SYMBOLIC_TERRAIN_NAMES,
    SYMBOLIC_VIEW_SHAPE,
)
from .environment import CrafterEnvironment
from .lhs_scoring import (
    LHS_ALIVE_ALPHA,
    LHS_COMPONENT_NAMES,
    LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT,
    LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT,
    LHS_FEEDBACK_SURVIVAL_TAIL_FRACTION,
    LHS_FIRST_UNLOCK_BASE_CREDIT,
    LHS_FIRST_UNLOCK_CREDIT_MAX,
    LHS_FIRST_UNLOCK_CREDITS,
    LHS_HEALTHY_WINDOW_CREDIT,
    LHS_HEALTHY_WINDOW_STEPS,
    LHS_MAINTENANCE_RESOURCE_SHARES,
    LHS_MAINTENANCE_RESTORE,
    LHS_MAINTENANCE_RESTORE_UNIT_CAPS,
    LHS_MAINTENANCE_UNIT_CREDITS,
    LHS_MAINTENANCE_WINDOW_CREDIT,
    LHS_POLICY_FAILURE_RETURN,
    LHS_PRODUCTIVITY_REPEAT_FRACTION,
    LHS_PRODUCTIVITY_REPEAT_QUOTAS,
    LHS_REPEAT_WINDOW_STEPS,
    LHS_REWARD_PROFILE,
    LHS_SECONDARY_COMPONENT_NAMES,
    LHS_SURVIVAL_COMPONENT_NAMES,
    LHS_SURVIVAL_THRESHOLDS,
    LHS_VITAL_AGE_BANDS,
    LHS_VITAL_ALPHA,
    LHSScoringState,
    lhs_feedback_score,
    lhs_score_delta,
)
from .symbolic import symbolic_observation_arrays

_EPISODE_SEED_DOMAIN = b"evopolicygym-crafter/episode-seed/v1\0"
_EPISODE_ARTIFACT_SCENARIO_KEY = "publish_detailed_artifacts"
_SPLITS = frozenset({"train", "validation", "test"})
_OBSERVATION_SHAPE = (64, 64, 3)
_OBSERVATION_BYTES = 64 * 64 * 3
_OBSERVATION_CHUNK_FRAMES = 1_024
_MP4_REPLAY_FPS = 10
_MP4_REPLAY_SIZE = 256
_MP4_REPLAY_BITRATE = "96k"
_DETAILED_FEEDBACK_MAX_EPISODES = 64
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
_V3_SCORE_TOLERANCE = 1e-12


class CrafterBenchmark:
    """Official shifted-geometric achievement score over seeded Episodes."""

    _artifact_score_profile = "upstream"

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
            reward_profile="upstream",
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
            score_profile=self._artifact_score_profile,
            detailed_artifacts=_detailed_artifacts_enabled(records),
            include_mp4=self._config.include_mp4_feedback,
            observation_profile=self._config.observation_profile,
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


class CrafterLongHorizonSurvivalBenchmark(CrafterBenchmark):
    """Default Long-Horizon Survival Score with a survival-selected tail."""

    def __init__(self, config: CrafterConfig | None = None) -> None:
        super().__init__(config)
        self._spec = _long_horizon_survival_spec(self._config)

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return CrafterEnvironment(
            _environment_episode(episode),
            config=self._config,
            reward_profile=LHS_REWARD_PROFILE,
        )

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        analyses = tuple(_lhs_episode_analysis(record) for record in records)
        content = _lhs_profile(records, analyses)
        score_value = content["long_horizon_survival_score"]
        if not isinstance(score_value, float):
            raise RuntimeError("Crafter LHS score is invalid")
        artifacts, artifact_summary = _complete_feedback_artifacts(
            records,
            score_profile=LHS_REWARD_PROFILE,
            failure_return=LHS_POLICY_FAILURE_RETURN,
            detailed_artifacts=_detailed_artifacts_enabled(records),
            include_mp4=self._config.include_mp4_feedback,
            observation_profile=self._config.observation_profile,
        )
        content["detailed_feedback"] = artifact_summary
        return Feedback(
            score=score_value,
            content=content,
            artifacts=artifacts,
        )


def _profiled_benchmark_id(config: CrafterConfig, metric: str) -> str:
    prefix = "crafter/CrafterReward-v1"
    if config.observation_profile == "local-symbolic-v1":
        return f"{prefix}/local-symbolic-v1/{metric}"
    return f"{prefix}/{metric}"


def _observation_description(config: CrafterConfig) -> str:
    if config.observation_profile == "rgb":
        return "canonical 64x64 RGB observations"
    return "a player-centered 7x9 local-symbolic projection"


def _trajectory_schema(config: CrafterConfig, rgb_schema: str) -> str:
    if config.observation_profile == "local-symbolic-v1":
        return "crafter/local-symbolic-feedback-manifest/v1"
    return rgb_schema


def _observation_space(config: CrafterConfig) -> dict[str, PolicyValue]:
    if config.observation_profile == "rgb":
        return {
            "type": "tensor",
            "dtype": "uint8",
            "shape": [64, 64, 3],
            "color_space": "RGB",
        }
    return {
        "type": "mapping",
        "fields": {
            "terrain": {
                "policy_carrier": "TensorValue",
                "dtype": "uint8",
                "shape": list(SYMBOLIC_VIEW_SHAPE),
                "layout": "row-major [row, column]",
                "value_meanings": {
                    str(index): (
                        "crafting table" if name == "table" else name
                    )
                    for index, name in enumerate(SYMBOLIC_TERRAIN_NAMES)
                },
            },
            "entities": {
                "policy_carrier": "TensorValue",
                "dtype": "uint8",
                "shape": list(SYMBOLIC_VIEW_SHAPE),
                "layout": "row-major [row, column]",
                "value_meanings": {
                    str(index): name
                    for index, name in enumerate(SYMBOLIC_ENTITY_NAMES)
                },
            },
            "inventory": {
                "policy_carrier": "mapping of exact int",
                "keys": list(SYMBOLIC_INVENTORY_KEYS),
                "minimum": 0,
                "maximum": 9,
            },
            "facing": {
                "policy_carrier": "str",
                "values": list(SYMBOLIC_FACING_NAMES),
            },
            "sleeping": {"policy_carrier": "bool"},
            "daylight": {
                "policy_carrier": "float",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
    }


def _public_observations(config: CrafterConfig) -> dict[str, PolicyValue]:
    common: dict[str, PolicyValue] = {
        "format": "compressed NumPy NPZ",
        "observations_per_artifact": _OBSERVATION_CHUNK_FRAMES,
        "source_alignment": "observation index is exact Policy input",
        "complete_artifact_episode_limit": _DETAILED_FEEDBACK_MAX_EPISODES,
        "detailed_artifact_splits": ["train"],
    }
    if config.observation_profile == "rgb":
        common.update(
            {
                "dtype": "uint8",
                "shape": [64, 64, 3],
                "layout": "HWC RGB",
                "frame_sampling": "none",
                "pixel_exact": True,
                "optional_mp4": {
                    "enabled": config.include_mp4_feedback,
                    "format": "H.264 MP4",
                    "frames_per_second": _MP4_REPLAY_FPS,
                    "frame_size": [_MP4_REPLAY_SIZE, _MP4_REPLAY_SIZE],
                    "target_bitrate": _MP4_REPLAY_BITRATE,
                    "frame_sampling": "none",
                    "pixel_exact": False,
                    "audio": False,
                    "role": "derived viewing aid; NPZ remains authoritative",
                },
            }
        )
        return common
    common.update(
        {
            "observation_profile": "local-symbolic-v1",
            "arrays": {
                "terrain": "uint8 [observation, 7, 9]",
                "entities": "uint8 [observation, 7, 9]",
                "inventory": "uint8 [observation, 16]",
                "facing": "uint8 [observation]",
                "sleeping": "bool [observation]",
                "daylight": "float64 [observation]",
                "observation_indices": "uint32 [observation]",
            },
            "inventory_order": list(SYMBOLIC_INVENTORY_KEYS),
            "facing_ids": {
                str(index): name
                for index, name in enumerate(SYMBOLIC_FACING_NAMES)
            },
            "lossless": True,
            "optional_mp4": {"enabled": False, "supported": False},
        }
    )
    return common


def _environment_parameters(config: CrafterConfig) -> dict[str, PolicyValue]:
    parameters: dict[str, PolicyValue] = {
        "area": [64, 64],
        "view": [9, 9],
        "image_size": [64, 64],
        "reward": True,
        "max_episode_steps": config.max_episode_steps,
        "include_mp4_feedback": config.include_mp4_feedback,
    }
    if config.observation_profile == "local-symbolic-v1":
        parameters.update(
            {
                "observation_profile": "local-symbolic-v1",
                "symbolic_view_rows": SYMBOLIC_VIEW_SHAPE[0],
                "symbolic_view_columns": SYMBOLIC_VIEW_SHAPE[1],
                "symbolic_player_row": SYMBOLIC_PLAYER_CENTER[0],
                "symbolic_player_column": SYMBOLIC_PLAYER_CENTER[1],
            }
        )
    return parameters


def _symbolic_metadata(config: CrafterConfig) -> dict[str, PolicyValue]:
    if config.observation_profile == "rgb":
        return {}
    return {
        "observation_profile": "local-symbolic-v1",
        "observation_source": (
            "Benchmark-authored local projection of pinned Crafter 1.8.3; "
            "not an upstream Crafter registration and not Craftax"
        ),
        "symbolic_geometry": {
            "shape": list(SYMBOLIC_VIEW_SHAPE),
            "player_center": list(SYMBOLIC_PLAYER_CENTER),
            "row_axis": "up-to-down",
            "column_axis": "left-to-right",
        },
        "terrain_ids": {
            str(index): name
            for index, name in enumerate(SYMBOLIC_TERRAIN_NAMES)
        },
        "entity_ids": {
            str(index): name
            for index, name in enumerate(SYMBOLIC_ENTITY_NAMES)
        },
        "inventory_keys": list(SYMBOLIC_INVENTORY_KEYS),
        "privacy_boundary": {
            "global_semantic_map": "forbidden",
            "absolute_player_position": "forbidden",
            "environment_seed": "forbidden",
            "world_rng": "forbidden",
            "achievement_counters": "forbidden as observation",
            "hidden_life_counters": "forbidden",
            "entity_health_and_cooldowns": "forbidden",
        },
    }


def _spec(config: CrafterConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=_profiled_benchmark_id(config, "achievement-score-v1"),
        description=(
            "Survive and unlock Crafter's 22 achievements from "
            f"{_observation_description(config)}. Maximize the official shifted-geometric "
            "achievement success score."
        ),
        observation_space=_observation_space(config),
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
            **_symbolic_metadata(config),
            "public_observations": _public_observations(config),
            "privileged_information_exposed": False,
        },
        environment_parameters=_environment_parameters(config),
        max_episode_steps=config.max_episode_steps,
        primary_metric="crafter_score_percent",
        score_direction="maximize",
    )


def _long_horizon_survival_spec(config: CrafterConfig) -> BenchmarkSpec:
    canonical = _spec(config)
    metadata = dict(canonical.metadata)
    maintenance_restore: dict[str, PolicyValue] = {
        name: {"resource": resource, "nominal_units": units}
        for name, (resource, units) in LHS_MAINTENANCE_RESTORE.items()
    }
    metadata.update(
        {
            "objective_profile": LHS_REWARD_PROFILE,
            "canonical_comparison_metric": "crafter_score_percent",
            "step_reward_formula": (
                "0.01 * alive + 0.03 * alive * "
                "min(health, food, drink) / 9 + first_unlock_delta + "
                "maintenance_repeat_delta + productivity_repeat_delta"
            ),
            "episode_return_formula": "sum(step_reward)",
            "feedback_score_formula": (
                "0.75 * mean(survival_return) + 0.25 * "
                "mean(bottom ceil(0.25*N) survival returns) + "
                "mean(secondary_return)"
            ),
            "survival": {
                "alive_alpha": LHS_ALIVE_ALPHA,
                "vital_alpha": LHS_VITAL_ALPHA,
                "vital_quality": "min(health, food, drink) / 9",
                "energy_scored": False,
                "healthy_window_steps": LHS_HEALTHY_WINDOW_STEPS,
                "healthy_window_credit": LHS_HEALTHY_WINDOW_CREDIT,
                "survival_component_names": list(
                    LHS_SURVIVAL_COMPONENT_NAMES
                ),
            },
            "first_unlock": {
                "credit_formula": "0.10 * log2(1 + raw_weight)",
                "base_credit": LHS_FIRST_UNLOCK_BASE_CREDIT,
                "credits": dict(LHS_FIRST_UNLOCK_CREDITS),
                "maximum_credit": LHS_FIRST_UNLOCK_CREDIT_MAX,
            },
            "maintenance_repeat": {
                "window_steps": LHS_REPEAT_WINDOW_STEPS,
                "window_credit": LHS_MAINTENANCE_WINDOW_CREDIT,
                "resource_shares": dict(
                    LHS_MAINTENANCE_RESOURCE_SHARES
                ),
                "credited_restore_unit_caps": dict(
                    LHS_MAINTENANCE_RESTORE_UNIT_CAPS
                ),
                "credit_per_restore_unit": dict(
                    LHS_MAINTENANCE_UNIT_CREDITS
                ),
                "event_restore": maintenance_restore,
                "repeats_only": True,
                "actual_restoration_only": True,
            },
            "productivity_repeat": {
                "window_steps": LHS_REPEAT_WINDOW_STEPS,
                "fraction_of_first_unlock_credit": (
                    LHS_PRODUCTIVITY_REPEAT_FRACTION
                ),
                "credited_event_quotas": dict(
                    LHS_PRODUCTIVITY_REPEAT_QUOTAS
                ),
                "repeats_only": True,
            },
            "feedback_aggregation": {
                "survival_mean_weight": (
                    LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT
                ),
                "survival_lower_tail_weight": (
                    LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT
                ),
                "survival_tail_fraction": (
                    LHS_FEEDBACK_SURVIVAL_TAIL_FRACTION
                ),
                "secondary_mean_weight": 1.0,
                "upper_tail_weight": 0.0,
                "tail_selection": "survival_return_only",
                "tail_count": "max(1, ceil(0.25 * Episodes))",
            },
            "policy_failure_return": LHS_POLICY_FAILURE_RETURN,
            "policy_failure_trace": (
                "partial trajectory retained for diagnosis; partial score "
                "discarded and formal Episode return is zero"
            ),
            "upstream_reward_scored": False,
            "upstream_reward_field": "upstream_reward",
            "trajectory_schema": _trajectory_schema(
                config, "crafter/complete-feedback-manifest/v8"
            ),
        }
    )
    environment_parameters = dict(canonical.environment_parameters)
    environment_parameters.update(
        {
            "reward_profile": LHS_REWARD_PROFILE,
            "alive_survival_alpha": LHS_ALIVE_ALPHA,
            "vital_survival_alpha": LHS_VITAL_ALPHA,
            "repeat_window_steps": LHS_REPEAT_WINDOW_STEPS,
        }
    )
    return BenchmarkSpec(
        id=_profiled_benchmark_id(
            config, "long-horizon-survival-score-v1"
        ),
        description=(
            "Prioritize robust long survival with independent alive and "
            "weakest-vital transition credit. Select the lower tail using "
            "survival alone, while retaining mean cadence-bounded maintenance "
            "and compressed development credit without an upper-tail bonus."
        ),
        observation_space=canonical.observation_space,
        action_space=canonical.action_space,
        metadata=metadata,
        environment_parameters=environment_parameters,
        max_episode_steps=canonical.max_episode_steps,
        primary_metric="long_horizon_survival_score",
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
        "lhs_repeat_diagnostics",
        "lhs_score_delta_components",
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
    _transition_lhs_repeat_diagnostics(metrics)
    _transition_lhs_score_delta_components(metrics)
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
    vital_names = ("health", "food", "drink")
    if type(value) is not dict or set(value) != set(vital_names):
        raise ValueError("Crafter maintenance vitals are invalid")
    vitals: dict[str, int] = {}
    for name in vital_names:
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


def _transition_lhs_score_delta_components(
    metrics: PolicyValue,
) -> dict[str, float] | None:
    if type(metrics) is not dict:
        raise ValueError("Crafter transition metrics are invalid")
    if "lhs_score_delta_components" not in metrics:
        return None
    value = metrics["lhs_score_delta_components"]
    expected = set(LHS_COMPONENT_NAMES)
    if type(value) is not dict or set(value) != expected:
        raise ValueError("Crafter LHS score delta components are invalid")
    components: dict[str, float] = {}
    for name in LHS_COMPONENT_NAMES:
        amount = value[name]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or float(amount) < 0.0
        ):
            raise ValueError("Crafter LHS score delta components are invalid")
        components[name] = float(amount)
    return components


def _transition_lhs_repeat_diagnostics(
    metrics: PolicyValue,
) -> dict[str, dict[str, int]] | None:
    if type(metrics) is not dict:
        raise ValueError("Crafter transition metrics are invalid")
    if "lhs_repeat_diagnostics" not in metrics:
        return None
    value = metrics["lhs_repeat_diagnostics"]
    expected = {
        "maintenance_credited_units",
        "productivity_credited_events",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("Crafter LHS repeat diagnostics are invalid")
    maintenance = value["maintenance_credited_units"]
    productivity = value["productivity_credited_events"]
    if type(maintenance) is not dict or set(maintenance) != {"drink", "food"}:
        raise ValueError("Crafter LHS maintenance diagnostics are invalid")
    if type(productivity) is not dict or not set(productivity).issubset(
        LHS_PRODUCTIVITY_REPEAT_QUOTAS
    ):
        raise ValueError("Crafter LHS productivity diagnostics are invalid")
    parsed_maintenance: dict[str, int] = {}
    parsed_productivity: dict[str, int] = {}
    for name, amount in maintenance.items():
        if type(amount) is not int or amount < 0:
            raise ValueError("Crafter LHS maintenance diagnostics are invalid")
        parsed_maintenance[name] = amount
    for name, amount in productivity.items():
        if type(name) is not str or type(amount) is not int or amount <= 0:
            raise ValueError("Crafter LHS productivity diagnostics are invalid")
        parsed_productivity[name] = amount
    return {
        "maintenance_credited_units": parsed_maintenance,
        "productivity_credited_events": parsed_productivity,
    }


def _crafter_score(success_rates: Sequence[float]) -> float:
    if len(success_rates) != len(ACHIEVEMENTS):
        raise ValueError("Crafter score requires every achievement")
    if any(not math.isfinite(rate) or not 0.0 <= rate <= 100.0 for rate in success_rates):
        raise ValueError("Crafter success rates are invalid")
    return math.expm1(statistics.fmean(math.log1p(rate) for rate in success_rates))


@dataclass(frozen=True, slots=True)
class _LHSEpisodeAnalysis:
    status: str
    failure: str | None
    terminated: bool
    truncated: bool
    steps: int
    effective_survival_steps: int
    scored_return: float
    partial_return: float
    components: dict[str, float]
    partial_components: dict[str, float]
    upstream_return: float
    unlocked: frozenset[str]
    partial_unlocked: frozenset[str]
    event_totals: dict[str, int]
    partial_event_totals: dict[str, int]
    maintenance_units: dict[str, int]
    productivity_events: dict[str, int]
    alive_vital_steps: int
    vital_quality_sum: float
    minimum_vital_sum: int
    zero_min_vital_steps: int
    low_vital_steps: dict[int, dict[str, int]]
    vital_age_bands: dict[str, tuple[int, float, int]]
    terminal_vitals: dict[str, int] | None


def _lhs_episode_analysis(record: EpisodeRecord) -> _LHSEpisodeAnalysis:
    state = LHSScoringState()
    component_values: dict[str, list[float]] = {
        name: [] for name in LHS_COMPONENT_NAMES
    }
    upstream_rewards: list[float] = []
    transition_rewards: list[float] = []
    unlocked_seen: set[str] = set()
    maintenance_units = {"drink": 0, "food": 0}
    productivity_events = {
        name: 0 for name in LHS_PRODUCTIVITY_REPEAT_QUOTAS
    }
    alive_vital_steps = 0
    vital_quality_sum = 0.0
    minimum_vital_sum = 0
    zero_min_vital_steps = 0
    low_vital_steps = {
        threshold: {"health": 0, "food": 0, "drink": 0}
        for threshold in (2, 5)
    }
    age_steps = {label: 0 for label, _, _ in LHS_VITAL_AGE_BANDS}
    age_quality = {label: 0.0 for label, _, _ in LHS_VITAL_AGE_BANDS}
    age_minimum = {label: 0 for label, _, _ in LHS_VITAL_AGE_BANDS}
    terminal_vitals: dict[str, int] | None = None

    for transition_index, transition in enumerate(record.transitions):
        metrics = transition.step.metrics
        unlocked, event_counts = _transition_metrics(metrics)
        vitals = _transition_maintenance_vitals(metrics)
        upstream_reward = _transition_upstream_reward(metrics)
        reported_components = _transition_lhs_score_delta_components(metrics)
        reported_diagnostics = _transition_lhs_repeat_diagnostics(metrics)
        if (
            vitals is None
            or upstream_reward is None
            or reported_components is None
            or reported_diagnostics is None
        ):
            raise ValueError(
                "Crafter LHS transitions require shaped reward metrics"
            )
        expected_components, expected_diagnostics = state.transition(
            terminated=transition.step.terminated,
            unlocked=unlocked,
            event_counts=event_counts,
            vitals=vitals,
        )
        for name in LHS_COMPONENT_NAMES:
            expected = expected_components[name]
            if not math.isclose(
                reported_components[name],
                expected,
                rel_tol=0.0,
                abs_tol=_V3_SCORE_TOLERANCE,
            ):
                raise ValueError("Crafter LHS reported score component drifted")
            component_values[name].append(expected)
        if reported_diagnostics != expected_diagnostics:
            raise ValueError("Crafter LHS repeat diagnostics drifted")
        expected_reward = lhs_score_delta(expected_components)
        if not math.isclose(
            transition.step.reward,
            expected_reward,
            rel_tol=0.0,
            abs_tol=_V3_SCORE_TOLERANCE,
        ):
            raise ValueError(
                "Crafter LHS Step.reward does not match its components"
            )
        transition_rewards.append(transition.step.reward)
        upstream_rewards.append(upstream_reward)
        unlocked_seen.update(unlocked)

        maintenance = cast(
            dict[str, int],
            expected_diagnostics["maintenance_credited_units"],
        )
        productivity = cast(
            dict[str, int],
            expected_diagnostics["productivity_credited_events"],
        )
        for name, units in maintenance.items():
            maintenance_units[name] += units
        for name, count in productivity.items():
            productivity_events[name] += count

        if not transition.step.terminated:
            alive_vital_steps += 1
            minimum = min(vitals.values())
            quality = minimum / 9.0
            vital_quality_sum += quality
            minimum_vital_sum += minimum
            zero_min_vital_steps += minimum == 0
            for threshold in low_vital_steps:
                for name, amount in vitals.items():
                    low_vital_steps[threshold][name] += amount <= threshold
            for label, start, stop in LHS_VITAL_AGE_BANDS:
                if transition_index >= start and (
                    stop is None or transition_index < stop
                ):
                    age_steps[label] += 1
                    age_quality[label] += quality
                    age_minimum[label] += minimum
                    break
        else:
            terminal_vitals = dict(vitals)

    partial_components = {
        name: math.fsum(component_values[name])
        for name in LHS_COMPONENT_NAMES
    }
    partial_return = math.fsum(transition_rewards)
    reconstructed_partial = math.fsum(partial_components.values())
    if not math.isclose(
        partial_return,
        reconstructed_partial,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Crafter LHS Episode return does not reconstruct")
    if not math.isclose(
        record.total_reward,
        partial_return,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Crafter LHS Episode total reward drifted")

    failed = record.policy_failure is not None
    zero_components = {name: 0.0 for name in LHS_COMPONENT_NAMES}
    zero_events = {name: 0 for name in ACHIEVEMENTS}
    zero_age_bands = {
        label: (0, 0.0, 0) for label, _, _ in LHS_VITAL_AGE_BANDS
    }
    return _LHSEpisodeAnalysis(
        status="policy_failed" if failed else "completed",
        failure=record.policy_failure,
        terminated=False if failed else _terminated(record),
        truncated=False if failed else _truncated(record),
        steps=record.steps,
        effective_survival_steps=(
            0 if failed else record.steps - int(_terminated(record))
        ),
        scored_return=LHS_POLICY_FAILURE_RETURN if failed else partial_return,
        partial_return=partial_return,
        components=zero_components if failed else dict(partial_components),
        partial_components=partial_components,
        upstream_return=math.fsum(upstream_rewards),
        unlocked=frozenset() if failed else frozenset(unlocked_seen),
        partial_unlocked=frozenset(unlocked_seen),
        event_totals=zero_events if failed else dict(state.event_totals),
        partial_event_totals=dict(state.event_totals),
        maintenance_units=(
            {"drink": 0, "food": 0} if failed else maintenance_units
        ),
        productivity_events=(
            {name: 0 for name in LHS_PRODUCTIVITY_REPEAT_QUOTAS}
            if failed
            else productivity_events
        ),
        alive_vital_steps=0 if failed else alive_vital_steps,
        vital_quality_sum=0.0 if failed else vital_quality_sum,
        minimum_vital_sum=0 if failed else minimum_vital_sum,
        zero_min_vital_steps=0 if failed else zero_min_vital_steps,
        low_vital_steps=(
            {
                threshold: {name: 0 for name in counts}
                for threshold, counts in low_vital_steps.items()
            }
            if failed
            else low_vital_steps
        ),
        vital_age_bands=(
            zero_age_bands
            if failed
            else {
                label: (
                    age_steps[label],
                    age_quality[label],
                    age_minimum[label],
                )
                for label, _, _ in LHS_VITAL_AGE_BANDS
            }
        ),
        terminal_vitals=None if failed else terminal_vitals,
    )


def _lhs_profile(
    records: Sequence[EpisodeRecord],
    analyses: Sequence[_LHSEpisodeAnalysis],
) -> dict[str, PolicyValue]:
    if len(records) != len(analyses) or not analyses:
        raise ValueError("Crafter LHS analyses do not align with Episodes")
    episodes = len(analyses)
    returns = tuple(item.scored_return for item in analyses)
    survival_returns = tuple(
        math.fsum(item.components[name] for name in LHS_SURVIVAL_COMPONENT_NAMES)
        for item in analyses
    )
    secondary_returns = tuple(
        math.fsum(item.components[name] for name in LHS_SECONDARY_COMPONENT_NAMES)
        for item in analyses
    )
    (
        mean_survival,
        lower_survival,
        mean_secondary,
        tail_count,
        score,
    ) = lhs_feedback_score(survival_returns, secondary_returns)
    ordered_indices = sorted(
        range(episodes), key=lambda index: (survival_returns[index], index)
    )
    lower_indices = ordered_indices[:tail_count]

    component_aggregates: dict[str, PolicyValue] = {}
    reconstructed_score = 0.0
    for name in LHS_COMPONENT_NAMES:
        mean_component = statistics.fmean(
            item.components[name] for item in analyses
        )
        if name in LHS_SURVIVAL_COMPONENT_NAMES:
            lower_component = statistics.fmean(
                analyses[index].components[name] for index in lower_indices
            )
            contribution = math.fsum(
                (
                    LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT * mean_component,
                    LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT
                    * lower_component,
                )
            )
            component_aggregates[name] = {
                "mean": mean_component,
                "survival_lower_tail_mean": lower_component,
                "feedback_contribution": contribution,
            }
        else:
            contribution = mean_component
            component_aggregates[name] = {
                "mean": mean_component,
                "feedback_contribution": contribution,
            }
        reconstructed_score += contribution

    effective_steps = tuple(item.effective_survival_steps for item in analyses)
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
    maintenance_units = {
        resource: sum(item.maintenance_units[resource] for item in analyses)
        for resource in LHS_MAINTENANCE_RESTORE_UNIT_CAPS
    }
    productivity_events = {
        name: sum(item.productivity_events[name] for item in analyses)
        for name in LHS_PRODUCTIVITY_REPEAT_QUOTAS
    }
    mean_return = statistics.fmean(returns)
    return_variance = statistics.variance(returns) if episodes > 1 else 0.0
    return_standard_deviation = math.sqrt(return_variance)
    return_standard_error = return_standard_deviation / math.sqrt(episodes)
    confidence_half_width = 1.96 * return_standard_error
    survival_at_steps: dict[str, PolicyValue] = {}
    for threshold in LHS_SURVIVAL_THRESHOLDS:
        count = sum(value >= threshold for value in effective_steps)
        survival_at_steps[str(threshold)] = {
            "count": count,
            "percent": 100.0 * count / episodes,
        }
    alive_vital_steps = sum(item.alive_vital_steps for item in analyses)
    vital_quality_sum = math.fsum(item.vital_quality_sum for item in analyses)
    minimum_vital_sum = sum(item.minimum_vital_sum for item in analyses)
    low_vital_steps: dict[str, PolicyValue] = {
        str(threshold): {
            name: sum(
                item.low_vital_steps[threshold][name] for item in analyses
            )
            for name in ("health", "food", "drink")
        }
        for threshold in (2, 5)
    }
    vital_quality_by_age: dict[str, PolicyValue] = {}
    for label, start, stop in LHS_VITAL_AGE_BANDS:
        band_steps = sum(item.vital_age_bands[label][0] for item in analyses)
        band_quality = math.fsum(
            item.vital_age_bands[label][1] for item in analyses
        )
        band_minimum = sum(
            item.vital_age_bands[label][2] for item in analyses
        )
        vital_quality_by_age[label] = {
            "start_step_inclusive": start,
            "stop_step_exclusive": stop,
            "alive_steps": band_steps,
            "mean_vital_quality": (
                band_quality / band_steps if band_steps else 0.0
            ),
            "mean_min_vital": (
                band_minimum / band_steps if band_steps else 0.0
            ),
        }
    deaths = tuple(item for item in analyses if item.terminated)
    failure_counts: dict[str, int] = {}
    for item in analyses:
        if item.failure is not None:
            code = _policy_failure_detail(item.failure, item.steps)["code"]
            assert isinstance(code, str)
            failure_counts[code] = failure_counts.get(code, 0) + 1

    episode_summaries: list[PolicyValue] = []
    for index, item in enumerate(analyses):
        survival_return = math.fsum(
            item.components[name] for name in LHS_SURVIVAL_COMPONENT_NAMES
        )
        secondary_return = math.fsum(
            item.components[name] for name in LHS_SECONDARY_COMPONENT_NAMES
        )
        episode_summaries.append(
            {
                "episode_index": index,
                "status": item.status,
                "terminated": item.terminated,
                "truncated": item.truncated,
                "failure": (
                    None
                    if item.failure is None
                    else _policy_failure_detail(item.failure, item.steps)
                ),
                "steps": item.steps,
                "effective_survival_steps": item.effective_survival_steps,
                "return": item.scored_return,
                "survival_return": survival_return,
                "secondary_return": secondary_return,
                "partial_return": item.partial_return,
                "partial_credit_discarded": item.failure is not None,
                "components": dict(item.components),
                "partial_components": dict(item.partial_components),
            }
        )

    return {
        "schema": "crafter/long-horizon-survival-feedback/v1",
        "score_profile": LHS_REWARD_PROFILE,
        "summary": (
            f"Long-Horizon Survival Score {score:.3f} across {episodes} "
            f"Episodes; mean survival return {mean_survival:.3f}, "
            f"survival lower-tail mean {lower_survival:.3f}, mean secondary "
            f"return {mean_secondary:.3f}, and canonical Crafter score "
            f"{canonical_score:.3f}%."
        ),
        "long_horizon_survival_score": score,
        "feedback_aggregation": {
            "formula": (
                "0.75 * mean(survival_return) + 0.25 * "
                "survival_lower_tail_mean + mean(secondary_return)"
            ),
            "survival_mean_weight": LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT,
            "survival_lower_tail_weight": (
                LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT
            ),
            "secondary_mean_weight": 1.0,
            "upper_tail_weight": 0.0,
            "tail_fraction": LHS_FEEDBACK_SURVIVAL_TAIL_FRACTION,
            "tail_count": tail_count,
            "tail_selection": "survival_return_only",
            "mean_survival_return": mean_survival,
            "survival_lower_tail_mean": lower_survival,
            "mean_secondary_return": mean_secondary,
            "survival_lower_tail_episode_indices": cast(
                list[PolicyValue], lower_indices
            ),
        },
        "score_components": {
            "by_component": component_aggregates,
            "reconstructed_feedback_score": reconstructed_score,
            "reconstruction_error": score - reconstructed_score,
        },
        "episode_returns": {
            "mean": mean_return,
            "variance": return_variance,
            "standard_deviation": return_standard_deviation,
            "standard_error": return_standard_error,
            "confidence_interval_95": {
                "lower": mean_return - confidence_half_width,
                "upper": mean_return + confidence_half_width,
                "half_width": confidence_half_width,
                "method": "normal approximation for the arithmetic mean",
            },
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
        "policy_failure_return": LHS_POLICY_FAILURE_RETURN,
        "policy_failure_counts": cast(dict[str, PolicyValue], failure_counts),
        "survival_steps": {
            "mean": statistics.fmean(effective_steps),
            "median": statistics.median(effective_steps),
            "p10": _nearest_rank(effective_steps, 0.10),
            "p25": _nearest_rank(effective_steps, 0.25),
            "p90": _nearest_rank(effective_steps, 0.90),
            "min": min(effective_steps),
            "max": max(effective_steps),
        },
        "survival_at_steps": survival_at_steps,
        "vital_quality": {
            "mean": vital_quality_sum / alive_vital_steps if alive_vital_steps else 0.0,
            "mean_min_vital": (
                minimum_vital_sum / alive_vital_steps
                if alive_vital_steps
                else 0.0
            ),
            "zero_min_vital_steps": sum(
                item.zero_min_vital_steps for item in analyses
            ),
            "steps_at_or_below": low_vital_steps,
            "by_episode_age": vital_quality_by_age,
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
        },
        "first_unlock": {
            "credit_formula": "0.10 * log2(1 + raw_weight)",
            "base_credit": LHS_FIRST_UNLOCK_BASE_CREDIT,
            "maximum_credit_per_episode": LHS_FIRST_UNLOCK_CREDIT_MAX,
            "credits": dict(LHS_FIRST_UNLOCK_CREDITS),
            "achievement_success_percent": success_rates,
        },
        "maintenance_repeat": {
            "window_steps": LHS_REPEAT_WINDOW_STEPS,
            "window_credit": LHS_MAINTENANCE_WINDOW_CREDIT,
            "credited_restore_units": cast(
                dict[str, PolicyValue], maintenance_units
            ),
            "credit_per_restore_unit": dict(
                LHS_MAINTENANCE_UNIT_CREDITS
            ),
            "repeats_only": True,
        },
        "productivity_repeat": {
            "window_steps": LHS_REPEAT_WINDOW_STEPS,
            "fraction_of_first_unlock_credit": (
                LHS_PRODUCTIVITY_REPEAT_FRACTION
            ),
            "credited_event_quotas": dict(
                LHS_PRODUCTIVITY_REPEAT_QUOTAS
            ),
            "credited_events": cast(
                dict[str, PolicyValue], productivity_events
            ),
            "repeats_only": True,
        },
        "canonical_comparison": {
            "crafter_score_percent": canonical_score,
            "mean_upstream_return": statistics.fmean(
                item.upstream_return for item in analyses
            ),
            "achievement_event_counts": cast(
                dict[str, PolicyValue], event_totals
            ),
        },
        "health_change_diagnostics": _health_change_diagnostics(records),
        "episode_score_summaries": episode_summaries,
        "action_diagnostics": _action_diagnostics(records),
        "detailed_feedback": {},
    }


def _policy_failure_detail(failure: str, steps: int) -> dict[str, PolicyValue]:
    if type(failure) is not str or not failure:
        raise ValueError("Crafter Policy failure is invalid")
    code = failure if failure in {"invalid_action", "timeout"} else "policy_error"
    message = failure.splitlines()[0][:512]
    return {
        "code": code,
        "message": message,
        "step_index": steps,
        "last_valid_observation_index": steps,
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
    detailed_artifacts: bool = True,
    include_mp4: bool = False,
    observation_profile: ObservationProfile = "rgb",
) -> tuple[tuple[Artifact, ...], dict[str, PolicyValue]]:
    if score_profile not in {LHS_REWARD_PROFILE, "upstream"}:
        raise ValueError("Crafter Artifact score profile is invalid")
    if score_profile == LHS_REWARD_PROFILE and failure_return is None:
        raise ValueError("Crafter shaped Artifacts require a failure return")
    if type(detailed_artifacts) is not bool:
        raise TypeError("detailed_artifacts must be bool")
    if type(include_mp4) is not bool:
        raise TypeError("include_mp4 must be bool")
    if observation_profile not in {"rgb", "local-symbolic-v1"}:
        raise ValueError("Crafter Artifact observation profile is invalid")
    if observation_profile == "local-symbolic-v1" and include_mp4:
        raise ValueError("Crafter symbolic Artifacts do not support MP4")
    if not detailed_artifacts:
        return (), _aggregate_only_artifact_summary(
            records,
            score_profile=score_profile,
            reason="split_disables_detailed_artifacts",
            include_mp4=include_mp4,
            observation_profile=observation_profile,
        )
    if len(records) > _DETAILED_FEEDBACK_MAX_EPISODES:
        return (), _aggregate_only_artifact_summary(
            records,
            score_profile=score_profile,
            reason="episode_count_exceeds_detailed_artifact_limit",
            include_mp4=include_mp4,
            observation_profile=observation_profile,
        )
    artifacts: list[Artifact] = []
    trajectory_entries: list[dict[str, object]] = []
    observation_entries: list[dict[str, object]] = []
    replay_entries: list[dict[str, object]] = []
    replay_cache: dict[tuple[bytes, ...], bytes] = {}
    total_observations = 0
    total_transitions = 0

    for episode_index, record in enumerate(records):
        trajectory = _trajectory_artifact(
            record,
            episode_index=episode_index,
            score_profile=score_profile,
            failure_return=failure_return,
            observation_profile=observation_profile,
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
        episode_observations, episode_observation_entries = _observation_artifacts(
            record,
            episode_index=episode_index,
            observation_profile=observation_profile,
        )
        artifacts.extend(episode_observations)
        observation_entries.extend(episode_observation_entries)
        if include_mp4:
            replay, replay_entry = _mp4_replay_artifact(
                record,
                episode_index=episode_index,
                cache=replay_cache,
            )
            artifacts.append(replay)
            replay_entries.append(replay_entry)
        total_transitions += record.steps
        total_observations += record.steps + 1

    bulk_bytes = sum(
        artifact.size for artifact in artifacts if artifact.retention == "bulk"
    )
    observation_bytes = sum(
        artifact.size
        for artifact in artifacts
        if artifact.media_type == "application/x-npz"
    )
    trajectory_bytes = sum(
        artifact.size
        for artifact in artifacts
        if artifact.media_type == "application/gzip"
    )
    replay_bytes = sum(
        artifact.size
        for artifact in artifacts
        if artifact.media_type == "video/mp4"
    )
    manifest: dict[str, object] = {
        "schema": (
            "crafter/local-symbolic-feedback-manifest/v1"
            if observation_profile == "local-symbolic-v1"
            else (
                "crafter/complete-feedback-manifest/v8"
                if score_profile == LHS_REWARD_PROFILE
                else "crafter/complete-feedback-manifest/v6"
            )
        ),
        "complete": True,
        "score_profile": score_profile,
        "observation_profile": observation_profile,
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
        "observation_compressed_bytes": observation_bytes,
        "replay_compressed_bytes": replay_bytes,
        "retention": {
            "trajectories": "permanent",
            "observations": (
                "bulk capacity; newest submission is protected"
            ),
            "replays": (
                "bulk capacity; newest submission is protected"
            ),
        },
    }
    if observation_profile == "rgb":
        manifest.update(
            {
                "source_observation": {
                    "color_space": "RGB",
                    "dtype": "uint8",
                    "shape": list(_OBSERVATION_SHAPE),
                    "layout": "HWC",
                },
                "visual_evidence": {
                    "format": "compressed NumPy NPZ",
                    "arrays": {
                        "observations": "uint8 [frame_count, 64, 64, 3]",
                        "observation_indices": "uint32 [frame_count]",
                    },
                    "frame_sampling": "none",
                    "pixel_exact": True,
                    "resizing": "none",
                    "mp4_replays": {
                        "enabled": include_mp4,
                        "format": "H.264 MP4",
                        "frames_per_second": _MP4_REPLAY_FPS,
                        "frame_size": [_MP4_REPLAY_SIZE, _MP4_REPLAY_SIZE],
                        "target_bitrate": _MP4_REPLAY_BITRATE,
                        "frame_sampling": "none",
                        "pixel_exact": False,
                        "audio": False,
                        "role": "derived viewing aid; NPZ remains authoritative",
                    },
                },
            }
        )
    else:
        manifest.update(
            {
                "source_observation": {
                    "type": "mapping",
                    "spatial_shape": list(SYMBOLIC_VIEW_SHAPE),
                    "player_center": list(SYMBOLIC_PLAYER_CENTER),
                },
                "symbolic_evidence": {
                    "format": "compressed NumPy NPZ",
                    "arrays": {
                        "terrain": "uint8 [observation, 7, 9]",
                        "entities": "uint8 [observation, 7, 9]",
                        "inventory": "uint8 [observation, 16]",
                        "facing": "uint8 [observation]",
                        "sleeping": "bool [observation]",
                        "daylight": "float64 [observation]",
                        "observation_indices": "uint32 [observation]",
                    },
                    "inventory_order": list(SYMBOLIC_INVENTORY_KEYS),
                    "facing_ids": {
                        str(index): name
                        for index, name in enumerate(SYMBOLIC_FACING_NAMES)
                    },
                    "lossless": True,
                    "mp4_replays": {"enabled": False, "supported": False},
                },
            }
        )
    if score_profile == LHS_REWARD_PROFILE:
        manifest["reward_semantics"] = {
            "reward": (
                "LHS alive-survival + vital-survival + compressed "
                "first-unlock + cadence-bounded maintenance-repeat + "
                "cadence-bounded productivity-repeat"
            ),
            "episode_return": "sum(reward); Policy failure is formally zero",
            "feedback": (
                "survival-only lower tail plus mean secondary return; "
                "no upper-tail bonus"
            ),
            "upstream_reward": "pinned Crafter 1.8.3 reward; diagnostic only",
            "energy_scored": False,
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
    evidence_summary = (
        "full-frame-sequence-lossless-npz"
        if observation_profile == "rgb"
        else "complete-local-symbolic-sequence-lossless-npz"
    )
    return tuple(artifacts), {
        "schema": (
            "crafter/local-symbolic-feedback-summary/v1"
            if observation_profile == "local-symbolic-v1"
            else (
                "crafter/complete-feedback-summary/v7"
                if score_profile == LHS_REWARD_PROFILE
                else "crafter/complete-feedback-summary/v5"
            )
        ),
        "complete": True,
        "score_profile": score_profile,
        "observation_profile": observation_profile,
        "episodes": len(records),
        "transitions": total_transitions,
        "observations": total_observations,
        "observation_evidence": evidence_summary,
        "visual_evidence": (
            evidence_summary if observation_profile == "rgb" else "none"
        ),
        "mp4_replays_enabled": include_mp4,
        "frame_sampling": "none",
        "pixel_exact": True,
        "bulk_compressed_bytes": bulk_bytes,
        "trajectory_compressed_bytes": trajectory_bytes,
        "observation_compressed_bytes": observation_bytes,
        "replay_compressed_bytes": replay_bytes,
        "trajectory_artifacts": len(trajectory_entries),
        "observation_artifacts": len(observation_entries),
        "replay_artifacts": len(replay_entries),
    }


def _aggregate_only_artifact_summary(
    records: Sequence[EpisodeRecord],
    *,
    score_profile: str,
    reason: str,
    include_mp4: bool,
    observation_profile: ObservationProfile,
) -> dict[str, PolicyValue]:
    return {
        "schema": (
            "crafter/local-symbolic-feedback-summary/v1"
            if observation_profile == "local-symbolic-v1"
            else (
                "crafter/complete-feedback-summary/v7"
                if score_profile == LHS_REWARD_PROFILE
                else "crafter/complete-feedback-summary/v5"
            )
        ),
        "complete": False,
        "score_profile": score_profile,
        "observation_profile": observation_profile,
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
        "observation_compressed_bytes": 0,
        "replay_compressed_bytes": 0,
        "trajectory_artifacts": 0,
        "observation_artifacts": 0,
        "replay_artifacts": 0,
        "mp4_replays_enabled": include_mp4,
    }


def _trajectory_artifact(
    record: EpisodeRecord,
    *,
    episode_index: int,
    score_profile: str = "upstream",
    failure_return: float | None = None,
    observation_profile: ObservationProfile = "rgb",
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
        if observation_profile == "local-symbolic-v1":
            episode_header["observation_profile"] = observation_profile
        if score_profile == LHS_REWARD_PROFILE:
            if failure_return is None:
                raise ValueError(
                    "Crafter LHS trajectory requires failure return"
                )
            failed = record.policy_failure is not None
            scored_return = failure_return if failed else record.total_reward
            episode_header.update(
                {
                    "scored": not failed,
                    "valid_episode": not failed,
                    "included_in_feedback": True,
                    "partial_return": record.total_reward,
                    "scored_return": scored_return,
                    "return": scored_return,
                    "partial_credit_discarded": failed,
                    "reward_profile": LHS_REWARD_PROFILE,
                    "failure": (
                        None
                        if record.policy_failure is None
                        else _policy_failure_detail(
                            record.policy_failure,
                            record.steps,
                        )
                    ),
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
            if score_profile == LHS_REWARD_PROFILE:
                upstream_reward = _transition_upstream_reward(
                    transition.step.metrics
                )
                components = _transition_lhs_score_delta_components(
                    transition.step.metrics
                )
                diagnostics = _transition_lhs_repeat_diagnostics(
                    transition.step.metrics
                )
                vitals = _transition_maintenance_vitals(
                    transition.step.metrics
                )
                if (
                    upstream_reward is None
                    or components is None
                    or diagnostics is None
                    or vitals is None
                ):
                    raise ValueError(
                        "Crafter LHS trajectory metrics are incomplete"
                    )
                transition_line.update(
                    {
                        "upstream_reward": upstream_reward,
                        "maintenance_vitals": vitals,
                        "lhs_score_delta_components": components,
                        "lhs_repeat_diagnostics": diagnostics,
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


def _observation_artifacts(
    record: EpisodeRecord,
    *,
    episode_index: int,
    observation_profile: ObservationProfile = "rgb",
) -> tuple[list[Artifact], list[dict[str, object]]]:
    observations = _observations(record)
    artifacts: list[Artifact] = []
    entries: list[dict[str, object]] = []
    for chunk_index, start in enumerate(
        range(0, len(observations), _OBSERVATION_CHUNK_FRAMES)
    ):
        stop = min(start + _OBSERVATION_CHUNK_FRAMES, len(observations))
        output = io.BytesIO()
        selected = observations[start:stop]
        if observation_profile == "rgb":
            np.savez_compressed(
                output,
                observations=np.stack(
                    tuple(_observation_array(item) for item in selected)
                ),
                observation_indices=np.arange(start, stop, dtype=np.uint32),
            )
        else:
            decoded = tuple(symbolic_observation_arrays(item) for item in selected)
            np.savez_compressed(
                output,
                terrain=np.stack(tuple(item[0] for item in decoded)),
                entities=np.stack(tuple(item[1] for item in decoded)),
                inventory=np.stack(tuple(item[2] for item in decoded)),
                facing=np.asarray(tuple(item[3] for item in decoded), dtype=np.uint8),
                sleeping=np.asarray(tuple(item[4] for item in decoded), dtype=np.bool_),
                daylight=np.asarray(tuple(item[5] for item in decoded), dtype=np.float64),
                observation_indices=np.arange(start, stop, dtype=np.uint32),
            )
        name = (
            f"observations/episode-{episode_index:06d}/"
            f"observations-{chunk_index:06d}.npz"
        )
        artifact = Artifact(
            name=name,
            media_type="application/x-npz",
            content=output.getvalue(),
            retention="bulk",
        )
        artifacts.append(artifact)
        entries.append(
            {
                "episode_index": episode_index,
                "artifact": name,
                "chunk_index": chunk_index,
                "observations": stop - start,
                "first_observation_index": start,
                "last_observation_index": stop - 1,
                "compressed_bytes": artifact.size,
            }
        )
    return artifacts, entries


def _mp4_replay_artifact(
    record: EpisodeRecord,
    *,
    episode_index: int,
    cache: dict[tuple[bytes, ...], bytes],
) -> tuple[Artifact, dict[str, object]]:
    observations = _observations(record)
    cache_key = None
    if len(observations) <= 4:
        cache_key = tuple(
            _observation_array(observation).tobytes()
            for observation in observations
        )
    content = None if cache_key is None else cache.get(cache_key)
    if content is None:
        content = _encode_mp4_replay(observations)
        if cache_key is not None:
            cache[cache_key] = content
    name = f"replays/episode-{episode_index:06d}/replay.mp4"
    artifact = Artifact(
        name=name,
        media_type="video/mp4",
        content=content,
        retention="bulk",
    )
    return artifact, {
        "episode_index": episode_index,
        "artifact": name,
        "video_frames": len(observations),
        "first_observation_index": 0,
        "last_observation_index": len(observations) - 1,
        "frames_per_second": _MP4_REPLAY_FPS,
        "frame_size": [_MP4_REPLAY_SIZE, _MP4_REPLAY_SIZE],
        "target_bitrate": _MP4_REPLAY_BITRATE,
        "codec": "h264",
        "audio": False,
        "compressed_bytes": artifact.size,
    }


def _encode_mp4_replay(observations: Sequence[PolicyValue]) -> bytes:
    with tempfile.TemporaryDirectory(
        prefix="evopolicygym-crafter-replay-"
    ) as temporary:
        path = Path(temporary) / "replay.mp4"
        writer = imageio_ffmpeg.write_frames(
            path,
            (_MP4_REPLAY_SIZE, _MP4_REPLAY_SIZE),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=_MP4_REPLAY_FPS,
            bitrate=_MP4_REPLAY_BITRATE,
            codec="libx264",
            macro_block_size=16,
            ffmpeg_log_level="error",
            output_params=[
                "-maxrate",
                "112k",
                "-bufsize",
                "224k",
                "-an",
                "-movflags",
                "+faststart",
            ],
        )
        writer.send(None)
        try:
            for observation in observations:
                frame = _observation_array(observation)
                resized = np.asarray(
                    Image.fromarray(frame).resize(
                        (_MP4_REPLAY_SIZE, _MP4_REPLAY_SIZE),
                        resample=Image.Resampling.NEAREST,
                    ),
                    dtype=np.uint8,
                )
                writer.send(np.ascontiguousarray(resized))
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
    "CrafterLongHorizonSurvivalBenchmark",
]
