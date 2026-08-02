"""Deterministic NLE NetHackScore Benchmark and bounded public Feedback."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Sequence

from evopolicygym.authoring import (
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import NetHackConfig
from .constants import (
    ACTION_SPACE,
    AGGREGATE_FEEDBACK_SCOPE,
    BENCHMARK_ID,
    CHARACTER,
    FEEDBACK_SCOPE_KEY,
    NETHACK_OPTIONS,
    NETHACK_VERSION,
    PENALTY_MODE,
    PENALTY_STEP,
    PENALTY_TIME,
    PUBLIC_FEEDBACK_SCOPE,
    UPSTREAM_VERSION,
)
from .environment import NetHackEnvironment
from .evidence import MAX_PUBLIC_FEEDBACK_EPISODES, complete_feedback_artifacts

_EPISODE_SEED_DOMAIN = b"evopolicygym-nle-nethack/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})


class NetHackBenchmark:
    """Mean shaped return on deterministic NLE NetHackScore Episodes."""

    def __init__(self, config: NetHackConfig | None = None) -> None:
        selected = NetHackConfig() if config is None else config
        if type(selected) is not NetHackConfig:
            raise TypeError("config must be NetHackConfig or None")
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
        feedback_scope = (
            PUBLIC_FEEDBACK_SCOPE
            if split == "train"
            else AGGREGATE_FEEDBACK_SCOPE
        )
        return tuple(
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
                scenario={FEEDBACK_SCOPE_KEY: feedback_scope},
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return NetHackEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(_scored_return(record, self._config) for record in records)
        game_scores = tuple(_game_score(record) for record in records)
        depths = tuple(_max_int_metric(record, "max_depth") for record in records)
        frozen_steps = tuple(_frozen_steps(record) for record in records)
        frozen_step_total = sum(frozen_steps)
        mean_frozen_steps = statistics.fmean(frozen_steps)
        total_steps = sum(record.steps for record in records)
        score = statistics.fmean(returns)
        feedback_scope = _feedback_scope(records)
        if feedback_scope == PUBLIC_FEEDBACK_SCOPE:
            artifacts, artifact_summary = complete_feedback_artifacts(records)
        else:
            artifacts = ()
            artifact_summary = {
                "schema": "nle/aggregate-only-host-phase/v1",
                "complete": False,
                "reason": "validation and assessment retain aggregate results only",
            }
        return Feedback(
            score=score,
            content={
                "mean_return": score,
                "median_return": statistics.median(returns),
                "max_return": max(returns),
                "mean_game_score": statistics.fmean(game_scores),
                "max_game_score": max(game_scores),
                "mean_max_depth": statistics.fmean(depths),
                "max_depth": max(depths),
                "mean_steps": statistics.fmean(record.steps for record in records),
                "frozen_steps": frozen_step_total,
                "mean_frozen_steps": mean_frozen_steps,
                "frozen_step_fraction": (
                    frozen_step_total / total_steps if total_steps else 0.0
                ),
                "mean_frozen_penalty": (
                    PENALTY_STEP * mean_frozen_steps
                    if mean_frozen_steps
                    else 0.0
                ),
                "episodes": len(records),
                "deaths": sum(_died(record) for record in records),
                "ascensions": sum(_ascended(record) for record in records),
                "truncated_episodes": sum(_truncated(record) for record in records),
                "policy_failures": sum(
                    record.policy_failure is not None for record in records
                ),
                "failure_return": -float(self._config.max_episode_steps),
                "detailed_feedback": artifact_summary,
            },
            artifacts=artifacts,
        )


def _spec(config: NetHackConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=BENCHMARK_ID,
        description=(
            "Play the deterministic NLE NetHackScore task from public map, "
            "status, message, and inventory observations. Maximize mean return."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "screen": {
                    "type": "object",
                    "fields": {
                        "glyphs": {
                            "type": "tensor",
                            "dtype": "int16",
                            "shape": [21, 79],
                        },
                        "chars": {
                            "type": "tensor",
                            "dtype": "uint8",
                            "shape": [21, 79],
                            "encoding": "latin-1 code points",
                        },
                        "colors": {
                            "type": "tensor",
                            "dtype": "uint8",
                            "shape": [21, 79],
                        },
                    },
                },
                "stats": {"type": "object", "schema": "nle-blstats-named-v1"},
                "message": {"type": "string"},
                "inventory": {"type": "list", "items": "nle-inventory-entry-v1"},
                "input_mode": {
                    "type": "enum",
                    "values": ["normal", "yes_no", "get_line", "more"],
                },
            },
        },
        action_space=ACTION_SPACE,
        metadata={
            "environment": "NetHackScore-v0",
            "provider": "NLE",
            "upstream_version": UPSTREAM_VERSION,
            "nethack_version": NETHACK_VERSION,
            "upstream_license": "NetHack General Public License",
            "failure_return": -float(config.max_episode_steps),
            "partial_observability": True,
        },
        environment_parameters={
            "task": "NetHackScore-v0",
            "upstream_version": UPSTREAM_VERSION,
            "nethack_version": NETHACK_VERSION,
            "character": CHARACTER,
            "action_profile": "nle-task-actions-v1",
            "observation_profile": "semantic-core-v1",
            "max_episode_steps": config.max_episode_steps,
            "penalty_mode": PENALTY_MODE,
            "penalty_step": PENALTY_STEP,
            "penalty_time": PENALTY_TIME,
            "options": list(NETHACK_OPTIONS),
            "spawn_monsters": True,
            "anti_tas_reseed": False,
            "independent_level_seed": True,
            "deterministic_time_effects": True,
            "ttyrec": False,
            "visualization_generated": False,
            "public_training_feedback": "complete-policy-visible-trajectory-v1",
            "max_public_feedback_episodes": MAX_PUBLIC_FEEDBACK_EPISODES,
        },
        max_episode_steps=config.max_episode_steps,
        primary_metric="mean_return",
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


def _feedback_scope(records: Sequence[EpisodeRecord]) -> str:
    scopes = {_episode_feedback_scope(record.episode) for record in records}
    if len(scopes) != 1:
        raise ValueError("NLE Feedback Episodes mix incompatible phases")
    return scopes.pop()


def _episode_feedback_scope(episode: EpisodeSpec) -> str:
    scenario = episode.scenario
    if scenario is None:
        return PUBLIC_FEEDBACK_SCOPE
    if type(scenario) is not dict or set(scenario) != {FEEDBACK_SCOPE_KEY}:
        raise ValueError("NLE Episode scenario is invalid")
    scope = scenario[FEEDBACK_SCOPE_KEY]
    if type(scope) is not str or scope not in {
        PUBLIC_FEEDBACK_SCOPE,
        AGGREGATE_FEEDBACK_SCOPE,
    }:
        raise ValueError("NLE Episode Feedback scope is invalid")
    return scope


def _scored_return(record: EpisodeRecord, config: NetHackConfig) -> float:
    if record.policy_failure is not None:
        return -float(config.max_episode_steps)
    return record.total_reward


def _game_score(record: EpisodeRecord) -> int:
    if record.policy_failure is not None:
        return 0
    return _max_int_metric(record, "max_game_score")


def _max_int_metric(record: EpisodeRecord, name: str) -> int:
    values: list[int] = []
    for transition in record.transitions:
        metrics = transition.step.metrics
        if type(metrics) is not dict:
            raise ValueError("NLE transition metrics are invalid")
        value = metrics.get(name)
        if type(value) is not int:
            raise ValueError(f"NLE transition metric {name} is invalid")
        values.append(value)
    return max(values, default=0)


def _last_metrics(record: EpisodeRecord) -> dict[str, PolicyValue] | None:
    if not record.transitions:
        return None
    value = record.transitions[-1].step.metrics
    if type(value) is not dict:
        raise ValueError("NLE transition metrics are invalid")
    return value


def _ascended(record: EpisodeRecord) -> bool:
    metrics = _last_metrics(record)
    if metrics is None or record.policy_failure is not None:
        return False
    value = metrics.get("ascended")
    if type(value) is not bool:
        raise ValueError("NLE ascended metric is invalid")
    return value


def _died(record: EpisodeRecord) -> bool:
    metrics = _last_metrics(record)
    if metrics is None or record.policy_failure is not None:
        return False
    value = metrics.get("end_status")
    if type(value) is not int:
        raise ValueError("NLE end_status metric is invalid")
    return value == 1 and not _ascended(record)


def _truncated(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.truncated
    )


def _frozen_steps(record: EpisodeRecord) -> int:
    stats = record.initial_observation
    if type(stats) is not dict:
        raise ValueError("NLE initial observation is invalid")
    initial_stats = stats.get("stats")
    if type(initial_stats) is not dict:
        raise ValueError("NLE initial stats are invalid")
    previous_turn = initial_stats.get("turn")
    if type(previous_turn) is not int:
        raise ValueError("NLE initial turn is invalid")

    frozen = 0
    for transition in record.transitions:
        metrics = transition.step.metrics
        if type(metrics) is not dict:
            raise ValueError("NLE transition metrics are invalid")
        turn = metrics.get("turn")
        if type(turn) is not int:
            raise ValueError("NLE transition turn is invalid")
        if turn == previous_turn:
            frozen += 1
        previous_turn = turn
    return frozen


__all__ = ["NetHackBenchmark"]
