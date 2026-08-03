"""A parameterized InvertedPendulum-v5 Benchmark with public traces."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import InvertedPendulumConfig
from .environment import InvertedPendulumEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-inverted-pendulum/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 1_000
_FAILURE_RETURN = -1_000.0
_MODEL_TIMESTEP_SECONDS = 0.02
_ACTUATOR_GEAR = 100.0
_TERMINATION_ANGLE_RADIANS = 0.2
_OBSERVATION_FIELDS = (
    "cart_position",
    "pole_angle",
    "cart_velocity",
    "pole_angular_velocity",
)
_METRIC_FIELDS = frozenset(
    {
        "step_count",
        "remaining_steps",
        "seconds_per_step",
        "simulated_seconds",
        "requested_cart_control",
        "actuator_gear_scaled_cart_force",
        "cumulative_absolute_action",
        "cart_position",
        "minimum_cart_position",
        "maximum_cart_position",
        "cart_velocity",
        "maximum_absolute_cart_velocity",
        "pole_angle_radians",
        "pole_angle_degrees",
        "maximum_absolute_pole_angle_radians",
        "pole_angle_termination_threshold_radians",
        "pole_angle_margin_radians",
        "minimum_pole_angle_margin_radians",
        "pole_angular_velocity",
        "maximum_absolute_pole_angular_velocity",
        "healthy",
        "reward_survive",
        "cumulative_reward_survive",
        "cumulative_return",
        "terminal_reason",
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    final_cart_position: float
    minimum_cart_position: float
    maximum_cart_position: float
    maximum_absolute_cart_velocity: float
    final_pole_angle_radians: float
    maximum_absolute_pole_angle_radians: float
    minimum_pole_angle_margin_radians: float
    maximum_absolute_pole_angular_velocity: float
    mean_absolute_action: float
    outcome: str


class InvertedPendulumBenchmark:
    """Mean survival return over deterministic Episode plans."""

    def __init__(
        self,
        config: InvertedPendulumConfig | None = None,
    ) -> None:
        if config is None:
            config = InvertedPendulumConfig()
        if type(config) is not InvertedPendulumConfig:
            raise TypeError("config must be InvertedPendulumConfig")
        self._config = config
        self._spec = _benchmark_spec(config)

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
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return InvertedPendulumEnvironment(
            episode,
            config=self._config,
        )

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(
            (record.total_reward if record.policy_failure is None else _FAILURE_RETURN)
            for record in records
        )
        score = statistics.fmean(returns)
        diagnostics = tuple(
            _episode_diagnostics(record)
            for record in records
            if record.policy_failure is None and record.transitions
        )
        outcomes = tuple(_episode_outcome(record) for record in records)
        balanced = outcomes.count("time_limit")
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{balanced} balanced for all 1000 steps."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "mean_final_cart_position": _mean_or_none(
                    tuple(item.final_cart_position for item in diagnostics)
                ),
                "mean_episode_minimum_cart_position": _mean_or_none(
                    tuple(item.minimum_cart_position for item in diagnostics)
                ),
                "mean_episode_maximum_cart_position": _mean_or_none(
                    tuple(item.maximum_cart_position for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_cart_velocity": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_cart_velocity for item in diagnostics)
                    )
                ),
                "mean_final_pole_angle_radians": _mean_or_none(
                    tuple(item.final_pole_angle_radians for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_pole_angle_radians": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_pole_angle_radians for item in diagnostics)
                    )
                ),
                "mean_episode_minimum_pole_angle_margin_radians": (
                    _mean_or_none(
                        tuple(item.minimum_pole_angle_margin_radians for item in diagnostics)
                    )
                ),
                "mean_episode_maximum_absolute_pole_angular_velocity": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_pole_angular_velocity for item in diagnostics)
                    )
                ),
                "mean_absolute_action": _mean_or_none(
                    tuple(item.mean_absolute_action for item in diagnostics)
                ),
                "episodes": len(records),
                "full_horizon_balances": balanced,
                "fallen_episodes": outcomes.count("fallen"),
                "fallen_at_time_limit_episodes": outcomes.count("fallen_and_time_limit"),
                "incomplete_episodes": outcomes.count("incomplete"),
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(_trace_artifact(traced),),
        )


def _benchmark_spec(config: InvertedPendulumConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="gymnasium/InvertedPendulum-v5/mean-return-v1",
        description=(
            "Apply one cart actuator control in [-3, 3], scaled by MuJoCo "
            "gear 100, to keep the pole angle within 0.2 radians of upright. "
            "A step earns one point exactly when it does not terminate."
        ),
        observation_space={
            "type": "object",
            "policy_carrier": "dict[str, float]",
            "source_dtype": "float64",
            "fields": {
                "cart_position": {
                    "type": "float",
                    "unit": "meters",
                    "meaning": "Cart slider qpos; positive is rightward.",
                },
                "pole_angle": {
                    "type": "float",
                    "unit": "radians",
                    "meaning": (
                        "Pole hinge qpos relative to upright; the Episode "
                        "terminates when abs(angle) > 0.2."
                    ),
                },
                "cart_velocity": {
                    "type": "float",
                    "unit": "meters_per_second",
                    "meaning": "Unclipped cart slider qvel.",
                },
                "pole_angular_velocity": {
                    "type": "float",
                    "unit": "radians_per_second",
                    "meaning": "Unclipped pole hinge qvel.",
                },
            },
        },
        action_space={
            "type": "array",
            "shape": [1],
            "items": {
                "type": "float",
                "minimum": -3.0,
                "maximum": 3.0,
            },
            "policy_carrier": "list[float]",
            "components": ["cart_control"],
            "actuator_gears": [_ACTUATOR_GEAR],
            "meaning": (
                "The requested control is multiplied by gear 100 for the "
                "slider actuator; it is not itself a force in newtons."
            ),
        },
        metadata={
            "environment": "InvertedPendulum-v5",
            "provider": "Gymnasium",
            "reward_threshold": 950.0,
            "maximum_return": 1000.0,
            "healthy_angle_limit": 0.2,
            "official_model": "inverted_pendulum.xml",
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": _MODEL_TIMESTEP_SECONDS,
            "seconds_per_step": (_MODEL_TIMESTEP_SECONDS * config.frame_skip),
            "actuator_gear": _ACTUATOR_GEAR,
            "termination_angle_radians": _TERMINATION_ANGLE_RADIANS,
            "termination_rule": "abs(pole_angle) > 0.2",
            "reward_formula": "1.0 if not terminated else 0.0",
            "observation_clipping": "none",
            "reset_noise_scale": config.reset_noise_scale,
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
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


def _episode_outcome(record: EpisodeRecord) -> str:
    if record.policy_failure is not None:
        return "policy_failure"
    if not record.transitions:
        return "incomplete"
    metrics = _trace_metrics(record.transitions[-1].step.metrics)
    reason = metrics["terminal_reason"]
    if type(reason) is not str:
        raise ValueError("InvertedPendulum terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("InvertedPendulum diagnostics require a transition")
    final = _trace_metrics(record.transitions[-1].step.metrics)
    return _EpisodeDiagnostics(
        final_cart_position=_float_metric(final, "cart_position"),
        minimum_cart_position=_float_metric(final, "minimum_cart_position"),
        maximum_cart_position=_float_metric(final, "maximum_cart_position"),
        maximum_absolute_cart_velocity=_float_metric(
            final,
            "maximum_absolute_cart_velocity",
        ),
        final_pole_angle_radians=_float_metric(
            final,
            "pole_angle_radians",
        ),
        maximum_absolute_pole_angle_radians=_float_metric(
            final,
            "maximum_absolute_pole_angle_radians",
        ),
        minimum_pole_angle_margin_radians=_float_metric(
            final,
            "minimum_pole_angle_margin_radians",
        ),
        maximum_absolute_pole_angular_velocity=_float_metric(
            final,
            "maximum_absolute_pole_angular_velocity",
        ),
        mean_absolute_action=statistics.fmean(
            abs(_single_action(transition.action)) for transition in record.transitions
        ),
        outcome=_episode_outcome(record),
    )


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"InvertedPendulum metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _trace_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        diagnostics = (
            _episode_diagnostics(record)
            if record.policy_failure is None and record.transitions
            else None
        )
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "return": (
                        record.total_reward if record.policy_failure is None else _FAILURE_RETURN
                    ),
                    "outcome": _episode_outcome(record),
                    "balanced_full_horizon": (
                        diagnostics is not None and diagnostics.outcome == "time_limit"
                    ),
                    "final_cart_position": (
                        diagnostics.final_cart_position if diagnostics is not None else None
                    ),
                    "final_pole_angle_radians": (
                        diagnostics.final_pole_angle_radians if diagnostics is not None else None
                    ),
                    "maximum_absolute_pole_angle_radians": (
                        diagnostics.maximum_absolute_pole_angle_radians
                        if diagnostics is not None
                        else None
                    ),
                    "minimum_pole_angle_margin_radians": (
                        diagnostics.minimum_pole_angle_margin_radians
                        if diagnostics is not None
                        else None
                    ),
                    "mean_absolute_action": (
                        diagnostics.mean_absolute_action if diagnostics is not None else None
                    ),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(transition.action)
            next_observation = _trace_observation(transition.step.observation)
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "action_components": {
                            "cart_control": action[0],
                        },
                        "reward": transition.step.reward,
                        "metrics": _trace_metrics(transition.step.metrics),
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
            observation = next_observation
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _single_action(action: PolicyValue) -> float:
    return _trace_action(action)[0]


def _trace_action(action: PolicyValue) -> list[float]:
    if type(action) is not list or len(action) != 1:
        raise ValueError("InvertedPendulum trace Action is invalid")
    value = action[0]
    if type(value) is not float or not math.isfinite(value) or not -3.0 <= value <= 3.0:
        raise ValueError("InvertedPendulum trace Action is invalid")
    return [value]


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, float]:
    if type(observation) is not dict:
        raise ValueError("InvertedPendulum trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("InvertedPendulum trace observation is invalid")
    traced: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("InvertedPendulum trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("InvertedPendulum trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("InvertedPendulum trace metrics are invalid")
        elif key == "healthy":
            if type(value) is not bool:
                raise ValueError("InvertedPendulum trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("InvertedPendulum trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("InvertedPendulum trace metrics are invalid")
        traced[key] = value
    return traced


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


__all__ = ["InvertedPendulumBenchmark"]
