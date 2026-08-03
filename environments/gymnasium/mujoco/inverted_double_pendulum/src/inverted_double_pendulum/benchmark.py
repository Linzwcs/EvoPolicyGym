"""Parameterized InvertedDoublePendulum-v5 Benchmark and traces."""

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

from .config import InvertedDoublePendulumConfig
from .environment import InvertedDoublePendulumEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-inverted-double-pendulum/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 1_000
_OBSERVATION_FIELDS = (
    "cart_position",
    "pole1_sin",
    "pole2_relative_sin",
    "pole1_cos",
    "pole2_relative_cos",
    "cart_velocity",
    "pole1_angular_velocity",
    "pole2_relative_angular_velocity",
    "cart_constraint_force",
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
        "pole1_angle_radians",
        "pole2_relative_angle_radians",
        "pole2_absolute_angle_radians",
        "maximum_absolute_pole1_angle_radians",
        "maximum_absolute_pole2_absolute_angle_radians",
        "pole1_angular_velocity_observed",
        "pole2_relative_angular_velocity_observed",
        "maximum_observed_absolute_pole_angular_velocity",
        "velocity_observation_at_clip_limit",
        "velocity_clip_limit_step_fraction",
        "tip_x_position",
        "tip_y_position",
        "tip_x_position_from_observation",
        "tip_y_position_from_observation",
        "tip_position_reconstruction_error",
        "minimum_tip_y_position",
        "maximum_tip_y_position",
        "tip_height_termination_threshold",
        "tip_height_margin",
        "minimum_tip_height_margin",
        "maximum_absolute_tip_x_position",
        "reward_target_tip_height",
        "maximum_physical_tip_height",
        "unavoidable_upright_distance_penalty",
        "pole1_unit_circle_error",
        "pole2_relative_unit_circle_error",
        "reward_survive",
        "reward_distance_penalty",
        "reward_velocity_penalty",
        "reward_from_public_terms",
        "cumulative_reward_survive",
        "cumulative_reward_distance_penalty",
        "cumulative_reward_velocity_penalty",
        "cumulative_return",
        "terminal_reason",
    }
)


@dataclass(frozen=True)
class _EpisodeDiagnostics:
    final_tip_y_position: float
    minimum_tip_y_position: float
    maximum_tip_y_position: float
    minimum_tip_height_margin: float
    maximum_absolute_tip_x_position: float
    maximum_tip_position_reconstruction_error: float
    maximum_absolute_pole1_angle_radians: float
    maximum_absolute_pole2_absolute_angle_radians: float
    maximum_observed_absolute_pole_angular_velocity: float
    velocity_clip_limit_step_fraction: float
    minimum_cart_position: float
    maximum_cart_position: float
    mean_absolute_action: float
    cumulative_reward_survive: float
    cumulative_reward_distance_penalty: float
    cumulative_reward_velocity_penalty: float
    outcome: str


class InvertedDoublePendulumBenchmark:
    """Mean double-pendulum return over deterministic Episode plans."""

    def __init__(
        self,
        config: InvertedDoublePendulumConfig | None = None,
    ) -> None:
        if config is None:
            config = InvertedDoublePendulumConfig()
        if type(config) is not InvertedDoublePendulumConfig:
            raise TypeError("config must be InvertedDoublePendulumConfig")
        self._config = config
        self._failure_return = _failure_return(config)
        self._spec = _benchmark_spec(
            config,
            failure_return=self._failure_return,
        )

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
        return InvertedDoublePendulumEnvironment(
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
            record.total_reward if record.policy_failure is None else self._failure_return
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
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{balanced} balanced for all 1000 steps; "
                    f"{outcomes.count('fallen')} fell early."
                ),
                "mean_return": score,
                "mean_steps": statistics.fmean(record.steps for record in records),
                "mean_final_tip_y_position": _mean_or_none(
                    tuple(item.final_tip_y_position for item in diagnostics)
                ),
                "mean_episode_minimum_tip_y_position": _mean_or_none(
                    tuple(item.minimum_tip_y_position for item in diagnostics)
                ),
                "mean_episode_maximum_tip_y_position": _mean_or_none(
                    tuple(item.maximum_tip_y_position for item in diagnostics)
                ),
                "mean_episode_minimum_tip_height_margin": _mean_or_none(
                    tuple(item.minimum_tip_height_margin for item in diagnostics)
                ),
                "mean_episode_maximum_absolute_tip_x_position": _mean_or_none(
                    tuple(item.maximum_absolute_tip_x_position for item in diagnostics)
                ),
                "mean_episode_maximum_tip_position_reconstruction_error": (
                    _mean_or_none(
                        tuple(
                            item.maximum_tip_position_reconstruction_error for item in diagnostics
                        )
                    )
                ),
                "mean_episode_maximum_absolute_pole1_angle_radians": (
                    _mean_or_none(
                        tuple(item.maximum_absolute_pole1_angle_radians for item in diagnostics)
                    )
                ),
                "mean_episode_maximum_absolute_pole2_angle_radians": (
                    _mean_or_none(
                        tuple(
                            item.maximum_absolute_pole2_absolute_angle_radians
                            for item in diagnostics
                        )
                    )
                ),
                "mean_episode_maximum_observed_absolute_angular_velocity": (
                    _mean_or_none(
                        tuple(
                            item.maximum_observed_absolute_pole_angular_velocity
                            for item in diagnostics
                        )
                    )
                ),
                "mean_velocity_clip_limit_step_fraction": _mean_or_none(
                    tuple(item.velocity_clip_limit_step_fraction for item in diagnostics)
                ),
                "mean_episode_minimum_cart_position": _mean_or_none(
                    tuple(item.minimum_cart_position for item in diagnostics)
                ),
                "mean_episode_maximum_cart_position": _mean_or_none(
                    tuple(item.maximum_cart_position for item in diagnostics)
                ),
                "mean_absolute_action": _mean_or_none(
                    tuple(item.mean_absolute_action for item in diagnostics)
                ),
                "mean_episode_survival_reward": _mean_or_none(
                    tuple(item.cumulative_reward_survive for item in diagnostics)
                ),
                "mean_episode_distance_penalty_reward": _mean_or_none(
                    tuple(item.cumulative_reward_distance_penalty for item in diagnostics)
                ),
                "mean_episode_velocity_penalty_reward": _mean_or_none(
                    tuple(item.cumulative_reward_velocity_penalty for item in diagnostics)
                ),
                "full_horizon_balances": balanced,
                "fallen_episodes": outcomes.count("fallen"),
                "fallen_at_time_limit_episodes": outcomes.count("fallen_and_time_limit"),
                "incomplete_episodes": outcomes.count("incomplete"),
                "episodes": len(records),
                "policy_failures": sum(record.policy_failure is not None for record in records),
                "failure_return": self._failure_return,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    failure_return=self._failure_return,
                ),
            ),
        )


def _benchmark_spec(
    config: InvertedDoublePendulumConfig,
    *,
    failure_return: float,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="gymnasium/InvertedDoublePendulum-v5/mean-return-v1",
        description=(
            "Apply one bounded cart control, scaled by actuator gear 500, to "
            "balance two serial 0.6-meter poles. The second hinge angle is "
            "relative to pole 1. Reward is healthy_reward minus a tip-distance "
            "penalty targeting y=2.0 meters and squared angular-velocity "
            "penalties. The physical maximum tip height is 1.2 meters, so a "
            "perfect upright pose still receives -0.64 distance reward. The "
            "Episode terminates when tip y <= 1.0 meter."
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
                "pole1_sin": _trig_field("Sine of pole 1 angle relative to upright."),
                "pole2_relative_sin": _trig_field("Sine of pole 2 hinge angle relative to pole 1."),
                "pole1_cos": _trig_field("Cosine of pole 1 angle relative to upright."),
                "pole2_relative_cos": _trig_field(
                    "Cosine of pole 2 hinge angle relative to pole 1."
                ),
                "cart_velocity": _velocity_field("Clipped cart slider velocity."),
                "pole1_angular_velocity": _angular_velocity_field(
                    "Clipped pole 1 hinge angular velocity."
                ),
                "pole2_relative_angular_velocity": _angular_velocity_field(
                    "Clipped pole 2 hinge angular velocity relative to pole 1."
                ),
                "cart_constraint_force": {
                    "type": "float",
                    "unit": "generalized_force",
                    "minimum": -10.0,
                    "maximum": 10.0,
                    "meaning": "Clipped slider qfrc_constraint component.",
                },
            },
        },
        action_space={
            "type": "array",
            "shape": [1],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "policy_carrier": "list[float]",
            "components": ["cart_control"],
            "actuator_gears": [_ACTUATOR_GEAR],
            "meaning": ("The requested control is multiplied by gear 500 for the slider actuator."),
        },
        metadata={
            "environment": "InvertedDoublePendulum-v5",
            "provider": "Gymnasium",
            "reward_threshold": 9100.0,
            "unhealthy_tip_height": 1.0,
            "official_model": "inverted_double_pendulum.xml",
            "failure_return": failure_return,
        },
        environment_parameters={
            "frame_skip": config.frame_skip,
            "model_timestep_seconds": 0.01,
            "seconds_per_step": 0.01 * config.frame_skip,
            "actuator_gear": _ACTUATOR_GEAR,
            "pole_length_meters": _POLE_LENGTH_METERS,
            "maximum_physical_tip_height": 1.2,
            "reward_target_tip_height": 2.0,
            "termination_tip_height": 1.0,
            "healthy_reward": config.healthy_reward,
            "reward_formula": (
                "healthy_reward_if_tip_y>1-(0.01*tip_x^2+(tip_y-2)^2)-"
                "(0.001*pole1_qvel^2+0.005*pole2_relative_qvel^2)"
            ),
            "observation_velocity_clipping": [-10.0, 10.0],
            "reward_tip_position_source": "site_xpos",
            "observation_tip_reconstruction": (
                "qpos trigonometry; may differ slightly from site_xpos due to "
                "MuJoCo derived-geometry update timing"
            ),
            "reset_noise_scale": config.reset_noise_scale,
            "time_limit": _MAX_EPISODE_STEPS,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


_ACTUATOR_GEAR = 500.0
_POLE_LENGTH_METERS = 0.6


def _trig_field(meaning: str) -> dict[str, PolicyValue]:
    return {
        "type": "float",
        "minimum": -1.0,
        "maximum": 1.0,
        "meaning": meaning,
    }


def _velocity_field(meaning: str) -> dict[str, PolicyValue]:
    return {
        "type": "float",
        "unit": "meters_per_second",
        "minimum": -10.0,
        "maximum": 10.0,
        "meaning": meaning,
    }


def _angular_velocity_field(meaning: str) -> dict[str, PolicyValue]:
    return {
        "type": "float",
        "unit": "radians_per_second",
        "minimum": -10.0,
        "maximum": 10.0,
        "meaning": meaning,
    }


def _failure_return(config: InvertedDoublePendulumConfig) -> float:
    return -1_000.0 * max(1.0, config.healthy_reward / 10.0)


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
        raise ValueError("InvertedDoublePendulum terminal reason is invalid")
    return reason if reason != "none" else "incomplete"


def _episode_diagnostics(record: EpisodeRecord) -> _EpisodeDiagnostics:
    if not record.transitions:
        raise ValueError("InvertedDoublePendulum diagnostics require a transition")
    metrics = tuple(_trace_metrics(transition.step.metrics) for transition in record.transitions)
    final = metrics[-1]
    return _EpisodeDiagnostics(
        final_tip_y_position=_float_metric(final, "tip_y_position"),
        minimum_tip_y_position=_float_metric(final, "minimum_tip_y_position"),
        maximum_tip_y_position=_float_metric(final, "maximum_tip_y_position"),
        minimum_tip_height_margin=_float_metric(
            final,
            "minimum_tip_height_margin",
        ),
        maximum_absolute_tip_x_position=_float_metric(
            final,
            "maximum_absolute_tip_x_position",
        ),
        maximum_tip_position_reconstruction_error=max(
            _float_metric(item, "tip_position_reconstruction_error") for item in metrics
        ),
        maximum_absolute_pole1_angle_radians=_float_metric(
            final,
            "maximum_absolute_pole1_angle_radians",
        ),
        maximum_absolute_pole2_absolute_angle_radians=_float_metric(
            final,
            "maximum_absolute_pole2_absolute_angle_radians",
        ),
        maximum_observed_absolute_pole_angular_velocity=_float_metric(
            final,
            "maximum_observed_absolute_pole_angular_velocity",
        ),
        velocity_clip_limit_step_fraction=_float_metric(
            final,
            "velocity_clip_limit_step_fraction",
        ),
        minimum_cart_position=_float_metric(final, "minimum_cart_position"),
        maximum_cart_position=_float_metric(final, "maximum_cart_position"),
        mean_absolute_action=statistics.fmean(
            abs(_single_action(transition.action)) for transition in record.transitions
        ),
        cumulative_reward_survive=_float_metric(
            final,
            "cumulative_reward_survive",
        ),
        cumulative_reward_distance_penalty=_float_metric(
            final,
            "cumulative_reward_distance_penalty",
        ),
        cumulative_reward_velocity_penalty=_float_metric(
            final,
            "cumulative_reward_velocity_penalty",
        ),
        outcome=_episode_outcome(record),
    )


def _float_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float:
        raise ValueError(f"InvertedDoublePendulum metric {name} is invalid")
    return value


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    failure_return: float,
) -> Artifact:
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
                        record.total_reward if record.policy_failure is None else failure_return
                    ),
                    "outcome": _episode_outcome(record),
                    "balanced_full_horizon": (
                        diagnostics is not None and diagnostics.outcome == "time_limit"
                    ),
                    "final_tip_y_position": (
                        diagnostics.final_tip_y_position if diagnostics is not None else None
                    ),
                    "minimum_tip_y_position": (
                        diagnostics.minimum_tip_y_position if diagnostics is not None else None
                    ),
                    "minimum_tip_height_margin": (
                        diagnostics.minimum_tip_height_margin if diagnostics is not None else None
                    ),
                    "maximum_absolute_tip_x_position": (
                        diagnostics.maximum_absolute_tip_x_position
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
                        "action_components": {"cart_control": action[0]},
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
        raise ValueError("InvertedDoublePendulum trace Action is invalid")
    value = action[0]
    if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("InvertedDoublePendulum trace Action is invalid")
    return [value]


def _trace_observation(observation: PolicyValue) -> dict[str, float]:
    if type(observation) is not dict or set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("InvertedDoublePendulum trace observation is invalid")
    traced: dict[str, float] = {}
    for key in _OBSERVATION_FIELDS:
        value = observation[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("InvertedDoublePendulum trace observation is invalid")
        traced[key] = value
    return traced


def _trace_metrics(metrics: PolicyValue) -> dict[str, object]:
    if type(metrics) is not dict or set(metrics) != set(_METRIC_FIELDS):
        raise ValueError("InvertedDoublePendulum trace metrics are invalid")
    traced: dict[str, object] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if key in {"step_count", "remaining_steps"}:
            if type(value) is not int:
                raise ValueError("InvertedDoublePendulum trace metrics are invalid")
        elif key == "velocity_observation_at_clip_limit":
            if type(value) is not bool:
                raise ValueError("InvertedDoublePendulum trace metrics are invalid")
        elif key == "terminal_reason":
            if type(value) is not str:
                raise ValueError("InvertedDoublePendulum trace metrics are invalid")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError("InvertedDoublePendulum trace metrics are invalid")
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


__all__ = ["InvertedDoublePendulumBenchmark"]
