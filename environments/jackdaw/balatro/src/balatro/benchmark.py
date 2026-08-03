"""A deterministic Balatro Benchmark with semantic public replays."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import BalatroConfig
from .environment import (
    CONTENT_PROFILE,
    EXCLUDED_TAG_KEYS,
    EXCLUDED_VOUCHER_KEYS,
    MAX_EPISODE_STEPS,
    WIN_BONUS,
    BalatroEnvironment,
)
from .rules import POLICY_GUIDE

JACKDAW_UPSTREAM_BASE = "c84dca9227b40eb5f7ff9fd7cd78945aa07854ce"
JACKDAW_PATCHES = (
    "aaf24f93b4f22d3ee70a9099a211a7a6a93bef7e",
    "8e807df73797b500b1eccbdf26288f777619928c",
    "8dd66169014b58b7a077760ff1090efe1d4a022c",
    "a785574bc6deea1c71cd53fec5b102bb82d52e8f",
)
JACKDAW_LOCAL_PATCHES = (
    "content-exclusion-pool-plumbing-v1",
    "ceremonial-dagger-destroy-joker-v1",
)
JACKDAW_REVISION = "c84dca9+aaf24f9+8e807df+8dd6616+a785574+epg2"
_EPISODE_SEED_DOMAIN = b"evopolicygym-balatro/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_TRACE_PREFIX_STEPS = 192
_TRACE_SUFFIX_STEPS = 64
_MAX_REPLAY_BYTES = 15 * 1024 * 1024
_MAX_EPISODE_DIAGNOSTICS = 256


def _benchmark_spec(config: BalatroConfig) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="jackdaw/Balatro/red-deck-white-stake/run-score-v2",
        description=(
            "Play one complete white-stake Red Deck run through the Jackdaw "
            "headless Balatro engine. A win is worth 1000 points plus one point "
            "for every Blind cleared; failed Policies receive zero."
        ),
        observation_space={
            "type": "semantic_object",
            "schema": "evopolicygym-balatro/observation-v1",
            "fields": [
                "phase",
                "progress",
                "resources",
                "rules",
                "blind",
                "last_hand",
                "round_earnings",
                "hand",
                "jokers",
                "consumables",
                "shop",
                "pack",
                "deck",
                "poker_hands",
                "vouchers",
                "tags",
                "legal_actions",
            ],
            "notes": (
                "Entity indices refer to the matching list in the same "
                "observation. Draw-pile order and the Episode seed are hidden."
            ),
        },
        action_space={
            "type": "tagged_object",
            "discriminator": "kind",
            "kinds": [
                "play_hand",
                "discard",
                "select_blind",
                "skip_blind",
                "cash_out",
                "reroll_shop",
                "next_round",
                "skip_pack",
                "buy_card",
                "sell_joker",
                "sell_consumable",
                "use_consumable",
                "redeem_voucher",
                "open_booster",
                "pick_pack_card",
                "swap_joker_left",
                "swap_joker_right",
                "swap_hand_left",
                "swap_hand_right",
                "sort_hand_by_rank",
                "sort_hand_by_suit",
            ],
            "shapes": {
                "card_selection": {
                    "fields": ["kind", "card_indices"],
                    "applies_to": ["play_hand", "discard"],
                },
                "entity_target": {
                    "fields": ["kind", "target_index"],
                    "applies_to": [
                        "buy_card",
                        "sell_joker",
                        "sell_consumable",
                        "redeem_voucher",
                        "open_booster",
                        "swap_joker_left",
                        "swap_joker_right",
                        "swap_hand_left",
                        "swap_hand_right",
                    ],
                },
                "entity_and_cards": {
                    "fields": ["kind", "target_index", "card_indices"],
                    "applies_to": [
                        "use_consumable",
                        "pick_pack_card",
                    ],
                },
            },
            "notes": (
                "Actions must contain exactly the documented fields. Consult "
                "observation.legal_actions before every decision. Selection "
                "order matters when playing cards."
            ),
        },
        metadata={
            "environment": "Balatro",
            "engine": "Jackdaw",
            "engine_version": "0.1.0",
            "engine_revision": JACKDAW_REVISION,
            "engine_upstream_base": JACKDAW_UPSTREAM_BASE,
            "engine_patches": list(JACKDAW_PATCHES),
            "engine_local_patches": list(JACKDAW_LOCAL_PATCHES),
            "engine_license": "MIT",
            "excluded_content": {
                "tags": list(EXCLUDED_TAG_KEYS),
                "vouchers": list(EXCLUDED_VOUCHER_KEYS),
                "reason": (
                    "Excluded before RNG-backed pool selection because their gameplay "
                    "effects are not active in the pinned Jackdaw revision."
                ),
            },
            "win_bonus": WIN_BONUS,
            "unofficial": True,
            "original_game_assets_included": False,
            "feedback_diagnostics": (
                "Per-step and Episode summaries report legal action use, Blind chip "
                "progress, hand score and type, resource economy, purchases, pack "
                "choices, owned items, termination, and replay coverage."
            ),
            "visual_artifact": (
                "A bounded semantic replay is published as replay.jsonl. The headless "
                "distribution contains no official game art and does not synthesize a GIF."
            ),
            "policy_guide": POLICY_GUIDE,
        },
        max_episode_steps=MAX_EPISODE_STEPS,
        primary_metric="mean_run_score",
        score_direction="maximize",
        environment_parameters={
            "deck": config.deck,
            "stake": config.stake,
            "content_profile": CONTENT_PROFILE,
            "engine_revision": JACKDAW_REVISION,
        },
    )


class BalatroBenchmark:
    """Win-weighted progress over deterministic Jackdaw Balatro runs."""

    def __init__(self, config: BalatroConfig | None = None) -> None:
        selected = BalatroConfig() if config is None else config
        if type(selected) is not BalatroConfig:
            raise TypeError("config must be BalatroConfig or None")
        self._config = selected
        self._spec = _benchmark_spec(selected)

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
        return BalatroEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        outcomes = tuple(_outcome(record) for record in records)
        score = statistics.fmean(item["score"] for item in outcomes)
        wins = sum(bool(item["won"]) for item in outcomes)
        failures = sum(record.policy_failure is not None for record in records)
        action_counts = _action_counts(records)
        hand_type_counts = _aggregate_metric_counts(
            records,
            "hand_type_counts",
        )
        purchased_card_types = _aggregate_metric_counts(
            records,
            "purchased_card_types",
        )
        pack_pick_types = _aggregate_metric_counts(
            records,
            "pack_pick_types",
        )
        diagnostics: list[PolicyValue] = [
            _episode_diagnostics(record, episode_index=index)
            for index, record in enumerate(records[:_MAX_EPISODE_DIAGNOSTICS])
        ]
        replay, replay_episodes, replay_transitions = _replay_artifact(records)
        total_transitions = sum(record.steps for record in records)
        return Feedback(
            score=score,
            content={
                "summary": (f"Mean run score {score:.3f}; {wins}/{len(records)} runs won."),
                "mean_run_score": score,
                "win_rate": wins / len(records),
                "mean_ante_reached": statistics.fmean(item["ante"] for item in outcomes),
                "mean_rounds_cleared": statistics.fmean(
                    item["rounds_cleared"] for item in outcomes
                ),
                "mean_progress_ante_reached": _mean_final_metric(
                    records,
                    "ante",
                ),
                "mean_progress_rounds_cleared": _mean_final_metric(
                    records,
                    "rounds_cleared",
                ),
                "mean_episode_steps": statistics.fmean(record.steps for record in records),
                "mean_best_hand_score": _mean_final_metric(
                    records,
                    "best_hand_score",
                ),
                "mean_best_blind_progress_fraction": _mean_final_metric(
                    records,
                    "best_blind_progress_fraction",
                ),
                "mean_final_blind_progress_fraction": _mean_final_metric(
                    records,
                    "final_blind_progress_fraction",
                ),
                "mean_final_chips_to_target": _mean_final_metric(
                    records,
                    "chips_to_target",
                ),
                "mean_final_money": _mean_final_metric(records, "money"),
                "mean_peak_money": _mean_final_metric(records, "peak_money"),
                "total_money_gained": _sum_final_metric(
                    records,
                    "total_money_gained",
                ),
                "total_money_spent": _sum_final_metric(
                    records,
                    "total_money_spent",
                ),
                "total_purchase_cost": _sum_final_metric(
                    records,
                    "total_purchase_cost",
                ),
                "total_sale_value": _sum_final_metric(
                    records,
                    "total_sale_value",
                ),
                "action_counts": _policy_counts(action_counts),
                "hand_type_counts": _policy_counts(hand_type_counts),
                "purchased_card_types": _policy_counts(purchased_card_types),
                "pack_pick_types": _policy_counts(pack_pick_types),
                "episodes": len(records),
                "completed_episodes": len(records) - failures,
                "terminated_episodes": sum(_terminated(record) for record in records),
                "truncated_episodes": sum(_truncated(record) for record in records),
                "policy_failures": failures,
                "episode_diagnostics": diagnostics,
                "episode_diagnostics_omitted": len(records) - len(diagnostics),
                "engine_revision": JACKDAW_REVISION,
                "replay_episodes": replay_episodes,
                "replay_episodes_omitted": len(records) - replay_episodes,
                "replay_prefix_steps": _TRACE_PREFIX_STEPS,
                "replay_suffix_steps": _TRACE_SUFFIX_STEPS,
                "replay_transitions": replay_transitions,
                "replay_transitions_omitted": (total_transitions - replay_transitions),
            },
            artifacts=(replay,),
        )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _outcome(record: EpisodeRecord) -> dict[str, int | bool]:
    if record.policy_failure is not None:
        return {
            "score": 0,
            "won": False,
            "ante": 0,
            "rounds_cleared": 0,
        }
    if not record.transitions:
        raise ValueError("completed Balatro Episode has no transitions")
    metrics = record.transitions[-1].step.metrics
    if type(metrics) is not dict:
        raise ValueError("Balatro Episode metrics are invalid")
    score = _metric_int(metrics, "run_score")
    ante = _metric_int(metrics, "ante")
    rounds = _metric_int(metrics, "rounds_cleared")
    won = metrics.get("won")
    if type(won) is not bool:
        raise ValueError("Balatro won metric is invalid")
    if score != int(record.total_reward):
        raise ValueError("Balatro reward and run score disagree")
    return {
        "score": score,
        "won": won,
        "ante": ante,
        "rounds_cleared": rounds,
    }


def _metric_int(metrics: dict[str, PolicyValue], key: str) -> int:
    value = metrics.get(key)
    if type(value) is not int:
        raise ValueError(f"Balatro {key} metric is invalid")
    return value


def _replay_artifact(
    records: Sequence[EpisodeRecord],
) -> tuple[Artifact, int, int]:
    chunks: list[bytes] = []
    retained = 0
    retained_transitions = 0
    size = 0
    omission_reserve = len(
        _json_line(
            {
                "type": "episodes_omitted",
                "count": len(records),
            }
        )
    )
    for episode_index, record in enumerate(records):
        episode, episode_transitions = _replay_episode(
            episode_index,
            record,
        )
        reserve = omission_reserve if episode_index + 1 < len(records) else 0
        if size + len(episode) + reserve > _MAX_REPLAY_BYTES:
            break
        chunks.append(episode)
        retained += 1
        retained_transitions += episode_transitions
        size += len(episode)

    omitted = len(records) - retained
    if omitted:
        chunks.append(
            _json_line(
                {
                    "type": "episodes_omitted",
                    "count": omitted,
                }
            )
        )
    return (
        Artifact(
            name="replay.jsonl",
            media_type="application/x-ndjson",
            content=b"".join(chunks),
        ),
        retained,
        retained_transitions,
    )


def _replay_episode(
    episode_index: int,
    record: EpisodeRecord,
) -> tuple[bytes, int]:
    transition_indices = _replay_transition_indices(record.steps)
    lines = [
        _json_line(
            {
                "type": "episode",
                "episode_index": episode_index,
                "status": ("completed" if record.policy_failure is None else "policy_failed"),
                "steps": record.steps,
                "score": (record.total_reward if record.policy_failure is None else 0.0),
                "failure": record.policy_failure,
                "replayed_transitions": len(transition_indices),
                "transitions_omitted": (record.steps - len(transition_indices)),
                "initial_state": _policy_observation(
                    record.initial_observation,
                ),
            }
        )
    ]
    for step_index in transition_indices:
        transition = record.transitions[step_index]
        lines.append(
            _json_line(
                {
                    "type": "transition",
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "action": transition.action,
                    "reward": transition.step.reward,
                    "state": _policy_observation(
                        transition.step.observation,
                    ),
                    "metrics": transition.step.metrics,
                    "terminated": transition.step.terminated,
                    "truncated": transition.step.truncated,
                }
            )
        )
    omitted = record.steps - len(transition_indices)
    if omitted:
        lines.append(
            _json_line(
                {
                    "type": "transitions_omitted",
                    "episode_index": episode_index,
                    "count": omitted,
                    "prefix_steps": _TRACE_PREFIX_STEPS,
                    "suffix_steps": _TRACE_SUFFIX_STEPS,
                }
            )
        )
    return b"".join(lines), len(transition_indices)


def _replay_transition_indices(steps: int) -> tuple[int, ...]:
    if steps <= _TRACE_PREFIX_STEPS + _TRACE_SUFFIX_STEPS:
        return tuple(range(steps))
    return tuple(range(_TRACE_PREFIX_STEPS)) + tuple(range(steps - _TRACE_SUFFIX_STEPS, steps))


def _policy_observation(value: PolicyValue) -> dict[str, PolicyValue]:
    if type(value) is not dict:
        raise ValueError("Balatro Policy observation is invalid")
    return value


def _action_counts(records: Sequence[EpisodeRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for transition in record.transitions:
            action = transition.action
            if type(action) is not dict or type(action.get("kind")) is not str:
                raise ValueError("Balatro recorded Action is invalid")
            kind = action["kind"]
            assert type(kind) is str
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _final_metrics(record: EpisodeRecord) -> dict[str, PolicyValue] | None:
    if not record.transitions:
        return None
    metrics = record.transitions[-1].step.metrics
    if metrics is None:
        return None
    if type(metrics) is not dict:
        raise ValueError("Balatro Episode metrics are invalid")
    return metrics


def _aggregate_metric_counts(
    records: Sequence[EpisodeRecord],
    key: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        metrics = _final_metrics(record)
        if metrics is None:
            continue
        counts = metrics.get(key)
        if type(counts) is not dict:
            raise ValueError(f"Balatro {key} metric is invalid")
        for name, value in counts.items():
            if type(name) is not str or type(value) is not int:
                raise ValueError(f"Balatro {key} metric is invalid")
            result[name] = result.get(name, 0) + value
    return result


def _numeric_final_metrics(
    records: Sequence[EpisodeRecord],
    key: str,
) -> tuple[float, ...]:
    values: list[float] = []
    for record in records:
        metrics = _final_metrics(record)
        if metrics is None:
            continue
        value = metrics.get(key)
        if type(value) is int:
            values.append(float(value))
        elif type(value) is float:
            values.append(value)
        else:
            raise ValueError(f"Balatro {key} metric is invalid")
    return tuple(values)


def _mean_final_metric(
    records: Sequence[EpisodeRecord],
    key: str,
) -> float | None:
    values = _numeric_final_metrics(records, key)
    return statistics.fmean(values) if values else None


def _sum_final_metric(
    records: Sequence[EpisodeRecord],
    key: str,
) -> float | None:
    values = _numeric_final_metrics(records, key)
    return sum(values) if values else None


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


def _episode_diagnostics(
    record: EpisodeRecord,
    *,
    episode_index: int,
) -> dict[str, PolicyValue]:
    outcome = _outcome(record)
    metrics = _final_metrics(record)
    state = _final_observation(record)
    resources = _optional_public_dict(state, "resources")
    blind = _optional_public_dict(state, "blind")
    action_counts = _action_counts((record,))
    hand_type_counts = _metric_count_value(metrics, "hand_type_counts")
    purchased_card_types = _metric_count_value(
        metrics,
        "purchased_card_types",
    )
    pack_pick_types = _metric_count_value(metrics, "pack_pick_types")
    return {
        "episode_index": episode_index,
        "status": ("completed" if record.policy_failure is None else "policy_failed"),
        "score": outcome["score"],
        "progress_score_before_failure": record.total_reward,
        "won": outcome["won"],
        "ante_reached": outcome["ante"],
        "rounds_cleared": outcome["rounds_cleared"],
        "progress_ante_reached": _metric_value(metrics, "ante"),
        "progress_rounds_cleared": _metric_value(metrics, "rounds_cleared"),
        "steps": record.steps,
        "terminated": _terminated(record),
        "truncated": _truncated(record),
        "failure": record.policy_failure,
        "final_phase": _optional_text(state.get("phase")),
        "final_money": _optional_int(resources.get("money")),
        "final_hands_left": _optional_int(resources.get("hands_left")),
        "final_discards_left": _optional_int(resources.get("discards_left")),
        "final_blind_chips": _optional_int(resources.get("chips")),
        "final_blind_target_chips": _optional_int(blind.get("target_chips")),
        "final_chips_to_target": _metric_value(metrics, "chips_to_target"),
        "final_blind_progress_fraction": _metric_value(
            metrics,
            "final_blind_progress_fraction",
        ),
        "best_blind_progress_fraction": _metric_value(
            metrics,
            "best_blind_progress_fraction",
        ),
        "best_hand_score": _metric_value(metrics, "best_hand_score"),
        "mean_hand_score": _metric_value(metrics, "mean_hand_score"),
        "cards_played": _metric_value(metrics, "cards_played"),
        "cards_discarded": _metric_value(metrics, "cards_discarded"),
        "peak_money": _metric_value(metrics, "peak_money"),
        "total_money_gained": _metric_value(metrics, "total_money_gained"),
        "total_money_spent": _metric_value(metrics, "total_money_spent"),
        "total_purchase_cost": _metric_value(metrics, "total_purchase_cost"),
        "total_sale_value": _metric_value(metrics, "total_sale_value"),
        "owned_jokers": _metric_value(metrics, "owned_jokers"),
        "owned_consumables": _metric_value(metrics, "owned_consumables"),
        "owned_vouchers": _metric_value(metrics, "owned_vouchers"),
        "awarded_tags": _metric_value(metrics, "awarded_tags"),
        "action_counts": _policy_counts(action_counts),
        "hand_type_counts": _policy_counts(hand_type_counts),
        "purchased_card_types": _policy_counts(purchased_card_types),
        "pack_pick_types": _policy_counts(pack_pick_types),
    }


def _final_observation(record: EpisodeRecord) -> dict[str, PolicyValue]:
    value = (
        record.transitions[-1].step.observation
        if record.transitions
        else record.initial_observation
    )
    return _policy_observation(value)


def _optional_public_dict(
    value: dict[str, PolicyValue],
    key: str,
) -> dict[str, PolicyValue]:
    item = value.get(key)
    return item if type(item) is dict else {}


def _optional_int(value: PolicyValue | None) -> int | None:
    return value if type(value) is int else None


def _optional_text(value: PolicyValue | None) -> str | None:
    return value if type(value) is str else None


def _metric_value(
    metrics: dict[str, PolicyValue] | None,
    key: str,
) -> PolicyValue:
    return None if metrics is None else metrics.get(key)


def _metric_count_value(
    metrics: dict[str, PolicyValue] | None,
    key: str,
) -> dict[str, int]:
    if metrics is None:
        return {}
    value = metrics.get(key)
    if type(value) is not dict:
        raise ValueError(f"Balatro {key} metric is invalid")
    result: dict[str, int] = {}
    for name, count in value.items():
        if type(name) is not str or type(count) is not int:
            raise ValueError(f"Balatro {key} metric is invalid")
        result[name] = count
    return result


def _policy_counts(values: dict[str, int]) -> dict[str, PolicyValue]:
    return {key: value for key, value in sorted(values.items())}


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
    "BalatroBenchmark",
    "JACKDAW_LOCAL_PATCHES",
    "JACKDAW_PATCHES",
    "JACKDAW_REVISION",
    "JACKDAW_UPSTREAM_BASE",
]
