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

from .environment import (
    CONTENT_PROFILE,
    EXCLUDED_TAG_KEYS,
    EXCLUDED_VOUCHER_KEYS,
    MAX_EPISODE_STEPS,
    WIN_BONUS,
    BalatroEnvironment,
)
from .observation import replay_state
from .rules import POLICY_GUIDE

JACKDAW_UPSTREAM_BASE = "c84dca9227b40eb5f7ff9fd7cd78945aa07854ce"
JACKDAW_PATCHES = (
    "aaf24f93b4f22d3ee70a9099a211a7a6a93bef7e",
    "8e807df73797b500b1eccbdf26288f777619928c",
    "8dd66169014b58b7a077760ff1090efe1d4a022c",
    "a785574bc6deea1c71cd53fec5b102bb82d52e8f",
)
JACKDAW_LOCAL_PATCHES = ("content-exclusion-pool-plumbing-v1",)
JACKDAW_REVISION = "c84dca9+aaf24f9+8e807df+8dd6616+a785574+epg1"
_EPISODE_SEED_DOMAIN = b"evopolicygym-balatro/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_SCENARIO: dict[str, PolicyValue] = {"back": "b_red", "stake": 1}
_MAX_TRACED_TRANSITIONS = 256

_SPEC = BenchmarkSpec(
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
        "deck": "b_red",
        "stake": 1,
        "content_profile": CONTENT_PROFILE,
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
        "policy_guide": POLICY_GUIDE,
    },
    max_episode_steps=MAX_EPISODE_STEPS,
    primary_metric="mean_run_score",
    score_direction="maximize",
)


class BalatroBenchmark:
    """Win-weighted progress over deterministic Jackdaw Balatro runs."""

    @property
    def spec(self) -> BenchmarkSpec:
        return _SPEC

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
                scenario=_SCENARIO,
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return BalatroEnvironment(episode)

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
                "episodes": len(records),
                "policy_failures": failures,
                "engine_revision": JACKDAW_REVISION,
                "replay_episodes": len(records),
            },
            artifacts=(_replay_artifact(records),),
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


def _replay_artifact(records: Sequence[EpisodeRecord]) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": ("completed" if record.policy_failure is None else "policy_failed"),
                    "steps": record.steps,
                    "score": (record.total_reward if record.policy_failure is None else 0.0),
                    "failure": record.policy_failure,
                    "initial_state": replay_state(
                        record.initial_observation,
                    ),
                }
            )
        )
        transitions = record.transitions[:_MAX_TRACED_TRANSITIONS]
        for step_index, transition in enumerate(transitions):
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "action": transition.action,
                        "reward": transition.step.reward,
                        "state": replay_state(
                            transition.step.observation,
                        ),
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
        omitted = len(record.transitions) - len(transitions)
        if omitted:
            lines.append(
                _json_line(
                    {
                        "type": "transitions_omitted",
                        "episode_index": episode_index,
                        "count": omitted,
                    }
                )
            )
    return Artifact(
        name="replay.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


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
