"""Replay every public observation through the current Policy and gateway."""

from __future__ import annotations

import glob
import json
import os
import pathlib
import unittest
from typing import Any

from policy import make_policy

_SIMPLE = {
    "select_blind",
    "skip_blind",
    "cash_out",
    "reroll_shop",
    "next_round",
    "skip_pack",
    "sort_hand_by_rank",
    "sort_hand_by_suit",
}


def _descriptor(state: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(
        item for item in state["legal_actions"] if item.get("kind") == kind
    )


def _validate_action(state: dict[str, Any], action: dict[str, Any]) -> None:
    kind = action["kind"]
    descriptor = _descriptor(state, kind)
    if kind in _SIMPLE:
        if set(action) != {"kind"}:
            raise AssertionError(f"unexpected {kind} fields: {action}")
        return
    if kind in {"play_hand", "discard"}:
        if set(action) != {"kind", "card_indices"}:
            raise AssertionError(f"unexpected {kind} fields: {action}")
        indices = action["card_indices"]
        allowed = set(descriptor["card_indices"])
        if (
            len(indices) != len(set(indices))
            or not set(indices) <= allowed
            or not descriptor["min_cards"]
            <= len(indices)
            <= descriptor["max_cards"]
        ):
            raise AssertionError(f"invalid {kind} targets: {action}")
        return
    if kind in {"use_consumable", "pick_pack_card"}:
        if set(action) != {"kind", "target_index", "card_indices"}:
            raise AssertionError(f"unexpected {kind} fields: {action}")
        target = next(
            item
            for item in descriptor.get("targets", [])
            if item.get("target_index") == action["target_index"]
        )
        indices = action["card_indices"]
        if (
            len(indices) != len(set(indices))
            or not set(indices) <= set(target.get("card_indices", []))
            or not target.get("min_cards", 0)
            <= len(indices)
            <= target.get("max_cards", 0)
        ):
            raise AssertionError(f"invalid {kind} targets: {action}")
        return
    if set(action) != {"kind", "target_index"}:
        raise AssertionError(f"unexpected {kind} fields: {action}")
    if action["target_index"] not in descriptor["target_indices"]:
        raise AssertionError(f"invalid {kind} target: {action}")


def _replay_paths() -> list[pathlib.Path]:
    configured = os.environ.get("BALATRO_REPLAY_GLOB")
    repository = pathlib.Path(__file__).resolve().parents[4]
    pattern = configured or str(
        repository
        / (
            "runs/balatro-skill-ab-gpt-5.6-sol-retry-20260725-164423/"
            "submissions/submission-000019/artifacts/replay.jsonl"
        )
    )
    if configured:
        return [pathlib.Path(item) for item in sorted(glob.glob(pattern))]
    return [pathlib.Path(pattern)]


class ReplayTest(unittest.TestCase):
    def test_all_public_observations_produce_exact_legal_actions(self) -> None:
        checked = 0
        for path in _replay_paths():
            policies: dict[int, Any] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                episode_index = item["episode_index"]
                if item["type"] == "episode":
                    policies[episode_index] = make_policy(None)  # type: ignore[arg-type]
                    state = item["initial_state"]
                else:
                    state = item["state"]
                if state["phase"] == "game_over":
                    continue
                action = policies[episode_index].act(state)
                _validate_action(state, action)
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
