#!/usr/bin/env python3
"""Compare visible score estimates with public replay outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from policy_system.hands import classify
from policy_system.scoring import ScoringContext, estimate_score


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    return parser.parse_args()


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ratio(predicted: float, actual: float) -> float:
    return predicted / actual if actual else 0.0


def main() -> int:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    joker_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in _arguments().replays:
        states: dict[int, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("type") in {
                "episodes_omitted",
                "transitions_omitted",
            }:
                continue
            episode = int(item["episode_index"])
            if item["type"] == "episode":
                states[episode] = _mapping(item.get("initial_state"))
                continue
            before = states[episode]
            after = _mapping(item.get("state"))
            action = _mapping(item.get("action"))
            states[episode] = after
            if action.get("kind") != "play_hand":
                continue
            hand = list(before.get("hand") or [])
            selected_indices = [int(x) for x in action.get("card_indices", [])]
            selected = [hand[index] for index in selected_indices]
            hand_name, scoring_local = classify(selected)
            poker = {
                str(value.get("name")): value
                for value in before.get("poker_hands") or []
            }
            resources = _mapping(before.get("resources"))
            deck = _mapping(before.get("deck"))
            predicted = estimate_score(
                hand=hand,
                selected_indices=selected_indices,
                scoring_local=scoring_local,
                hand_name=hand_name,
                poker_hand=poker.get(hand_name, {}),
                jokers=list(before.get("jokers") or []),
                context=ScoringContext(
                    money=int(resources.get("money", 0)),
                    deck_remaining=int(deck.get("draw_pile", 0)),
                    discards_left=int(resources.get("discards_left", 0)),
                    hands_left=int(resources.get("hands_left", 0)),
                ),
            )
            actual = float(_mapping(after.get("last_hand")).get("total", 0))
            groups[hand_name].append((predicted, actual))
            for joker in before.get("jokers") or []:
                joker_groups[str(joker.get("name", "?"))].append(
                    (predicted, actual)
                )

    def rows(source: dict[str, list[tuple[float, float]]]) -> list[str]:
        result = []
        for name, values in source.items():
            if len(values) < 3:
                continue
            ratios = [_ratio(predicted, actual) for predicted, actual in values]
            errors = [
                abs(predicted - actual) / actual
                for predicted, actual in values
                if actual
            ]
            result.append(
                f"{name}\t{len(values)}\t"
                f"{statistics.median(ratios):.3f}\t"
                f"{statistics.mean(errors):.3f}"
            )
        return sorted(result, key=lambda row: float(row.rsplit("\t", 1)[1]), reverse=True)

    print("group\tn\tmedian_predicted/actual\tMAPE")
    print("\n".join(rows(groups)))
    print("\nJokers")
    print("group\tn\tmedian_predicted/actual\tMAPE")
    print("\n".join(rows(joker_groups)[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
