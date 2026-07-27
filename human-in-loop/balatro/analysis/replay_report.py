#!/usr/bin/env python3
"""Summarize public Balatro replays without reading private Host state."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Decision:
    before: JsonObject
    action: JsonObject
    after: JsonObject
    reward: float
    terminated: bool


@dataclass
class Episode:
    source: Path
    index: int
    score: float
    status: str
    failure: JsonObject | None
    initial_state: JsonObject
    decisions: list[Decision] = field(default_factory=list)

    @property
    def terminal_state(self) -> JsonObject:
        if self.decisions:
            return self.decisions[-1].after
        return self.initial_state


@dataclass(frozen=True)
class ShopExit:
    source: Path
    episode: int
    ante: int
    money: int
    joker_count: int
    joker_slots: int
    affordable: tuple[str, ...]


def _object(value: object) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def discover_paths(arguments: Sequence[str]) -> list[Path]:
    if arguments:
        discovered: list[Path] = []
        for argument in arguments:
            path = Path(argument).expanduser()
            if path.is_dir():
                discovered.extend(sorted(path.glob("**/replay.jsonl")))
            elif path.is_file():
                discovered.append(path)
            else:
                raise FileNotFoundError(path)
        return list(dict.fromkeys(item.resolve() for item in discovered))

    repository = Path(__file__).resolve().parents[3]
    run = repository / (
        "runs/balatro-skill-ab-gpt-5.6-sol-retry-20260725-164423/"
        "submissions/submission-000019/artifacts/replay.jsonl"
    )
    return [run]


def load_replays(paths: Iterable[Path]) -> list[Episode]:
    episodes: list[Episode] = []
    for path in paths:
        by_index: dict[int, Episode] = {}
        current_states: dict[int, JsonObject] = {}
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                if item.get("type") in {
                    "episodes_omitted",
                    "transitions_omitted",
                }:
                    continue
                episode_index = _integer(item.get("episode_index"))
                item_type = item.get("type")
                if item_type == "episode":
                    initial = _object(item.get("initial_state"))
                    episode = Episode(
                        source=path,
                        index=episode_index,
                        score=_number(item.get("score")),
                        status=str(item.get("status", "")),
                        failure=(
                            _object(item.get("failure"))
                            if item.get("failure") is not None
                            else None
                        ),
                        initial_state=initial,
                    )
                    by_index[episode_index] = episode
                    current_states[episode_index] = initial
                    episodes.append(episode)
                    continue
                if item_type != "transition":
                    raise ValueError(
                        f"{path}:{line_number}: unknown replay item {item_type!r}"
                    )
                current_episode = by_index.get(episode_index)
                before = current_states.get(episode_index)
                if current_episode is None or before is None:
                    raise ValueError(
                        f"{path}:{line_number}: transition before episode header"
                    )
                after = _object(item.get("state"))
                current_episode.decisions.append(
                    Decision(
                        before=before,
                        action=_object(item.get("action")),
                        after=after,
                        reward=_number(item.get("reward")),
                        terminated=bool(item.get("terminated", False)),
                    )
                )
                current_states[episode_index] = after
    return episodes


def _progress(state: JsonObject) -> JsonObject:
    return _object(state.get("progress"))


def _resources(state: JsonObject) -> JsonObject:
    return _object(state.get("resources"))


def _blind(state: JsonObject) -> JsonObject:
    return _object(state.get("blind"))


def _card_label(card: object) -> str:
    item = _object(card)
    name = str(item.get("name") or item.get("key") or "?")
    cost = _integer(item.get("cost"))
    return f"{name} (${cost})"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _affordable_shop_items(state: JsonObject) -> tuple[str, ...]:
    money = _integer(_resources(state).get("money"))
    shop = _object(state.get("shop"))
    choices: list[str] = []
    for area in ("cards", "vouchers", "boosters"):
        for card in _list(shop.get(area)):
            if _integer(_object(card).get("cost")) <= money:
                choices.append(f"{area}:{_card_label(card)}")
    return tuple(choices)


def shop_exits(episodes: Iterable[Episode]) -> list[ShopExit]:
    exits: list[ShopExit] = []
    for episode in episodes:
        for decision in episode.decisions:
            if (
                decision.action.get("kind") != "next_round"
                or decision.before.get("phase") != "shop"
            ):
                continue
            resources = _resources(decision.before)
            progress = _progress(decision.before)
            exits.append(
                ShopExit(
                    source=episode.source,
                    episode=episode.index,
                    ante=_integer(progress.get("ante")),
                    money=_integer(resources.get("money")),
                    joker_count=len(_list(decision.before.get("jokers"))),
                    joker_slots=_integer(resources.get("joker_slots")),
                    affordable=_affordable_shop_items(decision.before),
                )
            )
    return exits


def joker_roles(card: object) -> tuple[str, ...]:
    joker = _object(card)
    ability = _object(joker.get("ability"))
    summary = str(_object(joker.get("rule")).get("summary", "")).lower()
    roles: set[str] = set()
    if _number(ability.get("t_chips")) or "chips" in summary:
        roles.add("chips")
    if (
        _number(ability.get("t_mult"))
        or _number(ability.get("mult"))
        or "+mult" in summary
    ):
        roles.add("+mult")
    if _number(ability.get("x_mult")) > 1 or "x mult" in summary:
        roles.add("xmult")
    if "retrigger" in summary:
        roles.add("retrigger")
    if any(word in summary for word in ("scales", "increases", "gains")):
        roles.add("scaling")
    if any(
        word in summary
        for word in ("earn $", "dollars", "money", "sell value", "reroll")
    ):
        roles.add("economy")
    return tuple(sorted(roles)) or ("unknown",)


def _last_scoring_actions(episode: Episode, count: int = 3) -> str:
    plays = [
        decision
        for decision in episode.decisions
        if decision.action.get("kind") == "play_hand"
    ][-count:]
    labels = []
    for decision in plays:
        last_hand = _object(decision.after.get("last_hand"))
        labels.append(
            f"{last_hand.get('hand_type') or '?'}"
            f":{_integer(last_hand.get('total'))}"
        )
    return ", ".join(labels) or "-"


def _terminal_jokers(episode: Episode) -> str:
    labels = []
    for card in _list(episode.terminal_state.get("jokers")):
        joker = _object(card)
        roles = "/".join(joker_roles(joker))
        labels.append(f"{joker.get('name', '?')}[{roles}]")
    return ", ".join(labels) or "-"


def _action_counts(episodes: Iterable[Episode]) -> collections.Counter[str]:
    return collections.Counter(
        str(decision.action.get("kind", "?"))
        for episode in episodes
        for decision in episode.decisions
    )


def render_markdown(episodes: Sequence[Episode]) -> str:
    if not episodes:
        return "# Balatro replay report\n\nNo episodes found.\n"

    wins = sum(bool(_progress(item.terminal_state).get("won")) for item in episodes)
    rounds = [
        _integer(_progress(item.terminal_state).get("rounds_cleared"))
        for item in episodes
    ]
    antes = [
        _integer(_progress(item.terminal_state).get("ante")) for item in episodes
    ]
    failures = sum(item.failure is not None for item in episodes)
    counts = _action_counts(episodes)
    exits = shop_exits(episodes)
    risky_exits = [
        item
        for item in exits
        if item.money >= 15 and item.affordable
    ]

    lines = [
        "# Balatro replay report",
        "",
        f"- Episodes: {len(episodes)}",
        f"- Wins: {wins} ({wins / len(episodes):.1%})",
        f"- Mean Blinds cleared: {statistics.fmean(rounds):.2f}",
        f"- Mean / max Ante reached: {statistics.fmean(antes):.2f} / {max(antes)}",
        f"- Policy failures: {failures}",
        f"- Shop exits with $15+ and affordable inventory: "
        f"{len(risky_exits)} / {len(exits)}",
        "",
        "## Action mix",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {kind} | {count} |" for kind, count in counts.most_common()
    )
    lines.extend(
        [
            "",
            "## Episode terminals",
            "",
            "| Source | Ep | Score | Ante | Blinds | Money | "
            "Blind chips | Last scoring actions |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for episode in episodes:
        terminal = episode.terminal_state
        progress = _progress(terminal)
        resources = _resources(terminal)
        blind = _blind(terminal)
        lines.append(
            f"| {episode.source.parents[1].name} | {episode.index} | "
            f"{episode.score:g} | {_integer(progress.get('ante'))} | "
            f"{_integer(progress.get('rounds_cleared'))} | "
            f"${_integer(resources.get('money'))} | "
            f"{_integer(resources.get('chips'))}/"
            f"{_integer(blind.get('target_chips'))} | "
            f"{_last_scoring_actions(episode)} |"
        )

    lines.extend(
        [
            "",
            "## High-cash shop exits",
            "",
            "These are diagnosis candidates, not automatic proof that every "
            "available item should have been bought.",
            "",
            "| Source | Ep | Ante | Money | Jokers | Affordable inventory |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in sorted(
        risky_exits,
        key=lambda exit_: (exit_.money, exit_.ante),
        reverse=True,
    )[:30]:
        inventory = _markdown_cell(", ".join(item.affordable))
        lines.append(
            f"| {item.source.parents[1].name} | {item.episode} | "
            f"{item.ante} | ${item.money} | "
            f"{item.joker_count}/{item.joker_slots} | {inventory} |"
        )
    if not risky_exits:
        lines.append("| - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Terminal builds",
            "",
            "| Source | Ep | Jokers and heuristic roles |",
            "|---|---:|---|",
        ]
    )
    for episode in episodes:
        lines.append(
            f"| {episode.source.parents[1].name} | {episode.index} | "
            f"{_markdown_cell(_terminal_jokers(episode))} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Replay files or directories containing replay.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown to this path instead of stdout",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = render_markdown(load_replays(discover_paths(arguments.paths)))
    if arguments.output is None:
        print(report, end="")
    else:
        arguments.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
