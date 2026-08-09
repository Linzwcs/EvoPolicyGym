"""Render a bounded semantic Assessment trace as a leaderboard GIF.

Run this artifact helper with Pillow available, for example:

    uv run --with pillow scripts/render_semantic_rollout.py ...

The input must come from ``scripts/export_assessment_artifact.py`` so the
rendered rollout remains tied to the selected Program and held-out seed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_BACKGROUND = "#07111f"
_PANEL = "#0d1b2d"
_MUTED = "#8fa5bd"
_TEXT = "#edf7ff"
_ACCENT = "#5ee7ff"
_SUCCESS = "#55e68a"
_DANGER = "#ff6b86"
_MAX_FRAMES = 72


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if (
        namespace.output.exists() or namespace.output.is_symlink()
    ) and not namespace.overwrite:
        parser.error("--output must not already exist")
    if not namespace.output.parent.is_dir():
        parser.error("--output parent directory must exist")

    records = _load_jsonl(namespace.input)
    renderers = {
        "keycorridor": _render_keycorridor,
        "treants-forest": _render_treants,
        "balatro": _render_balatro,
    }
    frames = renderers[namespace.environment](records, namespace.title)
    if not frames:
        parser.error("trace produced no renderable frames")
    frames[0].save(
        namespace.output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=namespace.frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        json.dumps(
            {
                "frames": len(frames),
                "input": str(namespace.input),
                "output": str(namespace.output),
                "size": list(frames[0].size),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if type(value) is not dict:
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records or records[0].get("type") != "episode":
        raise ValueError(f"{path}: first record must describe the Episode")
    return records


def _render_keycorridor(
    records: Sequence[dict[str, Any]], title: str
) -> list[Image.Image]:
    episode = records[0]
    transitions = [item for item in records[1:] if item.get("type") == "transition"]
    observations = [episode["initial_observation"]]
    observations.extend(item["next_observation"] for item in transitions)
    selected = _selected_indices(len(observations))
    frames: list[Image.Image] = []
    for frame_index in selected:
        observation = observations[frame_index]
        transition = transitions[frame_index - 1] if frame_index else None
        frames.append(
            _keycorridor_frame(
                observation,
                transition,
                title=title,
                step=frame_index,
                total_steps=len(transitions),
                episode_return=float(episode.get("return", 0.0)),
                success=bool(episode.get("success", False)),
            )
        )
    return frames


def _keycorridor_frame(
    observation: dict[str, Any],
    transition: dict[str, Any] | None,
    *,
    title: str,
    step: int,
    total_steps: int,
    episode_return: float,
    success: bool,
) -> Image.Image:
    image = Image.new("RGB", (640, 440), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    _header(draw, title, "MiniGrid · KeyCorridor-S4R3", 640)

    cell = 46
    grid_left = 28
    grid_top = 88
    objects = {
        (int(item["x"]), int(item["y"])): item
        for item in observation.get("visible_objects", [])
    }
    for y in range(7):
        for x in range(7):
            left = grid_left + x * cell
            top = grid_top + y * cell
            symbol = observation["grid_rows"][y][x]
            fill = {
                "?": "#091321",
                " ": "#142337",
                "#": "#536276",
                ".": "#24364b",
                "G": "#215d43",
                "L": "#7c2838",
            }.get(symbol, "#182a3e")
            draw.rounded_rectangle(
                (left + 1, top + 1, left + cell - 2, top + cell - 2),
                radius=5,
                fill=fill,
                outline="#223850",
                width=1,
            )
            item = objects.get((x, y))
            if item is not None:
                _draw_minigrid_object(draw, item, left, top, cell)

    agent_x = grid_left + 3 * cell + cell // 2
    agent_y = grid_top + 6 * cell + cell // 2
    draw.polygon(
        [(agent_x, agent_y - 14), (agent_x - 12, agent_y + 11), (agent_x + 12, agent_y + 11)],
        fill=_ACCENT,
        outline="#d7fbff",
    )

    panel_left = 374
    draw.rounded_rectangle((panel_left, 88, 614, 410), radius=14, fill=_PANEL)
    mission = str(observation.get("mission", ""))
    _label(draw, panel_left + 18, 108, "MISSION")
    _wrapped(draw, mission, (panel_left + 18, 132), 210, _font(18, bold=True), _TEXT)
    direction = ("east", "south", "west", "north")[int(observation.get("direction", 0))]
    _metric(draw, panel_left + 18, 190, "World direction", direction)
    if transition is None:
        action = "Episode reset"
        stage = "search"
        carried = "none"
    else:
        metrics = transition.get("metrics", {})
        action = str(transition.get("action_meaning", "unknown"))
        stage = str(metrics.get("task_stage", "in progress"))
        carried = str(metrics.get("carried_object", "none")) or "none"
    _metric(draw, panel_left + 18, 242, "Action", action.replace("_", " "))
    _metric(draw, panel_left + 18, 294, "Stage", stage.replace("_", " "))
    _metric(draw, panel_left + 18, 346, "Carrying", carried)

    _footer(
        draw,
        width=640,
        step=step,
        total_steps=total_steps,
        score=episode_return,
        score_label="Episode return",
        terminal=step == total_steps,
        success=success,
    )
    return image


def _draw_minigrid_object(
    draw: ImageDraw.ImageDraw,
    item: dict[str, Any],
    left: int,
    top: int,
    cell: int,
) -> None:
    colors = {
        "red": "#ff5f72",
        "green": "#55e68a",
        "blue": "#53a9ff",
        "purple": "#b682ff",
        "yellow": "#ffd45e",
        "grey": "#8a9aad",
    }
    color = colors.get(str(item.get("color")), "#d4e4f2")
    kind = item.get("object")
    if kind == "door":
        draw.rounded_rectangle(
            (left + 9, top + 4, left + cell - 9, top + cell - 4),
            radius=3,
            fill=color if item.get("state") != "open" else "#16283c",
            outline=color,
            width=3,
        )
        draw.ellipse((left + cell - 16, top + 20, left + cell - 11, top + 25), fill="#07111f")
    elif kind == "key":
        draw.ellipse((left + 8, top + 9, left + 24, top + 25), outline=color, width=4)
        draw.line((left + 22, top + 23, left + 36, top + 37), fill=color, width=5)
    elif kind in {"ball", "box"}:
        bounds = (left + 9, top + 9, left + cell - 9, top + cell - 9)
        if kind == "ball":
            draw.ellipse(bounds, fill=color, outline="#f2fbff", width=2)
        else:
            draw.rounded_rectangle(bounds, radius=5, fill=color, outline="#f2fbff", width=2)
    elif kind == "wall":
        draw.line((left + 7, top + 15, left + cell - 7, top + 15), fill="#a7b5c5", width=2)


def _render_treants(
    records: Sequence[dict[str, Any]], title: str
) -> list[Image.Image]:
    episode = records[0]
    transitions = [item for item in records[1:] if item.get("type") == "transition"]
    if not transitions:
        return []
    initial_observation = transitions[0]["observation"]
    forest = initial_observation["initial"]
    original_trees = {tuple(item) for item in forest["trees"]}
    placed: set[tuple[int, int]] = set()
    revealed = {tuple(item) for item in initial_observation["newly_revealed"]}
    trail = {tuple(initial_observation["adventurer"])}
    selected = set(_selected_indices(len(transitions) + 1))
    frames: list[Image.Image] = []
    if 0 in selected:
        frames.append(
            _treants_frame(
                title=title,
                size=int(forest["size"]),
                entrance=tuple(forest["entrance"]),
                flower=tuple(forest["flower"]),
                original_trees=original_trees,
                placed=placed,
                revealed=revealed,
                newly_revealed=revealed,
                trail=trail,
                adventurer=tuple(initial_observation["adventurer"]),
                metrics=None,
                step=0,
                total_steps=len(transitions),
                episode_score=float(episode.get("score", 0.0)),
            )
        )
    for transition_index, transition in enumerate(transitions, start=1):
        placed.update(tuple(item) for item in transition["action"]["placements"])
        next_observation = transition["next_observation"]
        newly_revealed = {tuple(item) for item in next_observation["newly_revealed"]}
        revealed.update(newly_revealed)
        adventurer = tuple(next_observation["adventurer"])
        trail.add(adventurer)
        if transition_index not in selected:
            continue
        frames.append(
            _treants_frame(
                title=title,
                size=int(forest["size"]),
                entrance=tuple(forest["entrance"]),
                flower=tuple(forest["flower"]),
                original_trees=original_trees,
                placed=placed,
                revealed=revealed,
                newly_revealed=newly_revealed,
                trail=trail,
                adventurer=adventurer,
                metrics=transition.get("metrics"),
                step=transition_index,
                total_steps=len(transitions),
                episode_score=float(episode.get("score", 0.0)),
            )
        )
    return frames


def _treants_frame(
    *,
    title: str,
    size: int,
    entrance: tuple[int, int],
    flower: tuple[int, int],
    original_trees: set[tuple[int, int]],
    placed: set[tuple[int, int]],
    revealed: set[tuple[int, int]],
    newly_revealed: set[tuple[int, int]],
    trail: set[tuple[int, int]],
    adventurer: tuple[int, int],
    metrics: dict[str, Any] | None,
    step: int,
    total_steps: int,
    episode_score: float,
) -> Image.Image:
    image = Image.new("RGB", (680, 500), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    _header(draw, title, "AtCoder AHC054 · Treants Forest", 680)
    grid_left = 28
    grid_top = 92
    cell = 9
    for row in range(size):
        for column in range(size):
            coordinate = (row, column)
            left = grid_left + column * cell
            top = grid_top + row * cell
            fill = "#0a1723"
            if coordinate in revealed:
                fill = "#18382e"
            if coordinate in trail:
                fill = "#1b4a46"
            if coordinate in original_trees:
                fill = "#214c36"
            if coordinate in placed:
                fill = "#31875a"
            if coordinate in newly_revealed:
                fill = "#316866"
            draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=fill)
    _grid_marker(draw, grid_left, grid_top, cell, entrance, "#61d8ff", radius=3)
    _grid_marker(draw, grid_left, grid_top, cell, flower, "#ff73c8", radius=4)
    _grid_marker(draw, grid_left, grid_top, cell, adventurer, "#fff4a8", radius=4)

    panel_left = 416
    draw.rounded_rectangle((panel_left, 92, 652, 452), radius=14, fill=_PANEL)
    _label(draw, panel_left + 18, 112, "FOREST STATE")
    _metric(draw, panel_left + 18, 140, "Turn", f"{step:,} / {total_steps:,}")
    _metric(draw, panel_left + 18, 192, "Placed Treants", f"{len(placed):,}")
    _metric(draw, panel_left + 18, 244, "Revealed cells", f"{len(revealed):,}")
    if metrics is None:
        path_length = "N/A"
        distance = abs(adventurer[0] - flower[0]) + abs(adventurer[1] - flower[1])
    else:
        path_length = f"{int(metrics.get('flower_path_length', 0)):,}"
        distance = int(metrics.get("flower_manhattan_distance", 0))
    _metric(draw, panel_left + 18, 296, "Flower path", path_length)
    _metric(draw, panel_left + 18, 348, "Manhattan distance", str(distance))
    _metric(draw, panel_left + 18, 400, "Episode score", f"{episode_score:,.0f}")
    _footer(
        draw,
        width=680,
        step=step,
        total_steps=total_steps,
        score=episode_score,
        score_label="Capped turns",
        terminal=step == total_steps,
        success=step == total_steps and step >= 2048,
        top=464,
    )
    return image


def _grid_marker(
    draw: ImageDraw.ImageDraw,
    grid_left: int,
    grid_top: int,
    cell: int,
    coordinate: tuple[int, int],
    fill: str,
    *,
    radius: int,
) -> None:
    row, column = coordinate
    center_x = grid_left + column * cell + cell // 2
    center_y = grid_top + row * cell + cell // 2
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=fill,
        outline="#f4fbff",
    )


def _render_balatro(
    records: Sequence[dict[str, Any]], title: str
) -> list[Image.Image]:
    episode = records[0]
    transitions = [item for item in records[1:] if item.get("type") == "transition"]
    states = [episode["initial_state"]]
    states.extend(item["state"] for item in transitions)
    selected = _selected_indices(len(states))
    frames: list[Image.Image] = []
    for frame_index in selected:
        transition = transitions[frame_index - 1] if frame_index else None
        frames.append(
            _balatro_frame(
                state=states[frame_index],
                action=transition.get("action") if transition else None,
                title=title,
                step=frame_index,
                total_steps=len(transitions),
                episode_score=float(episode.get("score", 0.0)),
                terminal=bool(transition and transition.get("terminated")),
            )
        )
    return frames


def _balatro_frame(
    *,
    state: dict[str, Any],
    action: dict[str, Any] | None,
    title: str,
    step: int,
    total_steps: int,
    episode_score: float,
    terminal: bool,
) -> Image.Image:
    image = Image.new("RGB", (720, 500), "#071612")
    draw = ImageDraw.Draw(image)
    _header(draw, title, "Jackdaw · Balatro Red Deck / White Stake", 720)

    progress = state.get("progress", {})
    resources = state.get("resources", {})
    blind = state.get("blind", {})
    phase = str(state.get("phase", "unknown")).replace("_", " ").title()
    draw.rounded_rectangle((24, 86, 696, 136), radius=12, fill="#102b24")
    _pill(draw, 40, 100, f"ANTE {progress.get('ante', 1)} / {progress.get('win_ante', 8)}")
    _pill(draw, 155, 100, phase.upper(), accent=True)
    _pill(draw, 365, 100, f"${resources.get('money', 0)}")
    _pill(draw, 446, 100, f"HANDS {resources.get('hands_left', 0)}")
    _pill(draw, 565, 100, f"DISCARDS {resources.get('discards_left', 0)}")

    target = max(1.0, float(blind.get("target_chips", 1) or 1))
    chips = float(resources.get("chips", 0) or 0)
    _label(draw, 34, 154, str(blind.get("name", "Blind")))
    draw.rounded_rectangle((34, 178, 686, 198), radius=8, fill="#132d29")
    progress_width = int(652 * min(1.0, chips / target))
    if progress_width:
        draw.rounded_rectangle(
            (34, 178, 34 + progress_width, 198),
            radius=8,
            fill="#43d89b",
        )
    _right_text(draw, 682, 154, f"{chips:,.0f} / {target:,.0f} CHIPS", _font(13, bold=True), _TEXT)

    _label(draw, 34, 218, "HAND")
    hand = state.get("hand", [])
    selected_cards = set(action.get("card_indices", [])) if action else set()
    for index, card in enumerate(hand[:10]):
        _playing_card(draw, 34 + index * 56, 242, card, index in selected_cards)

    _label(draw, 34, 338, "JOKERS")
    jokers = state.get("jokers", [])
    if jokers:
        for index, joker in enumerate(jokers[:5]):
            left = 34 + index * 124
            draw.rounded_rectangle(
                (left, 362, left + 112, 408),
                radius=8,
                fill="#251f43",
                outline="#8c79e8",
                width=2,
            )
            _fit_text(
                draw,
                str(joker.get("name", "Joker")),
                (left + 8, 372),
                96,
                _font(13, bold=True),
                "#f1eaff",
            )
    else:
        draw.text((34, 364), "No Jokers", font=_font(16), fill=_MUTED)

    action_label = _balatro_action_label(action)
    draw.rounded_rectangle((24, 421, 696, 462), radius=10, fill="#102b24")
    draw.text((38, 434), "ACTION", font=_font(11, bold=True), fill=_MUTED)
    _fit_text(draw, action_label, (104, 430), 570, _font(16, bold=True), _TEXT)

    _footer(
        draw,
        width=720,
        step=step,
        total_steps=total_steps,
        score=episode_score,
        score_label="Final score",
        terminal=terminal or step == total_steps,
        success=bool(progress.get("won", False)),
        top=468,
    )
    return image


def _playing_card(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    card: dict[str, Any],
    selected: bool,
) -> None:
    facing = card.get("facing", "front")
    outline = "#5ee7ff" if selected else "#94a8ba"
    width = 4 if selected else 1
    draw.rounded_rectangle(
        (left, top, left + 48, top + 76),
        radius=7,
        fill="#f3f7f8" if facing == "front" else "#304a66",
        outline=outline,
        width=width,
    )
    if facing != "front":
        draw.text((left + 15, top + 26), "?", font=_font(21, bold=True), fill="#dcecff")
        return
    suit = str(card.get("suit", ""))
    suit_symbols = {"Hearts": "♥", "Diamonds": "♦", "Spades": "♠", "Clubs": "♣"}
    color = "#d83b57" if suit in {"Hearts", "Diamonds"} else "#182334"
    rank = str(card.get("rank", "?"))
    draw.text((left + 6, top + 5), rank, font=_font(14, bold=True), fill=color)
    symbol = suit_symbols.get(suit, "·")
    draw.text((left + 14, top + 34), symbol, font=_font(23, bold=True), fill=color)
    if card.get("debuffed"):
        draw.line((left + 5, top + 70, left + 43, top + 6), fill="#d83b57", width=3)


def _balatro_action_label(action: dict[str, Any] | None) -> str:
    if action is None:
        return "Episode reset"
    kind = str(action.get("kind", "unknown")).replace("_", " ")
    details: list[str] = []
    if "card_indices" in action:
        details.append("cards " + ", ".join(str(item) for item in action["card_indices"]))
    for key in ("index", "shop_index", "pack_index", "target_index"):
        if key in action:
            details.append(f"{key.replace('_', ' ')} {action[key]}")
    return kind.title() + (" · " + " · ".join(details) if details else "")


def _selected_indices(length: int, maximum: int = _MAX_FRAMES) -> tuple[int, ...]:
    if length <= maximum:
        return tuple(range(length))
    return tuple(
        dict.fromkeys(round(index * (length - 1) / (maximum - 1)) for index in range(maximum))
    )


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> None:
    draw.text((24, 18), title, font=_font(22, bold=True), fill=_TEXT)
    _right_text(draw, width - 24, 22, subtitle, _font(13), _MUTED)
    draw.line((24, 62, width - 24, 62), fill="#21415b", width=1)


def _footer(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    step: int,
    total_steps: int,
    score: float,
    score_label: str,
    terminal: bool,
    success: bool,
    top: int = 414,
) -> None:
    if top < 468:
        draw.line((24, top, width - 24, top), fill="#21415b", width=1)
    state = "SUCCESS" if terminal and success else ("COMPLETE" if terminal else "RUNNING")
    state_color = _SUCCESS if terminal and success else (_ACCENT if not terminal else _DANGER)
    draw.text((24, top + 10), f"STEP {step:,} / {total_steps:,}", font=_font(12, bold=True), fill=_MUTED)
    _right_text(
        draw,
        width - 24,
        top + 10,
        f"{score_label.upper()} {score:,.4g}  ·  {state}",
        _font(12, bold=True),
        state_color,
    )


def _label(draw: ImageDraw.ImageDraw, left: int, top: int, value: str) -> None:
    draw.text((left, top), value, font=_font(11, bold=True), fill=_MUTED)


def _metric(draw: ImageDraw.ImageDraw, left: int, top: int, label: str, value: str) -> None:
    _label(draw, left, top, label.upper())
    _fit_text(draw, value, (left, top + 19), 200, _font(17, bold=True), _TEXT)


def _pill(
    draw: ImageDraw.ImageDraw, left: int, top: int, value: str, *, accent: bool = False
) -> None:
    font = _font(11, bold=True)
    bounds = draw.textbbox((0, 0), value, font=font)
    width = bounds[2] - bounds[0] + 20
    fill = "#174c55" if accent else "#183a31"
    draw.rounded_rectangle((left, top, left + width, top + 23), radius=9, fill=fill)
    draw.text((left + 10, top + 5), value, font=font, fill=_TEXT)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    position: tuple[int, int],
    maximum_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    clipped = value
    while clipped and draw.textlength(clipped, font=font) > maximum_width:
        clipped = clipped[:-1]
    if clipped != value and len(clipped) > 1:
        clipped = clipped[:-1] + "…"
    draw.text(position, clipped, font=font, fill=fill)


def _wrapped(
    draw: ImageDraw.ImageDraw,
    value: str,
    position: tuple[int, int],
    maximum_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > maximum_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for index, line in enumerate(lines[:2]):
        draw.text((position[0], position[1] + index * 22), line, font=font, fill=fill)


def _right_text(
    draw: ImageDraw.ImageDraw,
    right: int,
    top: int,
    value: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    width = math.ceil(draw.textlength(value, font=font))
    draw.text((right - width, top), value, font=font, fill=fill)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(family, size)
    except OSError:
        return ImageFont.load_default()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a semantic Assessment Episode trace as an animated GIF."
    )
    parser.add_argument(
        "--environment",
        choices=("keycorridor", "treants-forest", "balatro"),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
