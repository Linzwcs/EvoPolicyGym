#!/usr/bin/env python3
"""Build the standalone EvoPolicyGym showcase site.

The clips generated here are not synthetic strategy sketches.  Each clip is
rendered from persisted run artifacts: the selected submit's trajectory rows
and recorded observation arrays under ``runs/.../workspace/feedback``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figs" / "data"
OUT_DIR = ROOT / "web"
CANVAS = (720, 405)
VISUAL_BOX = (28, 72, 448, 352)
TELEMETRY_X = 480

MODEL_ACCENTS = {
    "gpt_5_5": "#2f6fed",
    "claude_opus_4_7": "#c85f32",
    "minimax_m3": "#1f9a72",
    "deepseek_v4_pro": "#7a4bd6",
}

ENV_SLUGS = {
    "racing": "CarRacing",
    "halfcheetah5": "HalfCheetah",
    "minigrid_obstructedmaze": "ObstructedMaze",
    "fetch_push": "FetchPush",
}


@dataclass(frozen=True)
class EpisodeArtifact:
    directory: Path
    trajectory: Path
    observations: Path | None
    episode_id: int
    local_index: int
    episode_return: float | None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    media_dir = out_dir / "media"
    data_dir = out_dir / "data"
    media_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    runs = keyed(read_csv(DATA_DIR / "runs.csv"), "run_id")
    leaderboard = read_csv(DATA_DIR / "leaderboard_raw.csv")
    strategy_epochs = read_csv(DATA_DIR / "strategy_2x2_epochs.csv")
    strategy_submissions = read_csv(DATA_DIR / "strategy_2x2_submissions.csv")
    display_labels = {
        (row["env_id"], row["model_slug"], row["epoch_id"]): row["display_label"]
        for row in read_csv(DATA_DIR / "strategy_2x2_display_labels.csv")
    }

    submissions_by_lane: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    submissions_by_epoch: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in strategy_submissions:
        lane = (row["env_id"], row["model_slug"])
        submissions_by_lane[lane].append(row)
        submissions_by_epoch[(row["env_id"], row["model_slug"], row["epoch_id"])].append(row)

    epochs_by_lane: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in strategy_epochs:
        epochs_by_lane[(row["env_id"], row["model_slug"])].append(row)

    clips: list[dict[str, Any]] = []
    warnings: list[str] = []
    if args.mode == "final":
        for run in sorted(runs.values(), key=lambda row: (intish(row["env_order"]), intish(row["model_order"]))):
            clip, warning = build_final_clip(
                run,
                media_dir,
                out_dir,
                max_frames=args.max_frames,
                frame_duration=args.frame_duration,
                force=args.force,
            )
            if clip is not None:
                clips.append(clip)
            if warning:
                warnings.append(warning)
    else:
        for lane, epoch_rows in sorted(epochs_by_lane.items()):
            selected = select_epochs(
                epoch_rows,
                submissions_by_lane[lane],
                all_strategies=args.all_strategies,
                clips_per_lane=args.clips_per_lane,
            )
            for epoch in selected:
                clip, warning = build_clip(
                    epoch,
                    submissions_by_epoch[(epoch["env_id"], epoch["model_slug"], epoch["epoch_id"])],
                    runs,
                    display_labels,
                    media_dir,
                    out_dir,
                    max_frames=args.max_frames,
                    frame_duration=args.frame_duration,
                    force=args.force,
                )
                if clip is not None:
                    clips.append(clip)
                if warning:
                    warnings.append(warning)

    payload = build_payload(
        runs=list(runs.values()),
        leaderboard=leaderboard,
        epochs=strategy_epochs,
        clips=clips,
        warnings=warnings,
        mode=args.mode,
        all_strategies=args.all_strategies,
        clips_per_lane=args.clips_per_lane,
    )
    write_data_js(data_dir / "showcase-data.js", payload)
    print(f"wrote {len(clips)} clips to {media_dir.relative_to(ROOT)}")
    print(f"wrote data to {(data_dir / 'showcase-data.js').relative_to(ROOT)}")
    if warnings:
        print(f"warnings: {len(warnings)}")
        for warning in warnings[:12]:
            print(f"- {warning}")
        if len(warnings) > 12:
            print(f"- ... {len(warnings) - 12} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR), help="showcase output directory")
    parser.add_argument(
        "--mode",
        choices=("final", "strategies"),
        default="final",
        help="final builds one clip from each run's selected final checkpoint; strategies builds representative audited epochs",
    )
    parser.add_argument("--clips-per-lane", type=int, default=3)
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--max-frames", type=int, default=36)
    parser.add_argument("--frame-duration", type=float, default=0.18)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def select_epochs(
    epoch_rows: list[dict[str, str]],
    submission_rows: list[dict[str, str]],
    *,
    all_strategies: bool,
    clips_per_lane: int,
) -> list[dict[str, str]]:
    rows = sorted(epoch_rows, key=lambda row: int(row["epoch_id"]))
    if all_strategies or clips_per_lane <= 0 or len(rows) <= clips_per_lane:
        return rows

    by_epoch = {row["epoch_id"]: row for row in rows}
    chosen: list[str] = []

    def add(epoch_id: str) -> None:
        if epoch_id in by_epoch and epoch_id not in chosen:
            chosen.append(epoch_id)

    add(rows[0]["epoch_id"])
    scored = [row for row in submission_rows if numeric(row.get("score")) is not None]
    if scored:
        best = max(scored, key=lambda row: numeric(row["score"]) or -math.inf)
        add(best["epoch_id"])
    add(rows[-1]["epoch_id"])

    if len(chosen) < clips_per_lane:
        improved = [
            row
            for row in sorted(
                submission_rows,
                key=lambda item: (intish(item.get("score_improved")), numeric(item.get("score")) or -math.inf),
                reverse=True,
            )
            if row.get("epoch_id") in by_epoch
        ]
        for row in improved:
            add(row["epoch_id"])
            if len(chosen) >= clips_per_lane:
                break

    if len(chosen) < clips_per_lane:
        for row in rows:
            add(row["epoch_id"])
            if len(chosen) >= clips_per_lane:
                break

    return [by_epoch[item] for item in sorted(chosen, key=int)]


def build_clip(
    epoch: dict[str, str],
    submissions: list[dict[str, str]],
    runs: dict[str, dict[str, str]],
    display_labels: dict[tuple[str, str, str], str],
    media_dir: Path,
    out_dir: Path,
    *,
    max_frames: int,
    frame_duration: float,
    force: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    row = representative_submission(submissions)
    if row is None:
        return None, f"no usable submit for {epoch['model_slug']} {epoch['env_id']} epoch {epoch['epoch_id']}"

    run = runs.get(row["run_id"])
    if not run:
        return None, f"missing run row for {row['run_id']}"

    run_path = resolve_run_path(run)
    submit_index = int(row["submit_index"])
    feedback_dir = run_path / "workspace" / "feedback" / f"submit_{submit_index:03d}"
    artifact = choose_episode(feedback_dir)
    if artifact is None:
        return None, f"no episode observations for {run['run_id']} submit {submit_index:03d}"

    strategy_label = display_labels.get(
        (epoch["env_id"], epoch["model_slug"], epoch["epoch_id"]),
        humanize(epoch["strategy_label"]),
    )
    filename = f"epoch_{int(epoch['epoch_id']):02d}_submit_{submit_index:03d}.gif"
    clip_dir = media_dir / safe_slug(epoch["model_slug"]) / safe_slug(epoch["env_id"])
    clip_dir.mkdir(parents=True, exist_ok=True)
    gif_path = clip_dir / filename

    if force or not gif_path.exists():
        frames = render_clip_frames(
            artifact,
            model=row["model_display"],
            env=row["env_display"],
            strategy=strategy_label,
            submit_index=submit_index,
            accent=MODEL_ACCENTS.get(row["model_slug"], "#2f6fed"),
            max_frames=max_frames,
        )
        if not frames:
            return None, f"could not render frames for {artifact.directory}"
        imageio.mimsave(gif_path, frames, duration=frame_duration, loop=0)

    return {
        "id": f"{row['model_slug']}__{row['env_id']}__epoch_{epoch['epoch_id']}",
        "model_slug": row["model_slug"],
        "model_display": row["model_display"],
        "env_id": row["env_id"],
        "env_display": row["env_display"],
        "category": row["category"],
        "epoch_id": int(epoch["epoch_id"]),
        "strategy_family": epoch["strategy_family"],
        "strategy_label": strategy_label,
        "strategy_source_label": epoch["strategy_label"],
        "dominant_mode": epoch["dominant_mode"],
        "budget_start": intish(epoch["budget_start"]),
        "budget_end": intish(epoch["budget_end"]),
        "submit_index": submit_index,
        "score": numeric(row.get("score")),
        "best_so_far": numeric(row.get("best_so_far")),
        "score_improved": row.get("score_improved") == "1",
        "event_notes": sentence(epoch["event_notes"]),
        "media": str(gif_path.relative_to(out_dir)),
        "run_id": row["run_id"],
        "run_path": run["run_path"],
        "episode_id": artifact.episode_id,
        "episode_return": artifact.episode_return,
        "trajectory_path": str(artifact.trajectory.relative_to(ROOT)),
        "observations_path": observation_ref(artifact.observations),
    }, None


def build_final_clip(
    run: dict[str, str],
    media_dir: Path,
    out_dir: Path,
    *,
    max_frames: int,
    frame_duration: float,
    force: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    submit_index = numeric(run.get("best_submit_index"))
    if submit_index is None:
        return None, f"no best_submit_index for {run['run_id']}"
    submit_index = int(submit_index)
    rerun = find_rerun_capture(run, submit_index, out_dir)
    run_path = resolve_run_path(run)
    feedback_dir = run_path / "workspace" / "feedback" / f"submit_{submit_index:03d}"
    artifact = choose_episode(feedback_dir)
    if artifact is None:
        return None, f"no episode observations for {run['run_id']} final submit {submit_index:03d}"

    final_score = numeric(run.get("final_score"))
    title = "final selected policy"
    filename = f"final_submit_{submit_index:03d}.gif"
    clip_dir = media_dir / safe_slug(run["model_slug"]) / safe_slug(run["env_id"])
    clip_dir.mkdir(parents=True, exist_ok=True)
    gif_path = clip_dir / filename

    if force or not gif_path.exists():
        frames = render_clip_frames(
            artifact,
            model=run["model_display"],
            env=run["env_display"],
            strategy=title,
            submit_index=submit_index,
            accent=MODEL_ACCENTS.get(run["model_slug"], "#2f6fed"),
            max_frames=max_frames,
        )
        if not frames:
            return None, f"could not render frames for {artifact.directory}"
        imageio.mimsave(gif_path, frames, duration=frame_duration, loop=0)

    return {
        "id": f"{run['model_slug']}__{run['env_id']}__final",
        "mode": "final",
        "model_slug": run["model_slug"],
        "model_display": run["model_display"],
        "model_order": intish(run["model_order"]),
        "harness": run["harness"],
        "env_id": run["env_id"],
        "env_display": run["env_display"],
        "env_order": intish(run["env_order"]),
        "category": run["category"],
        "epoch_id": None,
        "strategy_family": "final_policy",
        "strategy_label": title,
        "strategy_source_label": title,
        "dominant_mode": "Final",
        "budget_start": None,
        "budget_end": intish(run["consumed_budget"]),
        "submit_index": submit_index,
        "score": final_score,
        "best_so_far": final_score,
        "score_improved": True,
        "event_notes": final_event_notes(final_score, rerun),
        "media": rerun["media"] if rerun else str(gif_path.relative_to(out_dir)),
        "media_source": "original_env_rerun" if rerun else "recorded_artifact_replay",
        "capture_source": capture_source(run, rerun),
        "run_id": run["run_id"],
        "run_path": relative_or_original(run_path, run["run_path"]),
        "episode_id": artifact.episode_id,
        "episode_return": rerun.get("episode_return") if rerun else artifact.episode_return,
        "rerun_case_index": rerun.get("case_index") if rerun else None,
        "rerun_steps": rerun.get("steps") if rerun else None,
        "trajectory_path": str(artifact.trajectory.relative_to(ROOT)),
        "observations_path": observation_ref(artifact.observations),
    }, None


def representative_submission(rows: list[dict[str, str]]) -> dict[str, str] | None:
    usable = [row for row in rows if row.get("status") == "ok" and row.get("is_non_ok") != "1"]
    if not usable:
        usable = [row for row in rows if row.get("status") == "ok"]
    if not usable:
        return None
    return max(
        usable,
        key=lambda row: (
            intish(row.get("score_improved")),
            numeric(row.get("score")) if numeric(row.get("score")) is not None else -math.inf,
            intish(row.get("submit_index")),
        ),
    )


def find_rerun_capture(
    run: dict[str, str],
    submit_index: int,
    out_dir: Path,
) -> dict[str, Any] | None:
    pattern = (
        out_dir
        / "media"
        / safe_slug(run["model_slug"])
        / safe_slug(run["env_id"])
        / f"rerun_final_submit_{submit_index:03d}_case_*.json"
    )
    for path in sorted(pattern.parent.glob(pattern.name)):
        data = load_json(path)
        if data.get("run_id") != run["run_id"]:
            continue
        media = data.get("media")
        if not isinstance(media, str) or not (out_dir / media).exists():
            continue
        return data
    return None


def final_event_notes(final_score: float | None, rerun: dict[str, Any] | None) -> str:
    if rerun:
        return (
            "Final checkpoint rerun in the original environment; "
            f"case return {format_float(numeric(rerun.get('episode_return')))}."
        )
    return f"Selected checkpoint after visible validation; held-out score {format_float(final_score)}."


def capture_source(run: dict[str, str], rerun: dict[str, Any] | None) -> str | None:
    if not rerun:
        return None
    value = rerun.get("capture_source")
    if isinstance(value, str) and value:
        return value
    env_id = run["env_id"]
    if env_id in {"ant5", "halfcheetah5", "pusher5", "reacher5", "fetch_push", "fetch_pickandplace"}:
        return "direct_mujoco_renderer"
    if env_id in {"parking", "roundabout"}:
        return "highway_state_renderer"
    return "native_env_render"


def choose_episode(feedback_dir: Path) -> EpisodeArtifact | None:
    summary = load_json(feedback_dir / "summary.json")
    returns = summary.get("returns") if isinstance(summary, dict) else None
    first = summary.get("first_global_episode") if isinstance(summary, dict) else None
    return_by_episode: dict[int, float] = {}
    if isinstance(returns, list) and isinstance(first, int):
        for offset, value in enumerate(returns):
            score = numeric(value)
            if score is not None:
                return_by_episode[first + offset] = score

    episodes_dir = feedback_dir / "episodes"
    candidates: list[EpisodeArtifact] = []
    for directory in sorted(episodes_dir.glob("ep_*")):
        if not directory.is_dir():
            continue
        episode_id = parse_episode_id(directory)
        if episode_id is None:
            continue
        trajectory = directory / "trajectory.jsonl"
        observations = observation_path(directory)
        if not trajectory.exists():
            continue
        candidates.append(
            EpisodeArtifact(
                directory=directory,
                trajectory=trajectory,
                observations=observations,
                episode_id=episode_id,
                local_index=max(0, episode_id - first) if isinstance(first, int) else 0,
                episode_return=return_by_episode.get(episode_id),
            )
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.episode_return if item.episode_return is not None else -math.inf,
            -item.episode_id,
        ),
    )


def observation_path(directory: Path) -> Path | None:
    for name in ("observations.npy", "observations.npz"):
        path = directory / name
        if path.exists():
            return path
    return None


def render_clip_frames(
    artifact: EpisodeArtifact,
    *,
    model: str,
    env: str,
    strategy: str,
    submit_index: int,
    accent: str,
    max_frames: int,
) -> list[np.ndarray]:
    rows = read_jsonl(artifact.trajectory)
    observations = load_observations(artifact.observations) if artifact.observations else inline_observations(rows)
    length = min(observations_length(observations), len(rows))
    if length <= 0:
        return []
    indexes = sample_indexes(length, max_frames)
    rewards = [float(row.get("reward") or 0.0) for row in rows[:length]]
    total_rewards = cumulative(rewards)
    actions = [row.get("action") for row in rows[:length]]

    frames: list[np.ndarray] = []
    for frame_no, index in enumerate(indexes):
        canvas = Image.new("RGB", CANVAS, "#f7f4ee")
        draw = ImageDraw.Draw(canvas)
        draw_background(draw, accent)
        draw_header(
            draw,
            model=model,
            env=env,
            strategy=strategy,
            submit_index=submit_index,
            episode_id=artifact.episode_id,
            accent=accent,
        )
        draw_visual(canvas, draw, observations, index, rows[index], accent)
        draw_telemetry(
            draw,
            index=index,
            length=length,
            reward=rewards[index],
            total=total_rewards[index],
            final_total=total_rewards[-1],
            action=actions[index],
            rewards=rewards,
            accent=accent,
            frame_no=frame_no,
            n_frames=len(indexes),
        )
        frames.append(np.asarray(canvas))
    return frames


def load_observations(path: Path) -> dict[str, np.ndarray] | np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    arrays = np.load(path, allow_pickle=False)
    return {key: arrays[key] for key in arrays.files}


def inline_observations(rows: list[dict[str, Any]]) -> dict[str, np.ndarray] | np.ndarray:
    raw = [row.get("obs") for row in rows]
    dict_values = [item for item in raw if isinstance(item, dict)]
    if dict_values and len(dict_values) == len(raw):
        arrays: dict[str, np.ndarray] = {}
        keys = sorted({key for item in dict_values for key in item if key != "mission"})
        for key in keys:
            values = [item.get(key) for item in dict_values]
            try:
                arrays[str(key)] = np.asarray(values)
            except ValueError:
                flattened = [flatten_numbers(value) for value in values]
                width = max((len(row) for row in flattened), default=0)
                if width:
                    array = np.zeros((len(flattened), width), dtype=float)
                    for row_index, row in enumerate(flattened):
                        array[row_index, : len(row)] = row
                    arrays[str(key)] = array
        if arrays:
            return arrays

    values = [flatten_numbers(item) for item in raw]
    width = max((len(row) for row in values), default=0)
    if width == 0:
        return np.empty((0, 0), dtype=float)
    array = np.zeros((len(values), width), dtype=float)
    for row_index, row in enumerate(values):
        if row:
            array[row_index, : len(row)] = row
    return array


def observations_length(observations: dict[str, np.ndarray] | np.ndarray) -> int:
    if isinstance(observations, np.ndarray):
        return int(observations.shape[0]) if observations.ndim else 0
    if not observations:
        return 0
    return min(int(array.shape[0]) for array in observations.values() if array.ndim)


def observation_ref(path: Path | None) -> str:
    return str(path.relative_to(ROOT)) if path is not None else "trajectory.jsonl inline obs"


def draw_background(draw: ImageDraw.ImageDraw, accent: str) -> None:
    draw.rectangle((0, 0, CANVAS[0], CANVAS[1]), fill="#f7f4ee")
    draw.rectangle((0, 0, CANVAS[0], 64), fill="#1e2528")
    draw.rectangle((0, CANVAS[1] - 16, CANVAS[0], CANVAS[1]), fill=accent)
    draw.rounded_rectangle(VISUAL_BOX, radius=8, fill="#111819", outline="#d0c9bd", width=1)


def draw_header(
    draw: ImageDraw.ImageDraw,
    *,
    model: str,
    env: str,
    strategy: str,
    submit_index: int,
    episode_id: int,
    accent: str,
) -> None:
    draw.text((24, 14), f"{model} on {env}", fill="#ffffff", font=font(20, bold=True))
    draw.text(
        (24, 42),
        f"{strategy} | submit {submit_index:03d} | ep {episode_id:03d}",
        fill="#e8dfd1",
        font=font(11),
    )
    draw.rounded_rectangle((610, 14, 690, 36), radius=6, fill=accent)
    draw.text((624, 19), "REAL RUN", fill="#ffffff", font=font(10, bold=True))


def draw_visual(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    observations: dict[str, np.ndarray] | np.ndarray,
    index: int,
    row: dict[str, Any],
    accent: str,
) -> None:
    frame = observation_frame(observations, index)
    if frame is None:
        draw_numeric_panel(draw, observations, index, accent)
        return
    if is_minigrid_frame(frame):
        draw_minigrid(draw, frame, direction_value(observations, index), accent)
        return
    if is_rgb_frame(frame):
        paste_rgb_frame(canvas, frame)
        return
    draw_numeric_panel(draw, observations, index, accent)


def observation_frame(
    observations: dict[str, np.ndarray] | np.ndarray,
    index: int,
) -> np.ndarray | None:
    if isinstance(observations, np.ndarray):
        return observations[index]
    if "image" in observations:
        return observations["image"][index]
    for value in observations.values():
        if value.ndim >= 4 and value.shape[-1] in (1, 3, 4):
            return value[index]
    return None


def is_rgb_frame(frame: np.ndarray) -> bool:
    return frame.ndim == 3 and frame.shape[-1] in (3, 4) and max(frame.shape[:2]) > 16


def is_minigrid_frame(frame: np.ndarray) -> bool:
    return frame.ndim == 3 and frame.shape[-1] == 3 and max(frame.shape[:2]) <= 16


def paste_rgb_frame(canvas: Image.Image, frame: np.ndarray) -> None:
    array = frame
    if array.dtype != np.uint8:
        array = normalize_to_uint8(array)
    image = Image.fromarray(array[..., :3])
    x0, y0, x1, y1 = VISUAL_BOX
    max_w = x1 - x0 - 20
    max_h = y1 - y0 - 20
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resample = Image.Resampling.NEAREST if scale > 1.5 else Image.Resampling.BILINEAR
    image = image.resize(size, resample)
    px = x0 + (x1 - x0 - image.width) // 2
    py = y0 + (y1 - y0 - image.height) // 2
    canvas.paste(image, (px, py))


def draw_minigrid(
    draw: ImageDraw.ImageDraw,
    frame: np.ndarray,
    direction: int | None,
    accent: str,
) -> None:
    x0, y0, x1, y1 = VISUAL_BOX
    grid_h, grid_w = frame.shape[:2]
    tile = min((x1 - x0 - 48) // grid_w, (y1 - y0 - 48) // grid_h)
    start_x = x0 + (x1 - x0 - tile * grid_w) // 2
    start_y = y0 + (y1 - y0 - tile * grid_h) // 2
    object_colors = {
        0: "#263238",
        1: "#6b7280",
        2: "#d7c9aa",
        3: "#7a8c99",
        4: "#c0843c",
        5: "#f4c542",
        6: "#3f8c5f",
        7: "#d45454",
        8: "#5d73c5",
        9: "#8e63c7",
        10: "#2f6fed",
    }
    for row in range(grid_h):
        for col in range(grid_w):
            obj = int(frame[row, col, 0])
            color = object_colors.get(obj, accent)
            xx = start_x + col * tile
            yy = start_y + row * tile
            draw.rectangle((xx, yy, xx + tile - 2, yy + tile - 2), fill=color, outline="#111819")
            state = int(frame[row, col, 2])
            if state:
                draw.rectangle((xx + 4, yy + 4, xx + tile - 6, yy + tile - 6), outline="#ffffff", width=1)
    cx = start_x + (grid_w // 2) * tile + tile // 2
    cy = start_y + (grid_h // 2) * tile + tile // 2
    arrow = direction_arrow(direction)
    draw.text((cx - 8, cy - 10), arrow, fill="#ffffff", font=font(18, bold=True))
    draw.text((x0 + 18, y1 - 28), "recorded MiniGrid partial observation", fill="#c9d0d0", font=font(11))


def draw_numeric_panel(
    draw: ImageDraw.ImageDraw,
    observations: dict[str, np.ndarray] | np.ndarray,
    index: int,
    accent: str,
) -> None:
    x0, y0, x1, y1 = VISUAL_BOX
    draw.text((x0 + 18, y0 + 16), "recorded state trajectory", fill="#ffffff", font=font(18, bold=True))
    series = numeric_series(observations)
    colors = [accent, "#c85f32", "#1f9a72", "#444b54"]
    plot_box = (x0 + 22, y0 + 62, x1 - 22, y1 - 48)
    draw.rectangle(plot_box, fill="#f7f4ee", outline="#364247")
    if series.size:
        for col in range(min(series.shape[1], 4)):
            values = series[:, col]
            points = line_points(values, plot_box)
            if len(points) >= 2:
                draw.line(points, fill=colors[col], width=2)
        px = plot_box[0] + int((plot_box[2] - plot_box[0]) * (index / max(1, series.shape[0] - 1)))
        draw.line((px, plot_box[1], px, plot_box[3]), fill="#111819", width=2)
    draw.text((x0 + 22, y1 - 32), "state dims 0-3 from this rollout", fill="#c9d0d0", font=font(11))


def numeric_series(observations: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(observations, np.ndarray):
        array = observations
    else:
        arrays = [value for value in observations.values() if value.dtype.kind in "iufb" and value.ndim >= 2]
        if not arrays:
            return np.empty((0, 0))
        array = arrays[0]
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.asarray(array, dtype=float)


def direction_value(observations: dict[str, np.ndarray] | np.ndarray, index: int) -> int | None:
    if isinstance(observations, dict) and "direction" in observations:
        return int(observations["direction"][index])
    return None


def direction_arrow(direction: int | None) -> str:
    return {0: ">", 1: "v", 2: "<", 3: "^"}.get(direction, "+")


def draw_telemetry(
    draw: ImageDraw.ImageDraw,
    *,
    index: int,
    length: int,
    reward: float,
    total: float,
    final_total: float,
    action: Any,
    rewards: list[float],
    accent: str,
    frame_no: int,
    n_frames: int,
) -> None:
    x = TELEMETRY_X
    draw.text((x, 78), "rollout telemetry", fill="#1e2528", font=font(18, bold=True))
    metrics = [
        ("step", f"{index + 1}/{length}"),
        ("reward", f"{reward:+.3f}"),
        ("return", f"{total:+.2f}"),
        ("final", f"{final_total:+.2f}"),
    ]
    y = 112
    for label, value in metrics:
        draw.text((x, y), label.upper(), fill="#6a706f", font=font(9, bold=True))
        draw.text((x + 72, y - 4), value, fill="#1e2528", font=font(17, bold=True))
        y += 34

    draw.text((x, y + 4), "ACTION", fill="#6a706f", font=font(9, bold=True))
    draw_action(draw, action, (x, y + 24), accent)
    y += 100

    draw.text((x, y), "REWARD TRACE", fill="#6a706f", font=font(9, bold=True))
    plot_box = (x, y + 20, CANVAS[0] - 28, y + 78)
    draw.rectangle(plot_box, fill="#ffffff", outline="#d0c9bd")
    points = line_points(np.asarray(rewards, dtype=float), plot_box)
    if len(points) >= 2:
        draw.line(points, fill=accent, width=2)
    px = plot_box[0] + int((plot_box[2] - plot_box[0]) * (index / max(1, length - 1)))
    draw.line((px, plot_box[1], px, plot_box[3]), fill="#1e2528", width=2)

    progress = (CANVAS[0] - 56) * ((frame_no + 1) / max(1, n_frames))
    draw.rectangle((28, CANVAS[1] - 12, 28 + progress, CANVAS[1] - 8), fill="#ffffff")


def draw_action(draw: ImageDraw.ImageDraw, action: Any, origin: tuple[int, int], accent: str) -> None:
    values = flatten_numbers(action)[:5]
    if not values:
        draw.text(origin, str(action)[:30], fill="#1e2528", font=font(12))
        return
    x, y = origin
    for idx, value in enumerate(values):
        yy = y + idx * 14
        draw.text((x, yy - 2), f"a{idx}", fill="#6a706f", font=font(9))
        draw.rectangle((x + 24, yy, x + 174, yy + 8), fill="#e5ded2")
        clipped = max(-1.0, min(1.0, float(value)))
        center = x + 99
        draw.line((center, yy, center, yy + 8), fill="#9f9689")
        if clipped >= 0:
            draw.rectangle((center, yy, center + int(75 * clipped), yy + 8), fill=accent)
        else:
            draw.rectangle((center + int(75 * clipped), yy, center, yy + 8), fill=accent)


def line_points(values: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    x0, y0, x1, y1 = box
    step = max(1, len(values) // 220)
    points = []
    for i in range(0, len(values), step):
        value = max(lo, min(hi, float(values[i])))
        x = x0 + int((x1 - x0) * (i / max(1, len(values) - 1)))
        y = y1 - int((y1 - y0) * ((value - lo) / (hi - lo)))
        points.append((x, y))
    return points


def build_payload(
    *,
    runs: list[dict[str, str]],
    leaderboard: list[dict[str, str]],
    epochs: list[dict[str, str]],
    clips: list[dict[str, Any]],
    warnings: list[str],
    mode: str,
    all_strategies: bool,
    clips_per_lane: int,
) -> dict[str, Any]:
    strategy_envs = sorted({row["env_id"] for row in epochs})
    model_rows = {}
    env_rows = {}
    for row in runs:
        model_rows[row["model_slug"]] = {
            "slug": row["model_slug"],
            "display": row["model_display"],
            "harness": row["harness"],
            "order": intish(row["model_order"]),
        }
        env_rows[row["env_id"]] = {
            "id": row["env_id"],
            "display": row["env_display"],
            "category": row["category"],
            "order": intish(row["env_order"]),
            "strategy_focus": row["env_id"] in strategy_envs,
        }
    final_by_lane = {
        (row["model_slug"], row["env_id"]): {
            "score": numeric(row["final_score"]),
            "best_submit_index": intish(row["best_submit_index"]),
            "run_id": row["run_id"],
            "run_path": row["run_path"],
        }
        for row in runs
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "clip_policy": clip_policy(mode, all_strategies, clips_per_lane),
        "models": sorted(model_rows.values(), key=lambda row: row["order"]),
        "envs": sorted(env_rows.values(), key=lambda row: row["order"]),
        "strategy_envs": strategy_envs,
        "clips": sorted(clips, key=lambda row: (row["env_id"], row["model_slug"], row["epoch_id"])),
        "final_results": [
            {
                "model_slug": model,
                "env_id": env,
                **data,
            }
            for (model, env), data in sorted(final_by_lane.items())
        ],
        "leaderboard": normalize_leaderboard(leaderboard),
        "warnings": warnings,
    }


def clip_policy(mode: str, all_strategies: bool, clips_per_lane: int) -> str:
    if mode == "final":
        return "one clip per model-env lane from the run's selected final checkpoint"
    if all_strategies:
        return "all audited strategy epochs"
    return f"{clips_per_lane} representative strategy epochs per model-env lane"


def normalize_leaderboard(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "model_slug": row["model_slug"],
                "model_display": row["model_display"],
                "harness": row["harness"],
                "scores": {
                    key: numeric(value)
                    for key, value in row.items()
                    if key not in {"model_slug", "model_display", "harness"}
                },
            }
        )
    return normalized


def relative_or_original(path: Path, fallback: str) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return fallback


def resolve_run_path(run: dict[str, str]) -> Path:
    recorded = ROOT / run["run_path"]
    if recorded.exists():
        return recorded
    matches = sorted((ROOT / "runs" / "main-128").glob(f"**/{run['run_id']}"))
    for match in matches:
        if (match / "run.json").exists() or (match / "workspace" / "feedback").exists():
            return match
    return recorded


def write_data_js(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(f"window.SHOWCASE_DATA = {body};\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def sample_indexes(length: int, max_frames: int) -> list[int]:
    count = max(1, min(length, max_frames))
    if count == 1:
        return [0]
    return sorted({int(round(i * (length - 1) / (count - 1))) for i in range(count)})


def cumulative(values: list[float]) -> list[float]:
    total = 0.0
    out = []
    for value in values:
        total += value
        out.append(total)
    return out


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(finite.min())
    hi = float(finite.max())
    if math.isclose(lo, hi):
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    if isinstance(value, dict):
        out: list[float] = []
        for key in sorted(value):
            out.extend(flatten_numbers(value[key]))
        return out
    return []


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def intish(value: Any) -> int:
    score = numeric(value)
    return int(score) if score is not None else 0


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def parse_episode_id(path: Path) -> int | None:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def humanize(value: str) -> str:
    return value.replace("_", " ").strip()


def sentence(value: str) -> str:
    text = value.replace("_", " ").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
