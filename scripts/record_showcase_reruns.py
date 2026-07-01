#!/usr/bin/env python3
"""Record final policies by rerunning them in the original Gymnasium envs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from evopolicygym import Budget, Pool, PoolKind, Run, SubmitRecord
from evopolicygym.data import load as load_data
from evopolicygym.envs import registry
from evopolicygym.envs.gym.minigrid_assets import patch_minigrid_wfc_assets
from evopolicygym.envs.gym import space
from evopolicygym.envs.gym.world import _info, _reward, _seed


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figs" / "data"
CASE_ROOT = ROOT / "data" / "main-128"
OUT_DIR = ROOT / "web"
CANVAS = (720, 405)

MODEL_ACCENTS = {
    "gpt_5_5": "#2f6fed",
    "claude_opus_4_7": "#c85f32",
    "minimax_m3": "#1f9a72",
    "deepseek_v4_pro": "#7a4bd6",
}


@dataclass(frozen=True)
class FrameRecord:
    image: np.ndarray
    step: int
    reward: float
    total: float
    action: Any
    done: bool


@dataclass(frozen=True)
class EpisodeResult:
    frames: tuple[FrameRecord, ...]
    total: float
    steps: int
    terminated: bool
    truncated: bool
    error: str | None
    stdout: str
    stderr: str
    capture_source: str


def main() -> None:
    args = parse_args()
    rows = select_rows(read_csv(DATA_DIR / "runs.csv"), args)
    out_dir = Path(args.out)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        try:
            result = record_row(
                row,
                media_dir=media_dir,
                out_dir=out_dir,
                pool_kind=PoolKind(args.pool),
                case_index=args.case_index,
                max_steps=args.max_steps,
                max_frames=args.max_frames,
                frame_duration=args.frame_duration,
                force=args.force,
            )
            results.append(result)
            print(
                f"ok {row['model_slug']} {row['env_id']} submit {intish(row['best_submit_index']):03d} "
                f"case {args.case_index:03d} return {result['episode_return']:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - batch recording should continue.
            warning = f"{row['model_slug']} {row['env_id']}: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            print(f"warn {warning}", file=sys.stderr, flush=True)

    summary = {
        "schema_version": "0.1",
        "generated_at": now_iso(),
        "pool": args.pool,
        "case_index": args.case_index,
        "results": results,
        "warnings": warnings,
    }
    summary_path = out_dir / "data" / "rerun-captures.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(results)} rerun captures to {media_dir.relative_to(ROOT)}")
    print(f"wrote summary to {summary_path.relative_to(ROOT)}")
    if warnings:
        print(f"warnings: {len(warnings)}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR), help="showcase output directory")
    parser.add_argument("--env", action="append", default=[], help="env_id filter, repeatable")
    parser.add_argument("--model", action="append", default=[], help="model_slug filter, repeatable")
    parser.add_argument("--run-id", action="append", default=[], help="run_id filter, repeatable")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of selected lanes")
    parser.add_argument("--pool", choices=("train", "valid", "final"), default="final")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="cap episode steps; 0 uses env limit")
    parser.add_argument("--max-frames", type=int, default=56)
    parser.add_argument("--frame-duration", type=float, default=0.16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("run_status") == "completed"
        and row.get("best_submit_index") not in {"", None}
        and (not args.env or row["env_id"] in set(args.env))
        and (not args.model or row["model_slug"] in set(args.model))
        and (not args.run_id or row["run_id"] in set(args.run_id))
    ]
    selected.sort(key=lambda row: (intish(row["env_order"]), intish(row["model_order"])))
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    return selected


def record_row(
    row: dict[str, str],
    *,
    media_dir: Path,
    out_dir: Path,
    pool_kind: PoolKind,
    case_index: int,
    max_steps: int,
    max_frames: int,
    frame_duration: float,
    force: bool,
) -> dict[str, Any]:
    submit_index = intish(row["best_submit_index"])
    run_path = resolve_run_path(row)
    checkpoint = run_path / "checkpoints" / f"submit_{submit_index:03d}"
    policy_path = checkpoint / "policy.py"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    run_json = load_json(run_path / "run.json")
    env_name = run_json.get("env") or row["suite_id"]
    env = registry(bulk=env_name.startswith("gymnasium/"), filters=(env_name,)).get(env_name)
    pool = pool_for(env_name, env, pool_kind)
    if not pool.contains(case_index):
        raise ValueError(f"case {case_index} outside {pool_kind.value} pool of size {pool.size}")

    clip_dir = media_dir / safe_slug(row["model_slug"]) / safe_slug(row["env_id"])
    clip_dir.mkdir(parents=True, exist_ok=True)
    gif_path = clip_dir / f"rerun_final_submit_{submit_index:03d}_case_{case_index:03d}.gif"
    meta_path = gif_path.with_suffix(".json")

    if force or not gif_path.exists():
        policy = load_policy(checkpoint)
        task = env.task
        submit = SubmitRecord(index=submit_index, cases=(case_index,))
        run = Run(
            key=row["run_id"],
            model=row["model_display"],
            env=task.name,
            exp=row.get("suite_id", ""),
            protocol="protocol/v2.0-draft",
            budget=Budget(limit=intish(row.get("budget_limit")), used=intish(row.get("consumed_budget"))),
        )
        instance = instantiate_policy(policy, checkpoint, task, pool, run, submit)
        world = RecordingGym(env.make().spec)
        try:
            episode = run_episode(
                world,
                instance,
                task=task,
                pool=pool,
                case_index=case_index,
                max_steps=max_steps or task.steps,
            )
        finally:
            world.close()
        if not episode.frames:
            raise RuntimeError("environment produced no render frames")
        frames = compose_frames(
            episode.frames,
            model=row["model_display"],
            env=row["env_display"],
            submit_index=submit_index,
            case_index=case_index,
            score=numeric(row.get("final_score")),
            accent=MODEL_ACCENTS.get(row["model_slug"], "#2f6fed"),
            max_frames=max_frames,
        )
        imageio.mimsave(gif_path, frames, duration=frame_duration, loop=0)
        meta_path.write_text(
            json.dumps(
                {
                    "schema_version": "0.1",
                    "run_id": row["run_id"],
                    "model_slug": row["model_slug"],
                    "env_id": row["env_id"],
                    "env_name": env_name,
                    "submit_index": submit_index,
                    "pool": pool_kind.value,
                    "case_index": case_index,
                    "episode_return": episode.total,
                    "steps": episode.steps,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                    "error": episode.error,
                    "capture_source": episode.capture_source,
                    "stdout": episode.stdout[-4000:],
                    "stderr": episode.stderr[-4000:],
                    "media": str(gif_path.relative_to(out_dir)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        episode_meta = load_json(meta_path)
        episode = EpisodeResult(
            frames=(),
            total=float(episode_meta.get("episode_return", 0.0)),
            steps=int(episode_meta.get("steps", 0)),
            terminated=bool(episode_meta.get("terminated")),
            truncated=bool(episode_meta.get("truncated")),
            error=episode_meta.get("error"),
            stdout="",
            stderr="",
            capture_source=str(episode_meta.get("capture_source", "unknown")),
        )

    return {
        "run_id": row["run_id"],
        "model_slug": row["model_slug"],
        "model_display": row["model_display"],
        "env_id": row["env_id"],
        "env_display": row["env_display"],
        "submit_index": submit_index,
        "pool": pool_kind.value,
        "case_index": case_index,
        "episode_return": episode.total,
        "steps": episode.steps,
        "error": episode.error,
        "capture_source": episode.capture_source,
        "media": str(gif_path.relative_to(out_dir)),
        "metadata": str(meta_path.relative_to(out_dir)),
    }


def pool_for(env_name: str, env: Any, kind: PoolKind) -> Pool:
    data_root = CASE_ROOT / env_name
    if data_root.exists():
        return load_data(data_root, env=env_name).pool(kind)
    return env.pool(kind)


def load_policy(system: Path) -> type:
    policy_path = system / "policy.py"
    name = f"_evopolicygym_showcase_{abs(hash(system))}"
    spec = importlib.util.spec_from_file_location(name, policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(policy_path)
    module = importlib.util.module_from_spec(spec)
    with project(system):
        purge_helpers(system)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    policy = getattr(module, "Policy", None)
    if not isinstance(policy, type):
        raise AttributeError("policy.py must define class Policy")
    return policy


def instantiate_policy(
    cls: type,
    system: Path,
    task: Any,
    pool: Pool,
    run: Run,
    submit: SubmitRecord,
) -> object:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with project(system), redirect_stdout(stdout), redirect_stderr(stderr):
        return cls(task.obs, task.act, meta(run, submit, task, pool))


def meta(run: Run, submit: SubmitRecord, task: Any, pool: Pool) -> dict[str, Any]:
    left = max(0, run.budget.left - submit.cost) if pool.kind == PoolKind.train else 0
    body: dict[str, Any] = {
        "env": task.name,
        "submit_index": submit.index,
        "n_episodes_this_submit": len(submit.cases),
        "remaining_budget_after": left,
        "max_episode_steps": task.steps,
        "obs_space": task.obs,
        "action_space": task.act,
        "obs_storage": task.storage,
    }
    if task.rewards:
        body["reward_components"] = task.rewards
    return body


def run_episode(
    world: "RecordingGym",
    policy: object,
    *,
    task: Any,
    pool: Pool,
    case_index: int,
    max_steps: int,
) -> EpisodeResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    frames: list[FrameRecord] = []
    capture_source = "none"
    total = 0.0
    terminated = False
    truncated = False
    error: str | None = None
    action: Any = None

    with redirect_stdout(stdout), redirect_stderr(stderr):
        policy.reset(0)
    obs = world.reset(pool.case(case_index))
    frame, source = world.capture(obs, action=None, step=0, reward=0.0, total=0.0)
    if frame is not None:
        capture_source = source
        frames.append(FrameRecord(frame, 0, 0.0, 0.0, None, False))

    for step in range(max_steps):
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                action = policy.act(obs)
        except Exception as exc:  # noqa: BLE001 - mirror benchmark fallback behavior.
            action = world.sample()
            error = f"act_error: {type(exc).__name__}: {exc}"

        obs, reward, terminated, truncated, info = world.step(action)
        total += reward
        frame, source = world.capture(obs, action=action, step=step + 1, reward=reward, total=total)
        if frame is not None:
            if capture_source == "none":
                capture_source = source
            frames.append(
                FrameRecord(
                    frame,
                    step + 1,
                    reward,
                    total,
                    action,
                    bool(terminated or truncated or error),
                )
            )
        if terminated or truncated or error:
            break

    return EpisodeResult(
        frames=tuple(frames),
        total=total,
        steps=max(0, len(frames) - 1),
        terminated=terminated,
        truncated=truncated,
        error=error,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        capture_source=capture_source,
    )


class RecordingGym:
    def __init__(self, spec: Any):
        self.spec = spec
        self.env: Any | None = None
        self.renderer: Any | None = None
        self.obs: Any = None
        self.done = False
        self.elapsed = 0
        self.use_native_render = native_render_supported(str(spec.id))
        self.use_mujoco_renderer = mujoco_renderer_supported(str(spec.id))

    def reset(self, case: Any) -> Any:
        env = self.make()
        seed = _seed(case)
        with suppress(Exception):
            env.action_space.seed(seed + 1)
        with suppress(Exception):
            env.observation_space.seed(seed + 2)
        options = case.data.get("options")
        kwargs: dict[str, Any] = {"seed": seed}
        if isinstance(options, Mapping):
            kwargs["options"] = dict(options)
        obs, _info_body = env.reset(**kwargs)
        self.obs = space.encode(obs)
        self.done = False
        self.elapsed = 0
        return self.obs

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        env = self.make()
        raw, invalid = space.action(env.action_space, action)
        obs, reward, terminated, truncated, info = env.step(raw)
        self.obs = space.encode(obs)
        self.elapsed += 1
        self.done = bool(terminated or truncated)
        body = _info(info)
        body["steps"] = self.elapsed
        scalar_reward, raw_reward = _reward(reward)
        if raw_reward is not None:
            body["gym_reward"] = raw_reward
            body["reward_scalarization"] = "sum"
        if invalid:
            body["action_invalid"] = True
        return self.obs, scalar_reward, bool(terminated), bool(truncated), body

    def capture(
        self,
        obs: Any,
        *,
        action: Any,
        step: int,
        reward: float,
        total: float,
    ) -> tuple[np.ndarray | None, str]:
        if highway_render_supported(str(self.spec.id)):
            frame = highway_frame(self.make())
            if frame is not None:
                return frame, "highway_state_renderer"
        if self.use_mujoco_renderer:
            frame = self.mujoco_frame()
            normalized = normalize_frame(frame)
            if normalized is not None:
                return normalized, "direct_mujoco_renderer"
        if self.use_native_render:
            try:
                frame = self.make().render()
            except Exception:
                frame = None
            normalized = normalize_frame(frame)
            if normalized is not None:
                return normalized, "native_env_render"
        return (
            state_frame(
                obs,
                action=action,
                step=step,
                reward=reward,
                total=total,
                title=str(self.spec.id),
            ),
            "state_capture",
        )

    def sample(self) -> Any:
        return space.sample(self.make().action_space)

    def close(self) -> None:
        if self.renderer is not None:
            with suppress(Exception):
                self.renderer.close()
            self.renderer = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def make(self) -> Any:
        if self.env is None:
            import gymnasium

            patch_minigrid_wfc_assets(self.spec.id)
            kwargs = dict(self.spec.kwargs)
            if highway_render_supported(str(self.spec.id)):
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            if self.use_native_render:
                kwargs.setdefault("render_mode", "rgb_array")
            self.env = gymnasium.make(self.spec.id, **kwargs)
        return self.env

    def mujoco_frame(self) -> np.ndarray | None:
        env = self.make()
        base = env.unwrapped
        if self.renderer is None:
            try:
                import mujoco

                self.renderer = mujoco.Renderer(base.model, height=360, width=480)
            except Exception:
                return None
        try:
            self.renderer.update_scene(base.data)
            return self.renderer.render()
        except Exception:
            return None


def normalize_frame(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, list | tuple):
        if not frame:
            return None
        frame = frame[-1]
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[:, :, :3])


def native_render_supported(env_id: str) -> bool:
    blocked = (
        "Ant-",
        "HalfCheetah-",
        "Pusher-",
        "Reacher-",
        "FetchPush",
        "FetchPickAndPlace",
        "parking-",
        "roundabout-",
    )
    return not any(item in env_id for item in blocked)


def mujoco_renderer_supported(env_id: str) -> bool:
    return any(
        item in env_id
        for item in ("Ant-", "HalfCheetah-", "Pusher-", "Reacher-", "FetchPush", "FetchPickAndPlace")
    )


def highway_render_supported(env_id: str) -> bool:
    return "parking-" in env_id or "roundabout-" in env_id


def highway_frame(env: Any) -> np.ndarray | None:
    base = env.unwrapped
    road = getattr(base, "road", None)
    if road is None:
        return None

    lane_polylines = highway_lane_polylines(road)
    vehicles = list(getattr(road, "vehicles", []))
    points: list[tuple[float, float]] = []
    for line in lane_polylines:
        points.extend((float(x), float(y)) for x, y in line)
    for vehicle in vehicles:
        pos = np.asarray(getattr(vehicle, "position", []), dtype=float)
        if pos.shape == (2,):
            points.append((float(pos[0]), float(pos[1])))
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = 12.0
    bounds = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
    canvas = Image.new("RGB", CANVAS, "#eef3f0")
    draw = ImageDraw.Draw(canvas)
    project = highway_projector(bounds)

    for line in lane_polylines:
        pts = [project(point) for point in line]
        if len(pts) >= 2:
            draw.line(pts, fill="#67736f", width=8, joint="curve")
            draw.line(pts, fill="#f7faf8", width=3, joint="curve")

    for vehicle in vehicles:
        draw_vehicle(draw, vehicle, project, is_controlled=vehicle is getattr(base, "vehicle", None))

    draw.text((22, 18), str(getattr(base, "config", {}).get("observation", {}).get("type", "highway-env")), fill="#22302d", font=font(15, bold=True))
    return np.asarray(canvas)


def highway_lane_polylines(road: Any) -> list[list[tuple[float, float]]]:
    graph = getattr(getattr(road, "network", None), "graph", {})
    lines: list[list[tuple[float, float]]] = []
    if not isinstance(graph, Mapping):
        return lines
    for dests in graph.values():
        if not isinstance(dests, Mapping):
            continue
        for lanes in dests.values():
            for lane in lanes:
                length = float(getattr(lane, "length", 0.0) or 0.0)
                if length <= 0:
                    start = np.asarray(getattr(lane, "start", []), dtype=float)
                    end = np.asarray(getattr(lane, "end", []), dtype=float)
                    if start.shape == (2,) and end.shape == (2,):
                        lines.append([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))])
                    continue
                count = max(8, min(80, int(length / 2)))
                line = []
                for longitudinal in np.linspace(0.0, length, count):
                    with suppress(Exception):
                        pos = np.asarray(lane.position(float(longitudinal), 0.0), dtype=float)
                        if pos.shape == (2,):
                            line.append((float(pos[0]), float(pos[1])))
                if len(line) >= 2:
                    lines.append(line)
    return lines


def highway_projector(bounds: tuple[float, float, float, float]):
    min_x, min_y, max_x, max_y = bounds
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    pad_x, pad_y = 44, 50
    scale = min((CANVAS[0] - 2 * pad_x) / width, (CANVAS[1] - 2 * pad_y) / height)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    def project(point: Any) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        px = CANVAS[0] / 2 + (x - cx) * scale
        py = CANVAS[1] / 2 - (y - cy) * scale
        return int(round(px)), int(round(py))

    return project


def draw_vehicle(draw: ImageDraw.ImageDraw, vehicle: Any, project: Any, *, is_controlled: bool) -> None:
    pos = np.asarray(getattr(vehicle, "position", []), dtype=float)
    if pos.shape != (2,):
        return
    heading = float(getattr(vehicle, "heading", 0.0) or 0.0)
    length = float(getattr(vehicle, "LENGTH", 5.0) or 5.0)
    width = float(getattr(vehicle, "WIDTH", 2.0) or 2.0)
    corners = vehicle_corners(pos, heading, length, width)
    polygon = [project(point) for point in corners]
    color = "#2f6fed" if is_controlled else "#f1a13a"
    outline = "#0d1719" if is_controlled else "#6a3c13"
    draw.polygon(polygon, fill=color, outline=outline)
    nose = np.asarray(pos) + np.asarray([np.cos(heading), np.sin(heading)]) * length * 0.62
    draw.line((project(pos), project(nose)), fill="#ffffff", width=2)


def vehicle_corners(pos: np.ndarray, heading: float, length: float, width: float) -> list[np.ndarray]:
    forward = np.asarray([np.cos(heading), np.sin(heading)])
    side = np.asarray([-np.sin(heading), np.cos(heading)])
    return [
        pos + forward * length / 2 + side * width / 2,
        pos + forward * length / 2 - side * width / 2,
        pos - forward * length / 2 - side * width / 2,
        pos - forward * length / 2 + side * width / 2,
    ]


def state_frame(
    obs: Any,
    *,
    action: Any,
    step: int,
    reward: float,
    total: float,
    title: str,
) -> np.ndarray:
    visual = observation_visual(obs)
    if visual is not None:
        return visual

    canvas = Image.new("RGB", CANVAS, "#111819")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CANVAS[0], CANVAS[1]), fill="#111819")
    draw.text((24, 22), title, fill="#ffffff", font=font(20, bold=True))
    draw.text((24, 50), "original environment state capture", fill="#c8d2d0", font=font(12))

    values = flatten_numbers(obs)
    action_values = flatten_numbers(action)
    draw_state_bars(draw, values, box=(32, 92, 462, 332))
    draw_action_bars(draw, action_values, box=(500, 112, 684, 238))

    draw.text((500, 264), "STEP", fill="#8fa09d", font=font(10, bold=True))
    draw.text((580, 260), f"{step:04d}", fill="#ffffff", font=font(18, bold=True))
    draw.text((500, 294), "REWARD", fill="#8fa09d", font=font(10, bold=True))
    draw.text((580, 290), f"{reward:+.3f}", fill="#ffffff", font=font(18, bold=True))
    draw.text((500, 324), "RETURN", fill="#8fa09d", font=font(10, bold=True))
    draw.text((580, 320), f"{total:+.3f}", fill="#ffffff", font=font(18, bold=True))
    return np.asarray(canvas)


def observation_visual(obs: Any) -> np.ndarray | None:
    if isinstance(obs, Mapping):
        if "image" in obs:
            array = np.asarray(obs["image"])
            if array.ndim == 3 and array.shape[-1] == 3:
                if max(array.shape[:2]) <= 32:
                    return minigrid_frame(array, direction=obs.get("direction"))
                return fit_observation_image(array)
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                continue
        return None
    array = np.asarray(obs)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        return fit_observation_image(array)
    return None


def fit_observation_image(array: np.ndarray) -> np.ndarray:
    frame = normalize_frame(array)
    if frame is None:
        return state_frame([], action=None, step=0, reward=0.0, total=0.0, title="state")
    image = Image.fromarray(frame).convert("RGB")
    image = ImageOps.contain(image, CANVAS, method=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", CANVAS, "#111819")
    canvas.paste(image, ((CANVAS[0] - image.width) // 2, (CANVAS[1] - image.height) // 2))
    return np.asarray(canvas)


def minigrid_frame(array: np.ndarray, *, direction: Any) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS, "#111819")
    draw = ImageDraw.Draw(canvas)
    grid_h, grid_w = array.shape[:2]
    tile = min((CANVAS[0] - 96) // grid_w, (CANVAS[1] - 96) // grid_h)
    start_x = (CANVAS[0] - tile * grid_w) // 2
    start_y = (CANVAS[1] - tile * grid_h) // 2
    colors = {
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
            obj = int(array[row, col, 0])
            color = colors.get(obj, "#2f6fed")
            x = start_x + col * tile
            y = start_y + row * tile
            draw.rectangle((x, y, x + tile - 2, y + tile - 2), fill=color, outline="#111819")
            if int(array[row, col, 2]):
                draw.rectangle((x + 4, y + 4, x + tile - 6, y + tile - 6), outline="#ffffff")
    arrow = {0: ">", 1: "v", 2: "<", 3: "^"}.get(intish(direction), "+")
    cx = start_x + (grid_w // 2) * tile + tile // 2
    cy = start_y + (grid_h // 2) * tile + tile // 2
    draw.text((cx - 8, cy - 10), arrow, fill="#ffffff", font=font(18, bold=True))
    return np.asarray(canvas)


def draw_state_bars(draw: ImageDraw.ImageDraw, values: list[float], *, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 28), "STATE", fill="#ffffff", font=font(16, bold=True))
    if not values:
        draw.text((x0, y0), "no numeric state", fill="#8fa09d", font=font(12))
        return
    shown = values[:18]
    row_h = max(10, (y1 - y0) // len(shown))
    for idx, value in enumerate(shown):
        y = y0 + idx * row_h
        clipped = max(-1.0, min(1.0, squash(float(value))))
        center = x0 + 210
        draw.text((x0, y - 1), f"s{idx:02d}", fill="#8fa09d", font=font(9))
        draw.rectangle((x0 + 44, y + 2, x1, y + 9), fill="#263134")
        draw.line((center, y + 2, center, y + 9), fill="#64706f")
        if clipped >= 0:
            draw.rectangle((center, y + 2, center + int(170 * clipped), y + 9), fill="#2f6fed")
        else:
            draw.rectangle((center + int(170 * clipped), y + 2, center, y + 9), fill="#c85f32")


def draw_action_bars(draw: ImageDraw.ImageDraw, values: list[float], *, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 28), "ACTION", fill="#ffffff", font=font(16, bold=True))
    if not values:
        draw.text((x0, y0), "reset frame", fill="#8fa09d", font=font(12))
        return
    shown = values[:8]
    row_h = 18
    for idx, value in enumerate(shown):
        y = y0 + idx * row_h
        clipped = max(-1.0, min(1.0, squash(float(value))))
        center = x0 + 88
        draw.text((x0, y - 1), f"a{idx}", fill="#8fa09d", font=font(9))
        draw.rectangle((x0 + 24, y + 3, x1, y + 11), fill="#263134")
        draw.line((center, y + 3, center, y + 11), fill="#64706f")
        if clipped >= 0:
            draw.rectangle((center, y + 3, center + int(70 * clipped), y + 11), fill="#1f9a72")
        else:
            draw.rectangle((center + int(70 * clipped), y + 3, center, y + 11), fill="#c85f32")


def squash(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(value))


def flatten_numbers(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, Mapping):
        values: list[float] = []
        for key in sorted(value):
            if key == "mission":
                continue
            values.extend(flatten_numbers(value[key]))
        return values
    if isinstance(value, str | bytes | bytearray):
        return []
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iufb":
            return []
        return [float(item) for item in value.reshape(-1)[:256]]
    if isinstance(value, list | tuple):
        values: list[float] = []
        for item in value:
            values.extend(flatten_numbers(item))
        return values
    return []


def compose_frames(
    records: tuple[FrameRecord, ...],
    *,
    model: str,
    env: str,
    submit_index: int,
    case_index: int,
    score: float | None,
    accent: str,
    max_frames: int,
) -> list[np.ndarray]:
    indexes = sample_indexes(len(records), max_frames)
    return [
        np.asarray(
            compose_frame(
                records[index],
                model=model,
                env=env,
                submit_index=submit_index,
                case_index=case_index,
                heldout_score=score,
                accent=accent,
            )
        )
        for index in indexes
    ]


def compose_frame(
    record: FrameRecord,
    *,
    model: str,
    env: str,
    submit_index: int,
    case_index: int,
    heldout_score: float | None,
    accent: str,
) -> Image.Image:
    source = Image.fromarray(record.image).convert("RGB")
    fitted = ImageOps.contain(source, CANVAS, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", CANVAS, "#101719")
    canvas.paste(fitted, ((CANVAS[0] - fitted.width) // 2, (CANVAS[1] - fitted.height) // 2))
    overlay = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, CANVAS[0], 58), fill=(16, 23, 25, 214))
    draw.rectangle((0, CANVAS[1] - 42, CANVAS[0], CANVAS[1]), fill=(16, 23, 25, 214))
    draw.rectangle((0, CANVAS[1] - 5, CANVAS[0], CANVAS[1]), fill=hex_rgba(accent, 235))
    draw.text((18, 12), f"{model} on {env}", fill="#ffffff", font=font(19, bold=True))
    draw.text(
        (18, 37),
        f"original env rerun | submit {submit_index:03d} | heldout case {case_index:03d}",
        fill="#dfe6e3",
        font=font(11),
    )
    draw.rounded_rectangle((588, 14, 700, 38), radius=6, fill=hex_rgba(accent, 235))
    draw.text((603, 20), "RERUN", fill="#ffffff", font=font(10, bold=True))
    bottom = (
        f"step {record.step:04d}   reward {record.reward:+.3f}   "
        f"return {record.total:+.3f}   final score {format_float(heldout_score)}"
    )
    draw.text((18, CANVAS[1] - 28), bottom, fill="#ffffff", font=font(12, bold=True))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def sample_indexes(length: int, max_frames: int) -> list[int]:
    if length <= 0:
        return []
    if max_frames <= 0 or length <= max_frames:
        return list(range(length))
    values = np.linspace(0, length - 1, max_frames)
    return sorted({int(round(item)) for item in values})


def resolve_run_path(row: dict[str, str]) -> Path:
    raw = ROOT / row["run_path"]
    if raw.exists():
        return raw
    matches = sorted((ROOT / "runs" / "main-128").glob(f"**/{row['run_id']}"))
    if matches:
        return matches[0]
    return raw


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def intish(value: Any) -> int:
    if value in {None, ""}:
        return 0
    return int(float(value))


def numeric(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hex_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    text = value.lstrip("#")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), alpha)


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    for path in candidates if bold else candidates[1::2]:
        with suppress(Exception):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def purge_helpers(system: Path) -> None:
    stems = {path.stem for path in system.glob("*.py") if path.stem != "policy"}
    for name in list(sys.modules):
        if name.split(".", 1)[0] in stems:
            del sys.modules[name]


@contextmanager
def project(path: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


if __name__ == "__main__":
    main()
