"""Optional environment discovery for Gymnasium-compatible packages."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import importlib.metadata
import io
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Package:
    """Import status for an optional environment package."""

    name: str
    module: str
    group: str
    status: str
    version: str | None = None
    error: str | None = None
    output: str | None = None

    def as_dict(self) -> dict[str, object]:
        return _compact(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class Spec:
    """Discovered Gymnasium registry entry."""

    id: str
    entry_point: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "entry_point": self.entry_point}


@dataclass(frozen=True, slots=True)
class Family:
    """Environment ids grouped by upstream family."""

    name: str
    count: int
    ids: tuple[str, ...]
    source: str
    note: str | None = None

    def as_dict(self) -> dict[str, object]:
        return _compact(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class Blocked:
    """Known package that cannot share the current Python/dependency stack."""

    name: str
    reason: str
    workaround: str | None = None

    def as_dict(self) -> dict[str, object]:
        return _compact(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class Report:
    """Complete discovery report."""

    packages: tuple[Package, ...]
    families: tuple[Family, ...]
    blocked: tuple[Blocked, ...]

    @property
    def total(self) -> int:
        return sum(family.count for family in self.families)

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "packages": [package.as_dict() for package in self.packages],
            "families": [family.as_dict() for family in self.families],
            "blocked": [blocked.as_dict() for blocked in self.blocked],
        }


OPTIONAL_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("gymnasium", "gymnasium", "env-gym"),
    ("ale-py", "ale_py", "env-gym"),
    ("autorom", "AutoROM", "env-gym"),
    ("jax", "jax", "env-jax"),
    ("jaxlib", "jaxlib", "env-jax"),
    ("flax", "flax", "env-jax"),
    ("minigrid", "minigrid", "env-compatible"),
    ("miniworld", "miniworld", "env-visual"),
    ("highway-env", "highway_env", "env-compatible"),
    ("gymnasium-robotics", "gymnasium_robotics", "env-compatible"),
    ("mo-gymnasium", "mo_gymnasium", "env-compatible"),
    ("gym-super-mario-bros", "gym_super_mario_bros", "env-mario"),
    ("nes-py", "nes_py", "env-mario"),
    ("metaworld", "metaworld", "env-heavy"),
    ("shimmy", "shimmy", "env-compatible"),
    ("pettingzoo", "pettingzoo", "env-compatible"),
    ("mpe2", "mpe2", "env-multi"),
    ("magent2", "magent2", "env-multi"),
    ("momaland", "momaland", "env-multi"),
    ("browsergym-miniwob", "browsergym.miniwob", "env-web"),
)

REGISTRATION_MODULES: tuple[str, ...] = (
    "ale_py",
    "minigrid",
    "miniworld",
    "highway_env",
    "gymnasium_robotics",
    "mo_gymnasium",
    "browsergym.miniwob",
)

BLOCKED: tuple[Blocked, ...] = (
    Blocked(
        "procgen",
        "No cp312 wheels are published; uv cannot resolve it for Python 3.12.",
        "Use a Python 3.10 discovery/runtime environment or defer Procgen.",
    ),
    Blocked(
        "safety-gymnasium",
        "Published releases require gymnasium 0.26.3 or 0.28.1, which conflicts with gymnasium 1.3.0.",
        "Run Safety-Gymnasium in a separate compatibility environment.",
    ),
    Blocked(
        "pyflyt",
        "Depends on pybullet; pybullet wheel build failed on this macOS arm64 Python 3.12 environment.",
        "Use a platform with a compatible pybullet wheel or a dedicated PyFlyt env.",
    ),
    Blocked(
        "miniwob",
        "The legacy miniwob package requires gymnasium 0.27.1 or 0.29.0.",
        "Use browsergym-miniwob for the current dependency stack.",
    ),
)

CLASSIC = {"Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum"}
TOY_TEXT = {"Blackjack", "CliffWalking", "FrozenLake", "Taxi"}
BOX2D = {"BipedalWalker", "CarRacing", "LunarLander"}
MUJOCO = {
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "HumanoidStandup",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
    "Reacher",
    "Swimmer",
    "Walker2d",
}
HIGHWAY = {
    "highway",
    "highway-fast",
    "merge",
    "roundabout",
    "parking",
    "intersection",
    "racetrack",
}
JAX_PREFIXES = ("phys2d/", "tabular/")
ROBOTICS_PREFIXES = (
    "Fetch",
    "Hand",
    "Adroit",
    "PointMaze",
    "AntMaze",
    "FrankaKitchen",
    "PegInsertion",
)

INTEGRATED: dict[str, str] = {
    "Acrobot-v1": "available as `gym/acrobot`",
    "CartPole-v1": "available as built-in env `cartpole` and `gym/cartpole`",
    "Blackjack-v1": "available as `gym/blackjack`",
    "CliffWalking-v1": "available as `gym/cliff`",
    "FrozenLake-v1": "available as `gym/frozenlake`",
    "MountainCar-v0": "available as `gym/mountaincar`",
    "MountainCarContinuous-v0": "available as `gym/continuouscar`",
    "Pendulum-v1": "available as `gym/pendulum`",
    "Taxi-v4": "available as `gym/taxi`",
    "BipedalWalker-v3": "available as `gym/bipedal` (L1 smoke)",
    "CarRacing-v3": "available as `gym/racing` (L1 smoke)",
    "LunarLander-v3": "available as `gym/lunar` (L1 smoke)",
    "Ant-v4": "available as `gym/ant4` (L1 smoke)",
    "Ant-v5": "available as `gym/ant5` (L1 smoke)",
    "HalfCheetah-v4": "available as `gym/halfcheetah4` (L1 smoke)",
    "HalfCheetah-v5": "available as `gym/halfcheetah5` (L1 smoke)",
    "Hopper-v4": "available as `gym/hopper4` (L1 smoke)",
    "Hopper-v5": "available as `gym/hopper5` (L1 smoke)",
    "Humanoid-v4": "available as `gym/humanoid4` (L1 smoke)",
    "Humanoid-v5": "available as `gym/humanoid5` (L1 smoke)",
    "HumanoidStandup-v4": "available as `gym/standup4` (L1 smoke)",
    "HumanoidStandup-v5": "available as `gym/standup5` (L1 smoke)",
    "InvertedDoublePendulum-v4": "available as `gym/doublependulum4` (L1 smoke)",
    "InvertedDoublePendulum-v5": "available as `gym/doublependulum5` (L1 smoke)",
    "InvertedPendulum-v4": "available as `gym/invertedpendulum4` (L1 smoke)",
    "InvertedPendulum-v5": "available as `gym/invertedpendulum5` (L1 smoke)",
    "Pusher-v4": "available as `gym/pusher4` (L1 smoke)",
    "Pusher-v5": "available as `gym/pusher5` (L1 smoke)",
    "Reacher-v4": "available as `gym/reacher4` (L1 smoke)",
    "Reacher-v5": "available as `gym/reacher5` (L1 smoke)",
    "Swimmer-v4": "available as `gym/swimmer4` (L1 smoke)",
    "Swimmer-v5": "available as `gym/swimmer5` (L1 smoke)",
    "Walker2d-v4": "available as `gym/walker4` (L1 smoke)",
    "Walker2d-v5": "available as `gym/walker5` (L1 smoke)",
}

BLOCKED_FAMILIES: dict[str, tuple[str, ...]] = {
    "Procgen": (
        "bigfish",
        "bossfight",
        "caveflyer",
        "chaser",
        "climber",
        "coinrun",
        "dodgeball",
        "fruitbot",
        "heist",
        "jumper",
        "leaper",
        "maze",
        "miner",
        "ninja",
        "plunder",
        "starpilot",
    ),
    "Safety-Gymnasium": (
        "PointGoal",
        "PointButton",
        "PointPush",
        "PointCircle",
        "PointRun",
        "PointVelocity",
        "CarGoal",
        "CarButton",
        "CarPush",
        "CarCircle",
        "CarRun",
        "CarVelocity",
        "RacecarGoal",
        "RacecarButton",
        "RacecarPush",
        "RacecarCircle",
        "RacecarRun",
        "AntGoal",
        "AntButton",
        "AntPush",
        "AntCircle",
        "AntRun",
        "AntVelocity",
        "HalfCheetahVelocity",
        "HopperVelocity",
        "SwimmerVelocity",
        "Walker2dVelocity",
    ),
    "PyFlyt": (
        "QuadX-Hover",
        "QuadX-Waypoints",
        "QuadX-Gates",
        "QuadX-Pole-Balance",
        "Fixedwing-Waypoints",
        "Rocket-Landing",
    ),
}


def discover() -> Report:
    """Discover installed optional environment packages and registry ids."""

    modules: dict[str, object] = {}
    packages = tuple(_package(name, module, group, modules) for name, module, group in OPTIONAL_PACKAGES)
    gym = modules.get("gymnasium")
    if gym is not None:
        _register(gym, modules)
    specs = _gym_specs(gym) if gym is not None else ()
    families = classify(specs, modules)
    return Report(packages=packages, families=families, blocked=BLOCKED)


def classify(specs: Iterable[Spec], modules: dict[str, object] | None = None) -> tuple[Family, ...]:
    """Group raw Gymnasium specs into EvoPolicyGym planning families."""

    rows = tuple(sorted(specs, key=lambda spec: spec.id))
    modules = modules or {}
    families = [
        _known("Official Gymnasium: Classic Control", rows, CLASSIC, ("gymnasium.envs.classic_control",)),
        _known("Official Gymnasium: Toy Text", rows, TOY_TEXT, ("gymnasium.envs.toy_text",)),
        _prefix("Official Gymnasium: JAX", rows, JAX_PREFIXES),
        _known("Official Gymnasium: Box2D", rows, BOX2D, ("gymnasium.envs.box2d",)),
        _known("Official Gymnasium: MuJoCo", rows, MUJOCO, ("gymnasium.envs.mujoco",)),
        _entry("Official Gymnasium: Atari / ALE", rows, ("ale_py",), prefix=("ALE/",)),
        _prefix("MiniGrid", rows, ("MiniGrid-",)),
        _prefix("MiniWorld", rows, ("MiniWorld-",)),
        _known("HighwayEnv", rows, HIGHWAY, ("highway_env",)),
        _predicate("Gymnasium-Robotics", rows, lambda spec: _base(spec.id).startswith(ROBOTICS_PREFIXES)),
        _entry("MO-Gymnasium", rows, ("mo_gymnasium",)),
        _prefix("BrowserGym MiniWoB++", rows, ("browsergym/", "browsergym-")),
    ]
    metaworld = _metaworld(modules.get("metaworld"))
    if metaworld.count:
        families.append(metaworld)
    return tuple(family for family in families if family.count)


def write_json(report: Report, path: Path) -> None:
    """Write report as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(report: Report, path: Path) -> None:
    """Write report as a Markdown checklist."""

    lines = [
        "# Environment List",
        "",
        "> Generated from installed optional packages. This is the canonical",
        "> EvoPolicyGym environment backlog for the current dependency stack.",
        "",
        "Legend:",
        "",
        "- `[x]`: already available through an EvoPolicyGym registration.",
        "- `[ ]`: discovered upstream registry id, not yet integrated through the generic adapter.",
        "",
        "## Summary",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {family.name} | {family.count} |" for family in report.families)
    lines.append(f"| Total | {report.total} |")
    lines.append("")
    for family in report.families:
        lines.append(f"## {family.name} ({family.count})")
        if family.note:
            lines.extend(("", family.note))
        lines.append("")
        for env_id in family.ids:
            checked = "x" if env_id in INTEGRATED else " "
            note = f" - {INTEGRATED[env_id]}" if env_id in INTEGRATED else ""
            lines.append(f"- [{checked}] {env_id}{note}")
        lines.append("")
    lines.append("## Installed Packages")
    lines.append("")
    for package in report.packages:
        version = f" {package.version}" if package.version else ""
        detail = f": {package.error}" if package.error else ""
        lines.append(f"- {package.status}: {package.name}{version} ({package.group}){detail}")
    lines.append("")
    lines.append("## Blocked Packages")
    lines.append("")
    for blocked in report.blocked:
        workaround = f" Workaround: {blocked.workaround}" if blocked.workaround else ""
        lines.append(f"- {blocked.name}: {blocked.reason}{workaround}")
    lines.append("")
    lines.append("## Blocked Target Families")
    lines.append("")
    lines.append(
        "These candidates are tracked for coverage planning but are not included in the discovered total."
    )
    for family, ids in BLOCKED_FAMILIES.items():
        lines.append("")
        lines.append(f"### {family} ({len(ids)})")
        lines.append("")
        lines.extend(f"- [ ] {env_id}" for env_id in ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evopolicygym.envs.discover")
    parser.add_argument("--output", type=Path, help="write JSON report")
    parser.add_argument("--markdown", type=Path, help="write Markdown report")
    args = parser.parse_args(argv)

    report = discover()
    if args.output:
        write_json(report, args.output)
    if args.markdown:
        write_markdown(report, args.markdown)
    if not args.output and not args.markdown:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


def _package(name: str, module_name: str, group: str, modules: dict[str, object]) -> Package:
    module, error, output = _import(module_name)
    if module is None:
        return Package(name=name, module=module_name, group=group, status="missing", error=error, output=output)
    modules[module_name] = module
    version = _version(name)
    return Package(name=name, module=module_name, group=group, status="installed", version=version, output=output)


def _import(module_name: str) -> tuple[object | None, str | None, str | None]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - discovery should report, not fail.
        return None, f"{type(exc).__name__}: {exc}", _captured(stdout, stderr)
    return module, None, _captured(stdout, stderr)


def _register(gym: object, modules: dict[str, object]) -> None:
    register = getattr(gym, "register_envs", None)
    if not callable(register):
        return
    for module_name in REGISTRATION_MODULES:
        module = modules.get(module_name)
        if module is None:
            continue
        with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            register(module)


def _gym_specs(gym: object) -> tuple[Spec, ...]:
    envs = getattr(gym, "envs", None)
    registry = getattr(envs, "registry", {})
    specs = []
    for env_id, spec in registry.items():
        entry = getattr(spec, "entry_point", "") or ""
        specs.append(Spec(str(env_id), str(entry)))
    return tuple(sorted(specs, key=lambda spec: spec.id))


def _known(name: str, specs: Iterable[Spec], names: set[str], entry_prefix: tuple[str, ...]) -> Family:
    ids = tuple(
        spec.id
        for spec in specs
        if _base(spec.id) in names and spec.entry_point.startswith(entry_prefix)
    )
    return Family(name=name, count=len(ids), ids=ids, source="gymnasium.registry")


def _prefix(name: str, specs: Iterable[Spec], prefix: tuple[str, ...]) -> Family:
    ids = tuple(spec.id for spec in specs if spec.id.startswith(prefix))
    return Family(name=name, count=len(ids), ids=ids, source="gymnasium.registry")


def _entry(name: str, specs: Iterable[Spec], entry_prefix: tuple[str, ...], prefix: tuple[str, ...] = ()) -> Family:
    ids = tuple(
        spec.id
        for spec in specs
        if spec.entry_point.startswith(entry_prefix) or (prefix and spec.id.startswith(prefix))
    )
    return Family(name=name, count=len(ids), ids=ids, source="gymnasium.registry")


def _predicate(name: str, specs: Iterable[Spec], predicate) -> Family:
    ids = tuple(spec.id for spec in specs if predicate(spec))
    return Family(name=name, count=len(ids), ids=ids, source="gymnasium.registry")


def _metaworld(module: object | None) -> Family:
    if module is None or not hasattr(module, "MT50"):
        return Family("MetaWorld", 0, (), "metaworld.MT50")
    try:
        mt50 = module.MT50(seed=0)
        ids = tuple(sorted(str(name) for name in mt50.train_classes))
    except Exception as exc:  # noqa: BLE001
        return Family("MetaWorld", 0, (), "metaworld.MT50", note=f"discovery failed: {exc}")
    return Family("MetaWorld", len(ids), ids, "metaworld.MT50")


def _base(env_id: str) -> str:
    value = env_id.rsplit("/", 1)[-1]
    return re.sub(r"-v\d+$", "", value)


def _version(package: str) -> str | None:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(package)
    return None


def _captured(stdout: io.StringIO, stderr: io.StringIO) -> str | None:
    text = "\n".join(part.strip() for part in (stdout.getvalue(), stderr.getvalue()) if part.strip())
    if not text:
        return None
    return text[:500]


def _compact(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if item not in (None, "", (), [])}


if __name__ == "__main__":
    raise SystemExit(main())
