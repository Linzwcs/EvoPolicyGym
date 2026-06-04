"""Dynamic Gymnasium registry integration.

Bulk registrations use stable long names derived from upstream ids:
`gymnasium/<UpstreamEnvId>`. Curated short aliases in `spec.py` remain the
default surface and are not replaced by this module.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..discover import REGISTRATION_MODULES, Spec, classify
from .family import task_doc
from .spec import GymSpec

PREFIX = "gymnasium/"
_MINIWOB_LOCAL = Path("third_party/miniwob-plusplus/miniwob/html/miniwob")
_MINIWOB_SENTINELS = ("click-button.html", "ascending-numbers.html")


@dataclass(frozen=True, slots=True)
class BulkSpec:
    """One dynamically discovered Gymnasium registry id."""

    spec: GymSpec
    entry_point: str
    family: str

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def upstream(self) -> str:
        return self.spec.id


def name(env_id: str) -> str:
    """Return the EvoPolicyGym bulk name for a Gymnasium id."""

    return f"{PREFIX}{env_id}"


def discover(filters: Iterable[str] = ()) -> tuple[BulkSpec, ...]:
    """Return dynamic specs for installed Gymnasium registry ids."""

    selected = set(filters)
    gym = _gymnasium()
    _register(gym)
    raw = _registry(gym)
    if selected:
        raw = tuple(item for item in raw if _matches(item.id, selected))
    families = _families(raw)
    return tuple(
        BulkSpec(
            spec=GymSpec(
                name=name(item.id),
                id=item.id,
                steps=_steps(item.raw),
                kwargs=_kwargs(item.id),
                doc=task_doc(item.id, families.get(item.id, "Unclassified Gymnasium")),
            ),
            entry_point=item.entry_point,
            family=families.get(item.id, "Unclassified Gymnasium"),
        )
        for item in raw
    )


def specs(filters: Iterable[str] = ()) -> tuple[GymSpec, ...]:
    """Return only the GymSpec objects from dynamic discovery."""

    return tuple(item.spec for item in discover(filters))


@dataclass(frozen=True, slots=True)
class _Raw:
    id: str
    entry_point: str
    raw: object


def _gymnasium() -> object:
    try:
        return importlib.import_module("gymnasium")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gymnasium support requires `uv sync --extra env-gym`") from exc


def _register(gym: object) -> None:
    register = getattr(gym, "register_envs", None)
    if not callable(register):
        return
    for module_name in REGISTRATION_MODULES:
        _register_module(register, module_name)


def _register_module(register: object, module_name: str) -> None:
    with contextlib.suppress(Exception):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module = importlib.import_module(module_name)
            register(module)


def _registry(gym: object) -> tuple[_Raw, ...]:
    envs = getattr(gym, "envs", None)
    registry = getattr(envs, "registry", {})
    rows: list[_Raw] = []
    for env_id, item in registry.items():
        upstream = str(getattr(item, "id", env_id))
        entry_point = str(getattr(item, "entry_point", "") or "")
        rows.append(_Raw(upstream, entry_point, item))
    return tuple(sorted(rows, key=lambda item: item.id))


def _families(rows: tuple[_Raw, ...]) -> dict[str, str]:
    grouped = classify(Spec(item.id, item.entry_point) for item in rows)
    families: dict[str, str] = {}
    for family in grouped:
        for env_id in family.ids:
            families[env_id] = family.name
    return families


def _steps(item: object) -> int:
    value = getattr(item, "max_episode_steps", None)
    if isinstance(value, int) and value > 0:
        return value
    return 1000


def _kwargs(env_id: str) -> dict[str, object]:
    if env_id == "browsergym/openended":
        return {
            "task_kwargs": {
                "start_url": "about:blank",
                "goal": "Open-ended browser smoke task.",
            }
        }
    if env_id.startswith("browsergym/miniwob."):
        base_url = _miniwob_base_url()
        if base_url is not None:
            return {"task_kwargs": {"base_url": base_url}}
    return {}


def _miniwob_base_url(*, cwd: Path | None = None) -> str | None:
    """Return a BrowserGym MiniWoB++ base URL when configured or locally vendored."""

    configured = os.environ.get("MINIWOB_URL", "").strip()
    if configured:
        return _directory_url(configured)

    for candidate in _miniwob_candidates(cwd or Path.cwd()):
        if all((candidate / name).exists() for name in _MINIWOB_SENTINELS):
            return _directory_url(candidate)
    return None


def _miniwob_candidates(cwd: Path) -> tuple[Path, ...]:
    roots = [cwd, *cwd.parents, _source_root()]
    rows: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        candidate = (root / _MINIWOB_LOCAL).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        rows.append(candidate)
    return tuple(rows)


def _source_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return path.parents[4]


def _directory_url(value: str | Path) -> str:
    if isinstance(value, Path):
        text = value.expanduser().resolve().as_uri()
    elif "://" in value:
        text = value
    else:
        text = Path(value).expanduser().resolve().as_uri()
    return text if text.endswith("/") else f"{text}/"


def _matches(env_id: str, filters: set[str]) -> bool:
    return env_id in filters or name(env_id) in filters


__all__ = ["BulkSpec", "PREFIX", "discover", "name", "specs"]
