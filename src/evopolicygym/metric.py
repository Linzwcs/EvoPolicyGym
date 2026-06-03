"""Static engineering metrics for submitted policy snapshots."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

Json = Any

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_SKIP_FILES = {".keep", "_meta.json", "metrics.json"}


def measure(root: str | Path) -> dict[str, Json]:
    """Measure deterministic, host-only code metrics under `root`."""

    base = Path(root)
    files = _files(base)
    python = [path for path in files if path.suffix == ".py"]
    stats = [_python(base, path) for path in python]
    imports = sorted({item for stat in stats for item in stat["imports"]})
    parse_errors = [stat["path"] for stat in stats if stat["parse_error"]]

    return {
        "schema_version": "0.1",
        "files": len(files),
        "python_files": len(python),
        "bytes": sum(_size(path) for path in files),
        "policy_bytes": _size(base / "policy.py"),
        "lines": sum(stat["lines"] for stat in stats),
        "source_lines": sum(stat["source_lines"] for stat in stats),
        "classes": sum(stat["classes"] for stat in stats),
        "functions": sum(stat["functions"] for stat in stats),
        "imports": imports,
        "import_count": len(imports),
        "cyclomatic_total": sum(stat["cyclomatic_total"] for stat in stats),
        "cyclomatic_max": max((stat["cyclomatic_max"] for stat in stats), default=0),
        "parse_errors": parse_errors,
        "test_files": sum(1 for path in python if path.name.startswith("test_")),
        "tree_hash": _hash(base, files),
    }


def _files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    items: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts[:-1])
        if parts & _SKIP_DIRS:
            continue
        if path.name in _SKIP_FILES:
            continue
        items.append(path)
    return sorted(items, key=lambda item: item.relative_to(root).as_posix())


def _python(root: Path, path: Path) -> dict[str, Json]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    base = {
        "path": path.relative_to(root).as_posix(),
        "lines": len(lines),
        "source_lines": sum(1 for line in lines if _source(line)),
        "classes": 0,
        "functions": 0,
        "imports": [],
        "cyclomatic_total": 0,
        "cyclomatic_max": 0,
        "parse_error": False,
    }
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        base["parse_error"] = True
        return base

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    complexities = [_complexity(node) for node in functions]
    if not complexities and text.strip():
        complexities = [_complexity(tree)]
    base.update(
        {
            "classes": sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
            "functions": len(functions),
            "imports": sorted(_imports(tree)),
            "cyclomatic_total": sum(complexities),
            "cyclomatic_max": max(complexities, default=0),
        }
    )
    return base


def _source(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and not stripped.startswith("#"))


def _imports(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                values.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            values.add(node.module.split(".", 1)[0])
    return values


def _complexity(tree: ast.AST) -> int:
    score = 1
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.If | ast.For | ast.AsyncFor | ast.While | ast.IfExp | ast.ExceptHandler,
        ):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += max(0, len(node.values) - 1)
        elif isinstance(node, ast.Try):
            score += len(node.handlers)
        elif isinstance(node, ast.comprehension):
            score += len(node.ifs)
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):
            score += len(node.cases)
    return score


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _hash(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()
