"""Static complexity metrics for Python policy files."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from typing import Any


def analyze_policy_complexity(path: Path) -> dict[str, Any]:
    """Return JSON-serializable complexity metrics for a Python file."""

    source = path.read_text() if path.exists() else ""
    metrics = _basic_metrics(source)
    metrics.update(_ast_metrics(source))
    metrics.update(_radon_metrics(source))
    return metrics


def complexity_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "loc",
        "lloc",
        "sloc",
        "comment_lines",
        "blank_lines",
        "ast_node_count",
        "branch_count",
        "max_nesting_depth",
        "function_count",
        "class_count",
        "cyclomatic_complexity_total",
        "cyclomatic_complexity_mean",
        "cyclomatic_complexity_max",
        "maintainability_index",
    ]
    delta: dict[str, Any] = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta[key] = after_value - before_value
    return delta


def _basic_metrics(source: str) -> dict[str, Any]:
    lines = source.splitlines()
    comment_lines = 0
    blank_lines = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comment_lines = sum(1 for token in tokens if token.type == tokenize.COMMENT)
    except tokenize.TokenError:
        comment_lines = sum(1 for line in lines if line.lstrip().startswith("#"))
    blank_lines = sum(1 for line in lines if not line.strip())
    sloc = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return {
        "tool": "radon",
        "loc": len(lines),
        "sloc": sloc,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
    }


def _ast_metrics(source: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source or "\n")
    except SyntaxError as exc:
        return {
            "ast_parse_ok": False,
            "ast_error": f"{exc.__class__.__name__}: {exc}",
            "ast_node_count": 0,
            "branch_count": 0,
            "max_nesting_depth": 0,
            "function_count": 0,
            "class_count": 0,
        }

    visitor = _ComplexityVisitor()
    visitor.visit(tree)
    return {
        "ast_parse_ok": True,
        "ast_node_count": sum(1 for _ in ast.walk(tree)),
        "branch_count": visitor.branch_count,
        "max_nesting_depth": visitor.max_nesting_depth,
        "function_count": visitor.function_count,
        "class_count": visitor.class_count,
    }


def _radon_metrics(source: str) -> dict[str, Any]:
    try:
        from radon.complexity import cc_visit
        from radon.metrics import mi_visit
        from radon.raw import analyze
    except ImportError:
        return {
            "tool": "ast_fallback",
            "radon_available": False,
            "lloc": None,
            "multi_line_strings": None,
            "cyclomatic_complexity_total": None,
            "cyclomatic_complexity_mean": None,
            "cyclomatic_complexity_max": None,
            "maintainability_index": None,
        }

    try:
        raw = analyze(source)
        blocks = cc_visit(source)
        complexities = [int(block.complexity) for block in blocks]
        total = sum(complexities)
        maximum = max(complexities) if complexities else 0
        mean = total / len(complexities) if complexities else 0.0
        return {
            "tool": "radon",
            "radon_available": True,
            "loc": raw.loc,
            "lloc": raw.lloc,
            "sloc": raw.sloc,
            "comment_lines": raw.comments,
            "multi_line_strings": raw.multi,
            "blank_lines": raw.blank,
            "cyclomatic_complexity_total": total,
            "cyclomatic_complexity_mean": mean,
            "cyclomatic_complexity_max": maximum,
            "cyclomatic_complexity_blocks": len(complexities),
            "maintainability_index": mi_visit(source, multi=True),
        }
    except Exception as exc:  # pragma: no cover - defensive around optional dependency internals.
        return {
            "tool": "ast_fallback",
            "radon_available": False,
            "radon_error": f"{exc.__class__.__name__}: {exc}",
            "lloc": None,
            "multi_line_strings": None,
            "cyclomatic_complexity_total": None,
            "cyclomatic_complexity_mean": None,
            "cyclomatic_complexity_max": None,
            "maintainability_index": None,
        }


class _ComplexityVisitor(ast.NodeVisitor):
    _BRANCH_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.IfExp,
        ast.Match,
    )

    def __init__(self) -> None:
        self.branch_count = 0
        self.max_nesting_depth = 0
        self.function_count = 0
        self.class_count = 0
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.function_count += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.function_count += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.class_count += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.branch_count += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, self._BRANCH_NODES):
            self.branch_count += 1
            self._depth += 1
            self.max_nesting_depth = max(self.max_nesting_depth, self._depth)
            try:
                return super().visit(node)
            finally:
                self._depth -= 1
        return super().visit(node)
