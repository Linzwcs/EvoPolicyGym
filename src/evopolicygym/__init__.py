"""The small, lazily loaded user-facing EvoPolicyGym facade."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from .benchmark import Benchmark
    from .evaluation import EvaluationConfig, evaluate
    from .program import Program
    from .results import EvaluationResult, RunResult

_EXPORTS = {
    "Benchmark": (".benchmark", "Benchmark"),
    "EvaluationConfig": (".evaluation", "EvaluationConfig"),
    "EvaluationResult": (".results", "EvaluationResult"),
    "Program": (".program", "Program"),
    "RunResult": (".results", "RunResult"),
    "evaluate": (".evaluation", "evaluate"),
}

__all__ = [
    "Benchmark",
    "EvaluationConfig",
    "EvaluationResult",
    "Program",
    "RunResult",
    "__version__",
    "evaluate",
]


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name, __name__)
    # Cache all public values owned by the selected facade together.
    for export_name, (owner, owner_attribute) in _EXPORTS.items():
        if owner == module_name:
            globals()[export_name] = getattr(module, owner_attribute)
    return globals()[name]
