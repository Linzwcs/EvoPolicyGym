"""Scenario and environment validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hlbench.core.scenario import ScenarioSpec, load_scenario
from hlbench.core.task import WorkspaceContractSpec, build_task_contract
from hlbench.envs.registry import get_backend
from hlbench.envs.space_schema import validate_action


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str

    def to_record(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ScenarioValidationResult:
    scenario: str
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "ok": self.ok,
            "issues": [issue.to_record() for issue in self.issues],
        }


def validate_scenario(name: str, *, smoke_step: bool = True) -> ScenarioValidationResult:
    issues: list[ValidationIssue] = []
    try:
        scenario = load_scenario(name)
    except Exception as exc:
        return ScenarioValidationResult(
            scenario=name,
            ok=False,
            issues=[ValidationIssue("error", "scenario_load_failed", repr(exc))],
        )

    _validate_splits(scenario, issues)
    backend = None
    env_contract = None
    try:
        backend = get_backend(scenario.env_backend)
        env_contract = backend.describe(scenario)
        build_task_contract(
            scenario=scenario,
            env=env_contract,
            workspace=WorkspaceContractSpec(),
        )
    except Exception as exc:
        issues.append(ValidationIssue("error", "env_describe_failed", repr(exc)))

    if env_contract is not None:
        _validate_meanings(env_contract.action_schema, scenario, issues)
        try:
            action = sample_action(env_contract.action_schema)
            validate_action(env_contract.action_schema, action)
        except Exception as exc:
            issues.append(ValidationIssue("error", "action_schema_invalid", repr(exc)))

    if smoke_step and backend is not None and env_contract is not None:
        try:
            env = backend.make(scenario)
            try:
                first_seed = scenario.seeds_for_split("train", limit=1, sampler_seed=0)[0]
                env.reset(seed=first_seed, config={"scenario": scenario.scenario_id})
                env.step(sample_action(env.action_schema))
            finally:
                env.close()
        except Exception as exc:
            issues.append(ValidationIssue("error", "env_smoke_step_failed", repr(exc)))

    ok = not any(issue.severity == "error" for issue in issues)
    return ScenarioValidationResult(scenario=name, ok=ok, issues=issues)


def sample_action(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if kind == "discrete":
        return int(schema.get("start", 0))
    if kind == "box":
        shape = schema.get("shape") or []
        size = _shape_size(shape)
        values = [_midpoint(schema.get("low"), schema.get("high"), index, size) for index in range(size)]
        return _reshape(values, shape)
    if kind == "multi_discrete":
        return [0 for _ in _flatten(schema["nvec"])]
    if kind == "multi_binary":
        size = _shape_size(schema.get("n"))
        return [0 for _ in range(size)]
    if kind == "dict":
        return {key: sample_action(child) for key, child in schema.get("spaces", {}).items()}
    if kind == "tuple":
        return [sample_action(child) for child in schema.get("spaces", [])]
    return 0


def _validate_splits(scenario: ScenarioSpec, issues: list[ValidationIssue]) -> None:
    seen: dict[int, str] = {}
    for split_name, split in scenario.splits.items():
        if not split.seeds:
            issues.append(ValidationIssue("error", "empty_seed_pool", f"{split_name} has no seeds"))
        for seed in split.seeds:
            previous = seen.get(seed)
            if previous is not None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "overlapping_seed_pool",
                        f"seed {seed} appears in both {previous} and {split_name}",
                    )
                )
                return
            seen[seed] = split_name


def _validate_meanings(
    action_schema: dict[str, Any],
    scenario: ScenarioSpec,
    issues: list[ValidationIssue],
) -> None:
    if action_schema.get("type") == "discrete":
        expected = int(action_schema.get("n", 0))
        if len(scenario.action_meanings) < expected:
            severity = "error" if scenario.scenario_level == "official" else "warning"
            issues.append(
                ValidationIssue(
                    severity,
                    "incomplete_action_meanings",
                    f"discrete action space has {expected} actions but only {len(scenario.action_meanings)} meanings",
                )
            )


def _midpoint(low: Any, high: Any, index: int, size: int) -> float:
    lows = _broadcast(low, size)
    highs = _broadcast(high, size)
    lo, hi = lows[index], highs[index]
    if lo is None and hi is None:
        return 0.0
    if lo is None:
        return min(0.0, float(hi))
    if hi is None:
        return max(0.0, float(lo))
    return (float(lo) + float(hi)) / 2.0


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _broadcast(value: Any, size: int) -> list[Any | None]:
    if value is None:
        return [None] * size
    values = _flatten(value)
    if len(values) == 1:
        return values * size
    if len(values) != size:
        return [None] * size
    return values


def _shape_size(shape: Any) -> int:
    if shape is None:
        return 1
    if isinstance(shape, int):
        return shape
    size = 1
    for item in shape:
        size *= int(item)
    return size


def _reshape(values: list[float], shape: Any) -> Any:
    if not shape:
        return values[0] if values else 0.0
    if len(shape) == 1:
        return values
    stride = _shape_size(shape[1:])
    return [_reshape(values[index : index + stride], shape[1:]) for index in range(0, len(values), stride)]
