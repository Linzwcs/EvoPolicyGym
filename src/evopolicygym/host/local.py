"""Local host assembly for one EvoPolicyGym run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core import Budget, Env, Pool, PoolKind, Run, Runtime, Value
from ..data import load as load_data
from ..infra.fs import FileStore
from ..infra.http import Service
from ..infra.runtime import PolicyRuntime, Roller, Sandbox, SandboxRuntime
from ..judge import Limits
from ..protocol import PROTOCOL

Json = Any


@dataclass(slots=True)
class Host:
    """Assembled local server-side objects for one run."""

    env: Env
    store: FileStore
    runtime: Runtime
    service: Service
    train: Pool
    valid: Pool
    final: Pool
    limits: Limits

    @property
    def run(self) -> Run:
        return self.service.run


def local(
    root: str | Path,
    env: Env,
    *,
    key: str,
    model: str,
    exp: str,
    budget: int,
    limits: Limits | None = None,
    minimum: int = 1,
    maximum: int | None = None,
    valid_size: int | None = None,
    final_size: int | None = None,
    data: str | Path | None = None,
    protocol: str = PROTOCOL,
    dimensions: Mapping[str, Json] | None = None,
    versions: Mapping[str, Json] | None = None,
    denied: frozenset[str] = frozenset(),
    value: Value | None = None,
    sandbox: Sandbox | None = None,
    task_text: str = "",
    resource_limits: Mapping[str, Json] | None = None,
) -> Host:
    """Build and open a local filesystem-backed host for one run."""

    root = Path(root)
    train = env.pool(PoolKind.train)
    valid = env.pool(PoolKind.valid)
    final = env.pool(PoolKind.final)
    merged_versions = dict(versions or {})
    if data is not None:
        corpus = load_data(data, env=env.task.name)
        train = corpus.train.pool
        valid = corpus.valid.pool
        final = corpus.final.pool
        env = replace(env, task=replace(env.task, cases=train.size))
        merged_versions.update(corpus.versions())

    run = Run(
        key=key,
        model=model,
        env=env.task.name,
        exp=exp,
        protocol=protocol,
        budget=Budget(limit=budget),
    )
    store = FileStore(
        root,
        dimensions=dict(dimensions or {}),
        versions=merged_versions,
    )
    roller = Roller(env.make)
    score = value if value is not None else env.value
    runtime: Runtime
    if sandbox is None:
        runtime = PolicyRuntime(root, roller, denied=denied, value=score)
    else:
        runtime = SandboxRuntime(
            root,
            roller,
            sandbox=sandbox,
            denied=denied,
            value=score,
        )

    if valid_size is not None:
        valid = valid.trim(valid_size)
    if final_size is not None:
        final = final.trim(final_size)
    bounds = limits or Limits(minimum=minimum, maximum=maximum or budget)

    visible_limits: dict[str, Json] = {
        "rollout_wall_s": None,
        "memory_bytes": None,
    }
    if sandbox is not None:
        visible_limits.update(
            {
                "rollout_wall_s": sandbox.rollout,
                "memory_bytes": sandbox.memory,
            }
        )
    visible_limits.update(dict(resource_limits or {}))

    store.open(run)
    service = Service(
        run=run,
        task=env.task,
        train=train,
        store=store,
        runtime=runtime,
        limits=bounds,
        valid=valid,
        final=final,
        task_text=task_text or env.text,
        resource_limits=visible_limits,
        caps=env.caps,
        denied_imports=tuple(sorted(denied)),
        protocol=protocol,
        log=store,
    )
    return Host(
        env=env,
        store=store,
        runtime=runtime,
        service=service,
        train=train,
        valid=valid,
        final=final,
        limits=bounds,
    )
