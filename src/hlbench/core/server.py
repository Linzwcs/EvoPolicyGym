"""Per-run server.

One ``Server`` instance corresponds to one (model, env, exp-id) run.
The HTTP layer (post-MVP) wraps these methods one-to-one.

This module has no HTTP dependency — the Server can be driven directly
for tests and internal tooling. Per CLAUDE.md invariant 9, agents must
go through HTTP; the lib API is for tests / dev / orchestration only.

See:
- SPEC.md §1.1 (``GET /info`` schema)
- SPEC.md §3 (server interface)
- docs/architecture.md §4.1 (Server class sketch)
- docs/submit-protocol.md §2 (submit lifecycle)
"""

from __future__ import annotations

import hashlib
import shutil
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hlbench
from hlbench.core import feedback as fb
from hlbench.core import scoring
from hlbench.core.heldout import (
    HeldoutError,
    HeldoutResult,
    evaluate_heldout,
    snapshot_workspace_system,
)
from hlbench.core.sandbox import SandboxConfig
from hlbench.core.seed_manager import SeedManager
from hlbench.core.submit_handler import (
    SubmitConfig,
    SubmitHandler,
    SubmitOutcome,
    SubmitState,
)
from hlbench.envs.registry import EnvDefinition, get_env

# Located via source-tree walk: src/hlbench/core/server.py → repo root.
# When packaged into a wheel, the AGENTS.md may not be at this location;
# callers can override via ``agents_md_path``.
_REPO_ROOT_AGENTS_MD = Path(__file__).resolve().parents[3] / "AGENTS.md"


# --------------------------- result types ---------------------------------


@dataclass(frozen=True)
class SubmitResult:
    """Returned from ``Server.submit()``. Mirrors the HTTP /submit body.

    Schema-wise this is a thin wrapper around ``SubmitOutcome`` for the
    network layer; the full ``summary.json`` content is reachable on
    disk and via ``info()`` callers don't need it inline."""

    submit_id: int
    status: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class FinalResult:
    """Returned from ``Server.finalize()``. Mirrors a subset of
    ``run.json:outcome`` for callers that don't want to re-read the file."""

    status: str  # "completed" | "error"
    final_score: float | None
    held_out_mean_return: float | None
    held_out_std_return: float | None
    held_out_returns: list[float] | None
    final_submit_index: int | None
    error: dict[str, Any] | None
    run_json_path: Path


# --------------------------- server ---------------------------------------


class Server:
    """Per-run hlbench server.

    Lifecycle::

        srv = Server(env_id="pendulum", workspace_dir=Path("./run"))
        # workspace/{AGENTS.md, system/, feedback/} now exist.
        info = srv.info()                       # config + state snapshot
        result = srv.submit([0, 1, 2, 3])       # 4 episodes; writes feedback
        # ... iterate ...
        srv.finalize()                          # held-out (Day 8)

    The Server is single-tenant: one instance per (model, env, exp-id)
    run. Concurrency is the caller's responsibility — methods are not
    thread-safe.
    """

    def __init__(
        self,
        env_id: str,
        workspace_dir: Path,
        *,
        config_overrides: dict[str, Any] | None = None,
        agents_md_path: Path | None = None,
        run_dir: Path | None = None,
        model: str = "unknown",
        exp_id: str | None = None,
    ) -> None:
        """Initialize the run and stage workspace files.

        Args:
            env_id: Registered env identifier (e.g. ``"pendulum"``).
            workspace_dir: Local directory; created if missing. Server
                ensures ``AGENTS.md``, ``system/``, and
                ``feedback/`` exist underneath it. ``system/`` is left
                empty (ready for the agent to drop ``policy.py``);
                ``feedback/`` is empty until the first submit.
            config_overrides: Optional dict to override defaults like
                ``episode_budget``, ``act_wall_s``, ``init_wall_s``.
                Unknown keys raise. None ⇒ defaults from SPEC §1.1.
            agents_md_path: Source ``AGENTS.md`` to copy into the workspace.
                Defaults to the repo's ``AGENTS.md`` (works in a source
                checkout). If the source file doesn't exist, a minimal
                placeholder is written instead — callers running from a
                wheel should pass an explicit path.
            run_dir: Directory where ``run.json`` will be written by
                ``finalize()``. Defaults to the workspace's parent
                directory (so ``runs/<model>/<env>/<exp-id>/`` layouts
                that put workspace at ``runs/<...>/workspace/`` work
                out of the box). Created if missing.
            model: Recorded as ``run.json:model``. Free-form identifier
                for whatever produced ``system/policy.py``.
            exp_id: Recorded as ``run.json:exp_id``. Defaults to the
                ISO timestamp of run start so consecutive runs of the
                same model/env don't collide.
        """
        # Resolve env + seeds.
        self._env_def: EnvDefinition = get_env(env_id)
        self._sm = SeedManager(
            self._env_def.train_seeds_path,
            self._env_def.heldout_seeds_path,
        )

        self._workspace = Path(workspace_dir).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        (self._workspace / "system").mkdir(exist_ok=True)
        (self._workspace / "feedback").mkdir(exist_ok=True)

        # Stage AGENTS.md (the only static doc in the workspace; TASK.md is
        # served via GET /task per CLAUDE.md invariant 5).
        src = agents_md_path or _REPO_ROOT_AGENTS_MD
        agents_dst = self._workspace / "AGENTS.md"
        if src.exists():
            shutil.copy(src, agents_dst)
        else:
            agents_dst.write_text(
                "# AGENTS.md\n\n"
                "(placeholder — install hlbench from source to get the real one)\n"
            )
        self._agents_md_hash = "sha256:" + hashlib.sha256(
            agents_dst.read_bytes()
        ).hexdigest()

        # Build config from overrides.
        overrides = config_overrides or {}
        unknown = set(overrides) - {
            "episode_budget", "min_episodes_per_submit", "max_episodes_per_submit",
            "act_wall_s", "init_wall_s", "episode_wall_s", "max_rss_bytes",
        }
        if unknown:
            raise ValueError(f"unknown config_overrides keys: {sorted(unknown)}")

        sandbox_cfg = SandboxConfig(
            init_wall_s=overrides.get("init_wall_s", SandboxConfig.init_wall_s),
            act_wall_s=overrides.get("act_wall_s", SandboxConfig.act_wall_s),
            episode_wall_s=overrides.get(
                "episode_wall_s", SandboxConfig.episode_wall_s
            ),
            max_rss_bytes=overrides.get("max_rss_bytes", SandboxConfig.max_rss_bytes),
        )
        self._submit_cfg = SubmitConfig(
            episode_budget=overrides.get("episode_budget", 256),
            min_episodes_per_submit=overrides.get("min_episodes_per_submit", 1),
            max_episodes_per_submit=overrides.get("max_episodes_per_submit", 256),
            sandbox=sandbox_cfg,
        )
        self._handler = SubmitHandler(
            env_def=self._env_def,
            seed_manager=self._sm,
            workspace_dir=self._workspace,
            config=self._submit_cfg,
        )

        # Mutable run state.
        self._state = SubmitState(remaining_budget=self._submit_cfg.episode_budget)
        self._is_finalized = False
        self._final_result: FinalResult | None = None
        self._start_monotonic = time.monotonic()
        now = datetime.now(UTC)
        self._started_at = (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z"
        )

        # Run identity (used by run.json on finalize).
        self._model = model
        self._exp_id = exp_id or self._started_at.replace(":", "-").replace(".", "-")
        self._run_dir = (Path(run_dir).resolve() if run_dir
                         else self._workspace.parent)
        self._run_dir.mkdir(parents=True, exist_ok=True)

    # ------------- public API -------------------------------------------

    def task_md_text(self) -> str:
        """Return the env's task description as a markdown string.

        Served via ``GET /task``. The env package ships ``TASK.md`` as a
        static file; if the env didn't register a path or the file is
        missing, return a minimal placeholder so the endpoint always
        succeeds. Per CLAUDE.md invariant 5, this is NOT staged into
        the workspace — agents fetch it on demand."""
        path = self._env_def.task_md_path
        if path is not None and path.exists():
            return path.read_text()
        return (
            f"# {self._env_def.env_id}\n\n"
            "(no TASK.md template registered for this env)\n"
        )

    def info(self) -> dict[str, Any]:
        """Return ``GET /info`` content (static config + dynamic state).

        Per SPEC §1.1: this is the single source of truth for the run's
        configuration. Callers that need fresh state (notably
        ``remaining_budget``) re-call this between submits."""
        cfg = self._submit_cfg
        sb = cfg.sandbox
        return {
            "schema_version": "0.1",
            "env": self._env_def.env_id,
            "env_version": self._env_def.env_version,
            "harness_version": hlbench.__version__,
            "agents_md_hash": self._agents_md_hash,

            "episode_budget": cfg.episode_budget,
            "min_episodes_per_submit": cfg.min_episodes_per_submit,
            "max_episodes_per_submit": cfg.max_episodes_per_submit,

            "resource_limits": {
                # MVP placeholders — Phase 2/3 oversize/denied checks not
                # yet enforced (see submit_handler.py).
                "system_total_bytes": 50 * 1024 * 1024,
                "system_single_file_bytes": 5 * 1024 * 1024,
                "act_wall_ms": int(sb.act_wall_s * 1000),
                "policy_load_wall_s": int(sb.init_wall_s),
                "submit_wall_s": int(sb.episode_wall_s * cfg.max_episodes_per_submit),
                "submit_peak_rss_bytes": sb.max_rss_bytes or 0,
            },

            # MVP: no allow/deny enforcement; surface empty lists so the
            # field exists and post-MVP wiring is a value swap.
            "allowed_imports": [],
            "denied_imports": [],

            "env_meta": self._env_def.public_env_meta(),

            "state": {
                "remaining_budget": self._state.remaining_budget,
                "n_submits": self._state.n_submits,
                "n_successful_submits": self._state.n_successful_submits,
                "last_submit_index": self._state.last_submit_index,
                "last_submit_status": self._state.last_submit_status,
                "submit_in_progress": False,  # sync API; never observably true
                "in_progress_submit_id": None,
                "is_finalized": self._is_finalized,
                "started_at": self._started_at,
            },
        }

    def submit(self, env_instances: list[int]) -> SubmitResult:
        """Run one submit (sync). Writes ``feedback/submit_NNN/`` artifacts
        and updates the in-memory ``SubmitState``.

        Raises:
            RuntimeError: if ``finalize()`` has already been called — the
                run is closed (per submit-protocol.md §2.2 Phase 7).
        """
        if self._is_finalized:
            raise RuntimeError("submit() called after finalize(); run is closed")

        outcome: SubmitOutcome = self._handler.handle(env_instances, self._state)
        self._state = outcome.new_state
        return SubmitResult(
            submit_id=outcome.submit_index,
            status=outcome.status,
            summary=outcome.summary,
        )

    def finalize(self) -> FinalResult:
        """Run held-out evaluation and write ``run.json``. Idempotent.

        Per SPEC §5.2 and output.md §3:

        1. Snapshot the current ``workspace/system/`` (the agent's final
           code — MVP: most-recent-edit, not most-recent-successful-submit
           per SPEC §5.4 since we don't yet retain per-submit snapshots).
        2. Run the policy against all held-out seeds in
           ``env_def.heldout_seeds_path`` (M = 256 for Pendulum, hidden
           from the agent at all times).
        3. Compute normalized score::

               normalized = (mean_held_out - random_baseline)
                          / (expert_baseline - random_baseline)
               final_score = clip(normalized, 0.0, 1.2) * 100

        4. Atomically write ``<run_dir>/run.json`` per output.md §3.1.
        5. Mark the run finalized; further ``submit()`` calls raise.

        Held-out results never reach the agent — they live in
        ``run.json`` outside the workspace.
        """
        if self._is_finalized:
            assert self._final_result is not None
            return self._final_result

        # Mark finalized immediately so failures don't leave the door
        # open for more submits in an indeterminate state.
        self._is_finalized = True

        end_monotonic = time.monotonic()
        now = datetime.now(UTC)
        end_time = (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z"
        )
        wall_time = end_monotonic - self._start_monotonic

        snapshot_dir = snapshot_workspace_system(self._workspace)
        try:
            result_obj: HeldoutResult | None = None
            error_obj: dict[str, Any] | None = None
            try:
                result_obj = evaluate_heldout(
                    snapshot_dir=snapshot_dir,
                    env_def=self._env_def,
                    seed_manager=self._sm,
                    sandbox_config=self._submit_cfg.sandbox,
                )
            except HeldoutError as e:
                error_obj = {
                    "type": "HeldoutError",
                    "message": str(e),
                    "occurred_at_submit": self._state.last_submit_index,
                    "traceback": None,
                }
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        # Identify the agent's "final" submit (SPEC §5.4: most recent ok).
        final_submit_index: int | None = None
        for entry in reversed(self._state.submit_history):
            if entry.status == "ok":
                final_submit_index = entry.submit_index
                break

        # Build outcome.
        if result_obj is not None and result_obj.returns:
            outcome_status = "completed"
            held_out_mean = float(statistics.fmean(result_obj.returns))
            held_out_std = (
                float(statistics.stdev(result_obj.returns))
                if len(result_obj.returns) > 1 else 0.0
            )
            final_score = scoring.final_score(
                held_out_mean,
                expert=self._env_def.expert_baseline,
                random=self._env_def.random_baseline,
            )
            held_out_returns = result_obj.returns
        else:
            outcome_status = "error"
            held_out_mean = None
            held_out_std = None
            final_score = None
            held_out_returns = None

        auxiliary = scoring.build_auxiliary(
            self._state.submit_history,
            expert=self._env_def.expert_baseline,
            random=self._env_def.random_baseline,
            held_out_mean=held_out_mean,
            n_submits=self._state.n_submits,
            n_successful_submits=self._state.n_successful_submits,
            episodes_used=self._state.n_episodes_executed,
        )

        run_doc: dict[str, Any] = {
            "schema_version": "0.1",
            "model": self._model,
            "env": self._env_def.env_id,
            "exp_id": self._exp_id,

            "experiment_dimensions": {
                "episode_budget": self._submit_cfg.episode_budget,
                "min_episodes_per_submit": self._submit_cfg.min_episodes_per_submit,
                "max_episodes_per_submit": self._submit_cfg.max_episodes_per_submit,
                "seed_pool_id": "default",
                "agent_harness": f"hlbench@{hlbench.__version__}",
                "model_config": None,
            },

            "timing": {
                "start_time": self._started_at,
                "end_time": end_time,
                "wall_time_seconds": round(wall_time, 3),
            },

            "outcome": {
                "status": outcome_status,
                "error": error_obj,
                "final_submit_index": final_submit_index,
                "final_score": final_score,
                "held_out_mean_return": held_out_mean,
                "held_out_std_return": held_out_std,
                "held_out_returns": held_out_returns,
                "auxiliary": auxiliary,
            },

            "artifacts": {
                # Relative paths per output.md §3.2.
                "workspace": str(
                    self._workspace.relative_to(self._run_dir)
                ) if _is_subpath(self._workspace, self._run_dir)
                else str(self._workspace),
                "checkpoints": None,    # not produced by MVP
                "logs_harness": None,   # not produced by MVP
                "logs_agent": None,     # not produced by MVP
                "logs_env": None,       # not produced by MVP
            },

            "versions": {
                "harness": hlbench.__version__,
                "env": self._env_def.env_version,
                "agents_md_hash": self._agents_md_hash,
            },
        }

        run_json_path = self._run_dir / "run.json"
        fb.write_summary(run_json_path, run_doc)  # same atomic writer

        self._final_result = FinalResult(
            status=outcome_status,
            final_score=final_score,
            held_out_mean_return=held_out_mean,
            held_out_std_return=held_out_std,
            held_out_returns=held_out_returns,
            final_submit_index=final_submit_index,
            error=error_obj,
            run_json_path=run_json_path,
        )
        return self._final_result


# --------------------------- helpers --------------------------------------


def _is_subpath(path: Path, parent: Path) -> bool:
    """True if ``path`` is below ``parent``. Used so ``run.json:artifacts``
    paths stay relative when the layout is canonical, and absolute when
    the caller put workspace somewhere unexpected."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
