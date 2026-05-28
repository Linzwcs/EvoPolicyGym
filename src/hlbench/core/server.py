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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hlbench
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
# When packaged into a wheel, the AGENT.md may not be at this location;
# callers can override via ``agent_md_path``.
_REPO_ROOT_AGENT_MD = Path(__file__).resolve().parents[3] / "AGENT.md"


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
    """Returned from ``Server.finalize()`` (Day 8)."""

    final_score: float
    held_out_mean_return: float
    held_out_std_return: float


# --------------------------- server ---------------------------------------


class Server:
    """Per-run hlbench server.

    Lifecycle::

        srv = Server(env_id="pendulum", workspace_dir=Path("./run"))
        # workspace/{TASK.md, AGENT.md, system/, feedback/} now exist.
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
        agent_md_path: Path | None = None,
    ) -> None:
        """Initialize the run and stage workspace files.

        Args:
            env_id: Registered env identifier (e.g. ``"pendulum"``).
            workspace_dir: Local directory; created if missing. Server
                ensures ``TASK.md``, ``AGENT.md``, ``system/``, and
                ``feedback/`` exist underneath it. ``system/`` is left
                empty (ready for the agent to drop ``policy.py``);
                ``feedback/`` is empty until the first submit.
            config_overrides: Optional dict to override defaults like
                ``episode_budget``, ``act_wall_s``, ``init_wall_s``.
                Unknown keys raise. None ⇒ defaults from SPEC §1.1.
            agent_md_path: Source ``AGENT.md`` to copy into the workspace.
                Defaults to the repo's ``AGENT.md`` (works in a source
                checkout). If the source file doesn't exist, a minimal
                placeholder is written instead — callers running from a
                wheel should pass an explicit path.
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

        # Stage TASK.md.
        task_dst = self._workspace / "TASK.md"
        if self._env_def.task_md_path and self._env_def.task_md_path.exists():
            shutil.copy(self._env_def.task_md_path, task_dst)
        else:
            task_dst.write_text(
                f"# {env_id}\n\n(no TASK.md template registered for this env)\n"
            )

        # Stage AGENT.md.
        src = agent_md_path or _REPO_ROOT_AGENT_MD
        agent_dst = self._workspace / "AGENT.md"
        if src.exists():
            shutil.copy(src, agent_dst)
        else:
            agent_dst.write_text(
                "# AGENT.md\n\n"
                "(placeholder — install hlbench from source to get the real one)\n"
            )
        self._agent_md_hash = "sha256:" + hashlib.sha256(
            agent_dst.read_bytes()
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
        now = datetime.now(timezone.utc)
        self._started_at = (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z"
        )

    # ------------- public API -------------------------------------------

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
            "agent_md_hash": self._agent_md_hash,

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
        """Trigger held-out evaluation and write run.json (Day 8).

        Currently raises ``NotImplementedError``. Day 7's purpose is to
        validate the train-side loop end-to-end; held-out comes Day 8.
        """
        if self._is_finalized:
            assert self._final_result is not None
            return self._final_result
        raise NotImplementedError(
            "Server.finalize() is not implemented in MVP Day 7; arriving Day 8"
        )