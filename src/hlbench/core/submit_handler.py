"""Submit lifecycle: 7 phases per docs/submit-protocol.md §2.

Phase 1: Request    — validate env_instances list (range, count vs budget)
Phase 2: Snapshot   — copy ``workspace/system/`` to an isolated location
Phase 3: Validate   — ``policy.py`` exists with a ``Policy`` class
                      (denied-import scan and oversize check are post-MVP)
Phase 4: Compile    — Python ``import policy`` (sandboxed)
Phase 5: Initialize — construct ``Policy(obs_space, action_space, env_meta)``
Phase 6: Execute    — run episodes via ``Sandbox.run_episode``
Phase 7: Commit     — write ``summary.json`` atomically

Failure at any phase short-circuits later phases and returns the
appropriate verdict (SPEC.md §4.1, 11-value enum). Phase 1 failures
do NOT consume budget; everything from Phase 2 onward consumes the
full requested ``N`` (see submit-protocol.md §4.1).

This module is the integration point: it owns Sandbox lifecycle,
seed→env-instance resolution, feedback file layout, and SubmitState
bookkeeping. The HTTP layer (Day 7+) is a thin wrapper over
``SubmitHandler.handle``.
"""

from __future__ import annotations

import shutil
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hlbench.core import feedback as fb
from hlbench.core.env_runner import EpisodeRecord
from hlbench.core.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxDead,
    SandboxInitError,
)
from hlbench.core.seed_manager import SeedManager
from hlbench.envs.registry import EnvDefinition


# --------------------------- config / state -------------------------------


@dataclass(frozen=True)
class SubmitConfig:
    """Static run-level limits. Mirrors SPEC §1.1 ``resource_limits`` plus
    budget rules. ``sandbox`` carries the per-process wall-times.

    For MVP, ``system_total_bytes`` and ``denied_imports`` exist as fields
    but are not enforced (Phase 2/3 oversize and denied_import checks are
    post-MVP per architecture.md §4.4)."""

    episode_budget: int = 256
    min_episodes_per_submit: int = 1
    max_episodes_per_submit: int = 256
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


@dataclass(frozen=True)
class SubmitState:
    """Run-level counters that span submits.

    Owned by the Server (Day 7); SubmitHandler reads and returns an
    updated copy. Immutable so callers can keep the previous state for
    rollback / debugging if they wish.

    Fields mirror SPEC §1.1 ``state.*`` plus ``n_episodes_executed``
    (used to compute ``first_global_episode``)."""

    remaining_budget: int
    n_submits: int = 0
    n_successful_submits: int = 0
    n_episodes_executed: int = 0
    last_submit_index: int | None = None
    last_submit_status: str | None = None


@dataclass(frozen=True)
class SubmitOutcome:
    """Return value of ``SubmitHandler.handle``.

    The HTTP /submit response is built from ``summary``; the Server
    persists ``new_state`` for the next call."""

    submit_index: int
    status: str
    summary: dict[str, Any]
    new_state: SubmitState


# --------------------------- handler --------------------------------------


class SubmitHandler:
    """Orchestrates one submit end-to-end. Stateless across calls — the
    Server owns the SubmitState and feeds it in each time."""

    def __init__(
        self,
        *,
        env_def: EnvDefinition,
        seed_manager: SeedManager,
        workspace_dir: Path,
        config: SubmitConfig | None = None,
    ) -> None:
        self._env_def = env_def
        self._sm = seed_manager
        self._workspace = Path(workspace_dir)
        self._config = config or SubmitConfig()

        self._system_dir = self._workspace / "system"
        self._feedback_dir = self._workspace / "feedback"
        self._dir_width = fb.dir_width(self._config.episode_budget)

    # ------------- public entry point ------------------------------------

    def handle(
        self,
        env_instances: list[int],
        state: SubmitState,
    ) -> SubmitOutcome:
        """Run one submit. Always writes ``summary.json`` (and on failure
        ``errors.txt``) under ``feedback/submit_<NNN>/``.

        See submit-protocol.md §3.3 for the verdict→file mapping and
        §4.1 for budget consumption rules."""
        submit_index = state.n_submits
        submit_dir = self._feedback_dir / fb.submit_dir_name(submit_index, self._dir_width)
        submit_dir.mkdir(parents=True, exist_ok=False)

        started_at = fb.now_iso_utc()
        t0 = time.monotonic()

        # ---------- Phase 1: Request validation ----------
        invalid = [i for i in env_instances if not (0 <= i < self._sm.n_env_instances)]
        if invalid:
            return self._fail_pre_consume(
                state, submit_index, submit_dir, env_instances,
                category="invalid_env_instance",
                message=(
                    f"env_instance(s) {invalid} out of range "
                    f"[0, {self._sm.n_env_instances})"
                ),
                started_at=started_at, t0=t0,
            )

        n_req = len(env_instances)
        cfg = self._config
        if (
            n_req < cfg.min_episodes_per_submit
            or n_req > cfg.max_episodes_per_submit
            or n_req > state.remaining_budget
        ):
            return self._fail_pre_consume(
                state, submit_index, submit_dir, env_instances,
                category="budget_invalid",
                message=(
                    f"requested {n_req} episodes; valid range is "
                    f"[{cfg.min_episodes_per_submit}, "
                    f"min({cfg.max_episodes_per_submit}, "
                    f"{state.remaining_budget})]"
                ),
                started_at=started_at, t0=t0,
            )

        # ---------- Phase 2: Snapshot (budget consumed from here) ----------
        snapshot_dir = submit_dir / ".snapshot"
        try:
            shutil.copytree(self._system_dir, snapshot_dir / "system")
        except FileNotFoundError:
            return self._fail_post_consume(
                state, submit_index, submit_dir, env_instances,
                category="missing_policy",
                message=f"workspace/system/ does not exist at {self._system_dir}",
                started_at=started_at, t0=t0,
            )

        # ---------- Phase 3: Static validate ----------
        if not (snapshot_dir / "system" / "policy.py").exists():
            return self._fail_post_consume(
                state, submit_index, submit_dir, env_instances,
                category="missing_policy",
                message="system/policy.py not found in snapshot",
                started_at=started_at, t0=t0,
            )

        # ---------- Phase 4-5: Spawn sandbox + init Policy ----------
        sandbox = Sandbox(
            snapshot_dir=snapshot_dir,
            env_factory=self._env_def.factory,
            env_meta=self._env_def.public_env_meta(),
            config=cfg.sandbox,
            reward_components=self._env_def.reward_components,
        )
        try:
            try:
                sandbox.init_policy()
            except SandboxInitError as e:
                return self._fail_post_consume(
                    state, submit_index, submit_dir, env_instances,
                    category=e.category,
                    message=f"Policy initialization failed: {e}",
                    started_at=started_at, t0=t0,
                    traceback_str=e.traceback_str,
                )

            # ---------- Phase 6: Execute episodes ----------
            episodes_dir = submit_dir / "episodes"
            episodes_dir.mkdir()

            first_global = state.n_episodes_executed
            returns: list[float] = []
            episode_lengths: list[int] = []
            timeouts: list[int] = []
            errors: list[int] = []
            reward_components_per_episode: dict[str, list[float]] = {
                name: [] for name in (self._env_def.reward_components or {})
            }

            max_steps = self._env_def.max_episode_steps
            for local_i, env_inst in enumerate(env_instances):
                global_i = first_global + local_i
                ep_dir = episodes_dir / fb.episode_dir_name(global_i, self._dir_width)
                ep_dir.mkdir()

                real_seed = self._sm.real_seed_for_instance(env_inst)
                try:
                    rec = sandbox.run_episode(
                        real_seed=real_seed,
                        episode_index=local_i,
                        max_steps=max_steps,
                    )
                except SandboxDead as e:
                    # MVP: surface as a fatal at the submit level. Treat as
                    # oom for SPEC purposes (real RSS poll comes post-MVP).
                    # Episodes already written stay; mark partial completion.
                    return self._fail_partial_execute(
                        state, submit_index, submit_dir, env_instances,
                        category="oom",
                        message=f"sandbox died mid-execute: {e}",
                        started_at=started_at, t0=t0,
                        partial_returns=returns,
                        partial_lengths=episode_lengths,
                        partial_timeouts=timeouts,
                        partial_errors=errors,
                        first_global=first_global,
                        episodes_executed=local_i,
                    )

                fb.write_trajectory(ep_dir / "trajectory.jsonl", rec.trajectory)

                if rec.ended_with_error:
                    self._write_episode_error(ep_dir, rec)
                    if rec.error_category == "act_timeout":
                        timeouts.append(local_i)
                    elif rec.error_category in (
                        "act_error", "reset_error", "on_episode_end_error"
                    ):
                        errors.append(local_i)

                returns.append(rec.return_)
                episode_lengths.append(rec.length)

                if reward_components_per_episode:
                    for name in reward_components_per_episode:
                        total = sum(
                            float(step.get("reward_components", {}).get(name, 0.0))
                            for step in rec.trajectory
                        )
                        reward_components_per_episode[name].append(total)
        finally:
            sandbox.close()

        # ---------- Phase 7: Commit ----------
        completed_at = fb.now_iso_utc()
        wall = time.monotonic() - t0

        summary: dict[str, Any] = {
            "schema_version": "0.1",
            "submit_index": submit_index,
            "env": self._env_def.env_id,
            "status": "ok",
            "n_episodes": n_req,
            "first_global_episode": first_global,
            "env_instances": list(env_instances),
            "remaining_budget": state.remaining_budget - n_req,
            "submit_started_at": started_at,
            "submit_completed_at": completed_at,
            "wall_time_seconds": round(wall, 3),
            "returns": returns,
            "mean_return": _mean(returns),
            "std_return": _std(returns),
            "min_return": min(returns) if returns else None,
            "max_return": max(returns) if returns else None,
            "episode_lengths": episode_lengths,
            "mean_episode_length": _mean(episode_lengths),
            "timeouts": timeouts,
            "errors": errors,
        }
        if reward_components_per_episode:
            summary["reward_components_per_episode"] = reward_components_per_episode
            summary["reward_components_mean"] = {
                name: _mean(v) for name, v in reward_components_per_episode.items()
            }
        else:
            summary["reward_components_per_episode"] = None
            summary["reward_components_mean"] = None

        fb.write_summary(submit_dir / "summary.json", summary)

        new_state = replace(
            state,
            remaining_budget=state.remaining_budget - n_req,
            n_submits=state.n_submits + 1,
            n_successful_submits=state.n_successful_submits + 1,
            n_episodes_executed=state.n_episodes_executed + n_req,
            last_submit_index=submit_index,
            last_submit_status="ok",
        )
        return SubmitOutcome(
            submit_index=submit_index, status="ok", summary=summary, new_state=new_state,
        )

    # ------------- failure helpers ---------------------------------------

    def _fail_pre_consume(
        self,
        state: SubmitState,
        submit_index: int,
        submit_dir: Path,
        env_instances: list[int],
        *,
        category: str,
        message: str,
        started_at: str,
        t0: float,
    ) -> SubmitOutcome:
        """Phase 1 failure: budget NOT consumed, no episodes ran."""
        completed_at = fb.now_iso_utc()
        wall = time.monotonic() - t0
        summary = self._minimal_failure_summary(
            submit_index=submit_index,
            status=category,
            env_instances=env_instances,
            n_episodes=len(env_instances),
            remaining_budget=state.remaining_budget,
            first_global_episode=None,
            started_at=started_at,
            completed_at=completed_at,
            wall_time=wall,
        )
        fb.write_summary(submit_dir / "summary.json", summary)
        fb.write_submit_error(
            submit_dir / "errors.txt", category=category, message=message,
        )
        new_state = replace(
            state,
            n_submits=state.n_submits + 1,
            last_submit_index=submit_index,
            last_submit_status=category,
        )
        return SubmitOutcome(
            submit_index=submit_index, status=category, summary=summary, new_state=new_state,
        )

    def _fail_post_consume(
        self,
        state: SubmitState,
        submit_index: int,
        submit_dir: Path,
        env_instances: list[int],
        *,
        category: str,
        message: str,
        started_at: str,
        t0: float,
        traceback_str: str | None = None,
    ) -> SubmitOutcome:
        """Phase 2-5 failure: full N consumed from budget, no episodes ran."""
        completed_at = fb.now_iso_utc()
        wall = time.monotonic() - t0
        n_req = len(env_instances)
        summary = self._minimal_failure_summary(
            submit_index=submit_index,
            status=category,
            env_instances=env_instances,
            n_episodes=n_req,
            remaining_budget=state.remaining_budget - n_req,
            first_global_episode=None,
            started_at=started_at,
            completed_at=completed_at,
            wall_time=wall,
        )
        fb.write_summary(submit_dir / "summary.json", summary)
        fb.write_submit_error(
            submit_dir / "errors.txt",
            category=category, message=message, traceback_str=traceback_str,
        )
        new_state = replace(
            state,
            remaining_budget=state.remaining_budget - n_req,
            n_submits=state.n_submits + 1,
            last_submit_index=submit_index,
            last_submit_status=category,
        )
        return SubmitOutcome(
            submit_index=submit_index, status=category, summary=summary, new_state=new_state,
        )

    def _fail_partial_execute(
        self,
        state: SubmitState,
        submit_index: int,
        submit_dir: Path,
        env_instances: list[int],
        *,
        category: str,
        message: str,
        started_at: str,
        t0: float,
        partial_returns: list[float],
        partial_lengths: list[int],
        partial_timeouts: list[int],
        partial_errors: list[int],
        first_global: int,
        episodes_executed: int,
    ) -> SubmitOutcome:
        """Phase 6 partial failure (oom / submit_wall_exceeded): full N
        consumed, some episodes wrote successfully, both ``episodes/`` and
        ``errors.txt`` exist (the only verdict pair where this happens —
        SPEC §4.4.4 / submit-protocol §3.3)."""
        completed_at = fb.now_iso_utc()
        wall = time.monotonic() - t0
        n_req = len(env_instances)
        summary: dict[str, Any] = {
            "schema_version": "0.1",
            "submit_index": submit_index,
            "env": self._env_def.env_id,
            "status": category,
            "n_episodes": n_req,
            "first_global_episode": first_global if partial_returns else None,
            "env_instances": list(env_instances),
            "remaining_budget": state.remaining_budget - n_req,
            "submit_started_at": started_at,
            "submit_completed_at": completed_at,
            "wall_time_seconds": round(wall, 3),
            "returns": partial_returns or None,
            "mean_return": _mean(partial_returns) if partial_returns else None,
            "std_return": _std(partial_returns) if partial_returns else None,
            "min_return": min(partial_returns) if partial_returns else None,
            "max_return": max(partial_returns) if partial_returns else None,
            "episode_lengths": partial_lengths or None,
            "mean_episode_length": _mean(partial_lengths) if partial_lengths else None,
            "timeouts": partial_timeouts if partial_returns else None,
            "errors": partial_errors if partial_returns else None,
            "reward_components_mean": None,
            "reward_components_per_episode": None,
        }
        fb.write_summary(submit_dir / "summary.json", summary)
        fb.write_submit_error(
            submit_dir / "errors.txt", category=category, message=message,
        )
        new_state = replace(
            state,
            remaining_budget=state.remaining_budget - n_req,
            n_submits=state.n_submits + 1,
            n_episodes_executed=state.n_episodes_executed + episodes_executed,
            last_submit_index=submit_index,
            last_submit_status=category,
        )
        return SubmitOutcome(
            submit_index=submit_index, status=category, summary=summary, new_state=new_state,
        )

    def _minimal_failure_summary(
        self,
        *,
        submit_index: int,
        status: str,
        env_instances: list[int],
        n_episodes: int,
        remaining_budget: int,
        first_global_episode: int | None,
        started_at: str,
        completed_at: str,
        wall_time: float,
    ) -> dict[str, Any]:
        """Per SPEC §4.1.1: on failure all array/aggregate fields are ``null``
        (not ``[]``), so ``status == "ok"`` is the trivial "did this run?"
        check."""
        return {
            "schema_version": "0.1",
            "submit_index": submit_index,
            "env": self._env_def.env_id,
            "status": status,
            "n_episodes": n_episodes,
            "first_global_episode": first_global_episode,
            "env_instances": list(env_instances),
            "remaining_budget": remaining_budget,
            "submit_started_at": started_at,
            "submit_completed_at": completed_at,
            "wall_time_seconds": round(wall_time, 3),
            "returns": None,
            "mean_return": None,
            "std_return": None,
            "min_return": None,
            "max_return": None,
            "episode_lengths": None,
            "mean_episode_length": None,
            "timeouts": None,
            "errors": None,
            "reward_components_mean": None,
            "reward_components_per_episode": None,
        }

    # ------------- per-episode error.txt --------------------------------

    def _write_episode_error(self, ep_dir: Path, rec: EpisodeRecord) -> None:
        """Translate ``EpisodeRecord.error_*`` into the SPEC §4.4.3 schema."""
        cat = rec.error_category or "act_error"
        step = rec.error_step_index
        if cat == "act_timeout":
            wall_ms = int(self._config.sandbox.act_wall_s * 1000)
            msg = f"act() exceeded {wall_ms}ms wall time at step {step}"
        elif cat == "act_error":
            msg = f"act() raised at step {step}"
        elif cat == "reset_error":
            msg = "Policy.reset() raised"
        elif cat == "on_episode_end_error":
            msg = "Policy.on_episode_end() raised"
        else:  # pragma: no cover (defensive)
            msg = f"episode failed: {cat}"

        fb.write_episode_error(
            ep_dir / "error.txt",
            category=cat,
            message=msg,
            step_index=step,
            traceback_str=rec.error_traceback,
        )


# --------------------------- helpers --------------------------------------


def _mean(xs: list[float] | list[int]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _std(xs: list[float]) -> float | None:
    """Sample stdev (ddof=1) when N ≥ 2; ``0.0`` for single episode; ``None``
    when empty. Matches what most numpy users expect from ``std``."""
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return float(statistics.stdev(xs))
