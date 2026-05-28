"""Held-out evaluation: run all M held-out episodes against the final policy.

Per SPEC §5.2 + §6, this happens once at run finalization and is fully
invisible to the agent (no logs, feedback, or /info field reveal it).

We spin up a fresh Sandbox against a snapshot of ``workspace/system/``,
run all seeds from the env's ``heldout.json``, and return the raw returns.
Scoring lives in ``server.py`` so the policy aggregation rule (final
submit selection per §5.4) stays close to the run-state owner.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hlbench.core.sandbox import Sandbox, SandboxConfig, SandboxDead, SandboxInitError
from hlbench.core.seed_manager import SeedManager
from hlbench.envs.registry import EnvDefinition


@dataclass(frozen=True)
class HeldoutResult:
    returns: list[float]
    episode_lengths: list[int]
    n_failed_episodes: int  # episodes where ended_with_error=True


class HeldoutError(Exception):
    """Held-out evaluation could not run (policy failed to init, etc.).

    The Server surfaces this as ``run.json:outcome.status == "error"``
    rather than ``"completed"`` per output.md §3.2."""


def evaluate_heldout(
    *,
    snapshot_dir: Path,
    env_def: EnvDefinition,
    seed_manager: SeedManager,
    sandbox_config: SandboxConfig,
) -> HeldoutResult:
    """Run all held-out episodes against the policy in ``snapshot_dir``.

    Args:
        snapshot_dir: Contains ``system/policy.py`` — typically a copy
            of the final successful submit's snapshot.
        env_def: Used for the env factory, action-space type, and
            ``max_episode_steps``.
        seed_manager: ``held_out_seeds()`` provides the M hidden seeds.
        sandbox_config: Pass-through to ``Sandbox``; usually shares the
            run's submit-time wall-time settings so finalize doesn't
            accidentally accept a policy that would have failed a
            normal submit.

    Returns:
        ``HeldoutResult`` with raw per-episode returns (length M) plus
        episode lengths and a count of episodes that errored mid-flight.

    Raises:
        HeldoutError: if ``Policy.__init__`` raises / times out, or if
            the sandbox child dies before any episode completes. Partial
            success (some episodes ran, some failed) is reported via
            ``n_failed_episodes`` — not raised.
    """
    held_out_seeds = seed_manager.held_out_seeds()
    max_steps = env_def.max_episode_steps

    sandbox = Sandbox(
        snapshot_dir=snapshot_dir,
        env_factory=env_def.factory,
        env_meta=env_def.public_env_meta(),
        config=sandbox_config,
        reward_components=env_def.reward_components,
    )
    try:
        try:
            sandbox.init_policy()
        except SandboxInitError as e:
            raise HeldoutError(
                f"Held-out evaluation failed at Policy init: {e}"
            ) from e

        returns: list[float] = []
        episode_lengths: list[int] = []
        n_failed = 0

        for local_i, real_seed in enumerate(held_out_seeds):
            try:
                rec = sandbox.run_episode(
                    real_seed=real_seed,
                    episode_index=local_i,
                    max_steps=max_steps,
                )
            except SandboxDead as e:
                # Child crashed mid-evaluation. If nothing has completed,
                # that's an outright failure; otherwise it's partial and
                # we surface what we got.
                if not returns:
                    raise HeldoutError(
                        f"Sandbox died before any held-out episode completed: {e}"
                    ) from e
                break
            returns.append(rec.return_)
            episode_lengths.append(rec.length)
            if rec.ended_with_error:
                n_failed += 1

        return HeldoutResult(
            returns=returns,
            episode_lengths=episode_lengths,
            n_failed_episodes=n_failed,
        )
    finally:
        sandbox.close()


def snapshot_workspace_system(workspace_dir: Path) -> Path:
    """Copy ``workspace/system/`` to a fresh temp dir for held-out eval.

    The caller owns cleanup. Returns the parent dir (matching the
    sandbox's ``snapshot_dir`` contract: it expects ``<snapshot_dir>/system/``)."""
    tmp = Path(tempfile.mkdtemp(prefix="hlbench_heldout_"))
    shutil.copytree(workspace_dir / "system", tmp / "system")
    return tmp
