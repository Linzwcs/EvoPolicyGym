"""Sandbox for executing agent's Policy code.

MVP strategy (per docs/architecture.md §4.4):
- Subprocess via `multiprocessing` ("spawn" context for cross-platform parity)
- `signal.setitimer(ITIMER_REAL, ...)` inside the subprocess for `act()` wall-time
- Parent-side `Pipe.poll(timeout)` for `init`/episode wall-time
- `resource.setrlimit(RLIMIT_AS, ...)` for memory cap (best-effort on macOS)
- No `denied_imports` enforcement, no network blocking (post-MVP bundle)

A new Sandbox is spawned per submit and torn down after. The child holds
exactly one Policy instance; all episodes in a submit share that instance,
matching SPEC.md §2 ("Policy persists across episodes within a submit").

Failure → verdict mapping (SPEC.md §4.1, §4.4):

    Stage in submit       | Failure                | Category
    ----------------------|------------------------|------------------
    Sandbox.init_policy() | child died before ack  | import_error
                          | Policy.__init__ raised | init_error
                          | Policy() exceeded      | init_timeout
                          |   init_wall_s          |
    Sandbox.run_episode() | policy.act raised      | act_error (per-ep)
                          | policy.act over        | act_timeout (per-ep)
                          |   act_wall_s           |
                          | env.reset raised       | reset_error (per-ep)
                          | child died unexpectedly| oom (submit-level —
                          |                        |   parent surfaces it)

Failures classified as "per-episode" are reported on the returned
`EpisodeRecord` (ended_with_error=True with the category); the sandbox
itself does NOT raise on these — SubmitHandler keeps submitting more
episodes. Submit-level failures raise `SandboxInitError` or
`SandboxDead` for the caller to translate into a verdict.
"""

from __future__ import annotations

import multiprocessing as mp
import resource
import signal
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.core.env_runner import EpisodeRecord, run_episode


# ----------------------------- exceptions ----------------------------------


class SandboxInitError(Exception):
    """Policy could not be instantiated. Maps to submit-level verdict.

    `category` is one of "init_error", "init_timeout", "import_error".
    `traceback_str` is the Python traceback as captured in the child, or
    a synthetic message for `init_timeout` / `import_error` where no
    traceback is available.
    """

    def __init__(self, category: str, traceback_str: str) -> None:
        self.category = category
        self.traceback_str = traceback_str
        last_line = (traceback_str or "").strip().splitlines()
        suffix = last_line[-1] if last_line else "(no traceback)"
        super().__init__(f"[{category}] {suffix}")


class SandboxDead(Exception):
    """Child process died unexpectedly (segfault, OOM kill, etc.).

    Caller should map to `oom` verdict if exit-code matches an OOM signal,
    else a generic crash. MVP doesn't differentiate — leaves that to
    SubmitHandler (Day 6).
    """


# ----------------------------- child worker --------------------------------


class _ActTimeout(Exception):
    """Raised inside the child by SIGALRM when act() exceeds act_wall_s."""


def _sigalrm_handler(_signum: int, _frame: Any) -> None:
    raise _ActTimeout()


class _TimedPolicy:
    """Wrap a policy so act() runs under a SIGALRM-based wall-time guard."""

    def __init__(self, inner: Any, act_wall_s: float) -> None:
        self._inner = inner
        self._wall_s = act_wall_s

    def reset(self, episode_index: int) -> None:
        if hasattr(self._inner, "reset"):
            self._inner.reset(episode_index)

    def act(self, obs: Any) -> Any:
        # setitimer takes float seconds; resolution is OS-dependent but
        # ~ms on modern macOS/Linux. 0 disables.
        signal.setitimer(signal.ITIMER_REAL, self._wall_s)
        try:
            return self._inner.act(obs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def on_episode_end(self, episode_return: float) -> None:
        if hasattr(self._inner, "on_episode_end"):
            self._inner.on_episode_end(episode_return)


def _child_main(
    conn: Any,
    snapshot_dir: Path,
    env_factory: Any,
    env_meta: dict[str, Any],
    act_wall_s: float,
    max_rss_bytes: int | None,
    record_obs: bool,
    reward_components: dict[str, str] | None,
) -> None:
    """Subprocess entry point. Owns one Policy + one env instance.

    Protocol (all messages are pickled tuples):
        parent → child:  ("init", {})
                         ("run_episode", {"real_seed": int,
                                          "episode_index": int,
                                          "max_steps": int})
                         ("close", {})
        child → parent:  ("ok", payload)
                         ("init_error", {"category", "traceback_str"})

    The child exits its loop on a "close" message OR when the parent
    closes the pipe. On any unhandled exception it sends a final error
    message and exits — the parent will treat unexpected exits as
    `SandboxDead`.
    """
    # Memory cap. Best-effort on macOS (RLIMIT_AS is often unenforced
    # under Apple's malloc). On Linux it actually caps virtual address
    # space and OOMs allocate calls.
    if max_rss_bytes is not None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (max_rss_bytes, max_rss_bytes))
        except (ValueError, OSError):
            pass  # not supported on this platform / value; ignore

    # SIGALRM handler for act() wall-time.
    signal.signal(signal.SIGALRM, _sigalrm_handler)

    # Make `system/policy.py` importable as bare `policy`. SPEC §2: the
    # submitted code lives in system/ and is the only writable dir.
    system_dir = str((snapshot_dir / "system").resolve())
    sys.path.insert(0, system_dir)

    # 1. Initialize: import policy module + construct Policy().
    try:
        import policy as _policy_module  # type: ignore[import-not-found]
        PolicyClass = getattr(_policy_module, "Policy")
        inner_policy = PolicyClass(
            obs_space=env_meta.get("obs_space"),
            action_space=env_meta.get("action_space"),
            env_meta=env_meta,
        )
    except ImportError:
        conn.send(("init_error", {
            "category": "import_error",
            "traceback_str": traceback.format_exc(),
        }))
        return
    except Exception:
        conn.send(("init_error", {
            "category": "init_error",
            "traceback_str": traceback.format_exc(),
        }))
        return

    timed_policy = _TimedPolicy(inner_policy, act_wall_s)
    env = env_factory()
    action_space_type = env_meta["action_space"]["type"]

    conn.send(("ok", {"event": "init_done"}))

    # 2. Main loop: dispatch commands.
    try:
        while True:
            try:
                cmd, args = conn.recv()
            except EOFError:
                break

            if cmd == "close":
                conn.send(("ok", {"event": "closing"}))
                break

            if cmd == "run_episode":
                rec = run_episode(
                    timed_policy,
                    env,
                    real_seed=args["real_seed"],
                    episode_index=args["episode_index"],
                    action_space_type=action_space_type,
                    max_steps=args["max_steps"],
                    record_obs=record_obs,
                    reward_components=reward_components,
                    act_timeout_exc_class=_ActTimeout,
                )
                conn.send(("ok", {"event": "episode_done", "record": rec}))
                continue

            conn.send(("ok", {"event": "unknown_command", "cmd": cmd}))
    finally:
        try:
            env.close()
        except Exception:
            pass
        conn.close()


# ----------------------------- parent --------------------------------------


@dataclass
class SandboxConfig:
    """Tunables for a Sandbox; all wall-times in seconds."""

    init_wall_s: float = 30.0
    act_wall_s: float = 1.0
    episode_wall_s: float = 60.0
    max_rss_bytes: int | None = 1 << 30  # 1 GiB; None disables


class Sandbox:
    """Subprocess sandbox holding one Policy instance.

    Lifecycle within a submit:
        sb = Sandbox(snapshot_dir=..., env_factory=..., env_meta=...)
        sb.init_policy()                          # raises SandboxInitError
        rec = sb.run_episode(real_seed=...)       # may raise SandboxDead
        ...
        sb.close()

    The sandbox is NOT reusable across submits — `close()` is final.
    """

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        env_factory: Any,
        env_meta: dict[str, Any],
        config: SandboxConfig | None = None,
        record_obs: bool | None = None,
        reward_components: dict[str, str] | None = None,
    ) -> None:
        self._config = config or SandboxConfig()
        if record_obs is None:
            record_obs = env_meta.get("obs_storage", "inline") == "inline"

        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(
            target=_child_main,
            args=(
                child_conn,
                Path(snapshot_dir),
                env_factory,
                env_meta,
                self._config.act_wall_s,
                self._config.max_rss_bytes,
                record_obs,
                reward_components,
            ),
            daemon=True,
        )
        self._proc.start()
        # We don't need our copy of the child's pipe end.
        child_conn.close()

        self._initialized = False
        self._closed = False

    def init_policy(self) -> None:
        """Wait for the child to finish `Policy.__init__`.

        Raises:
            SandboxInitError(category="init_timeout") if no message arrives
                within `init_wall_s`.
            SandboxInitError(category="init_error"|"import_error") if the
                child sent back an init failure.
            SandboxDead if the child died before sending anything.
        """
        if self._initialized:
            raise RuntimeError("init_policy() already called")

        if not self._parent_conn.poll(self._config.init_wall_s):
            self._terminate()
            raise SandboxInitError(
                "init_timeout",
                f"Policy.__init__ exceeded {self._config.init_wall_s}s wall time",
            )

        try:
            tag, payload = self._parent_conn.recv()
        except EOFError as e:
            self._terminate()
            raise SandboxDead("child closed pipe before init ack") from e

        if tag == "init_error":
            self._terminate()
            raise SandboxInitError(payload["category"], payload["traceback_str"])
        if tag != "ok":
            self._terminate()
            raise SandboxDead(f"unexpected init reply: {tag!r}")

        self._initialized = True

    def run_episode(
        self,
        *,
        real_seed: int,
        episode_index: int,
        max_steps: int,
    ) -> EpisodeRecord:
        """Run one episode in the child, blocking until done.

        The returned EpisodeRecord may have `ended_with_error=True` with
        `error_category` ∈ {"act_error", "act_timeout", "reset_error",
        "on_episode_end_error"} — those are per-episode failures, NOT
        sandbox failures, so this method does not raise on them.

        Raises:
            SandboxDead: child died mid-episode (e.g. OOM, segfault, or
                no reply within `episode_wall_s`).
            RuntimeError: caller forgot init_policy() or already close()d.
        """
        if not self._initialized:
            raise RuntimeError("must call init_policy() before run_episode()")
        if self._closed:
            raise RuntimeError("sandbox is closed")

        self._parent_conn.send((
            "run_episode",
            {"real_seed": real_seed, "episode_index": episode_index, "max_steps": max_steps},
        ))

        if not self._parent_conn.poll(self._config.episode_wall_s):
            self._terminate()
            raise SandboxDead(
                f"episode exceeded {self._config.episode_wall_s}s wall time"
            )

        try:
            tag, payload = self._parent_conn.recv()
        except EOFError as e:
            self._terminate()
            raise SandboxDead("child closed pipe during episode") from e

        if tag != "ok" or payload.get("event") != "episode_done":
            self._terminate()
            raise SandboxDead(f"unexpected episode reply: {tag!r} {payload!r}")

        rec = payload["record"]
        if not isinstance(rec, EpisodeRecord):  # pragma: no cover
            raise SandboxDead(f"child returned non-EpisodeRecord: {type(rec).__name__}")
        return rec

    def close(self) -> None:
        """Terminate the child cleanly. Idempotent."""
        if self._closed:
            return
        self._closed = True

        if self._proc.is_alive():
            try:
                self._parent_conn.send(("close", {}))
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=2.0)

        self._terminate()

    def _terminate(self) -> None:
        """Force-kill the child if still alive, then close the pipe."""
        try:
            self._parent_conn.close()
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1.0)
