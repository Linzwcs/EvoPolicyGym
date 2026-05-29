"""Sandbox for executing agent's Policy code.

MVP strategy (per docs/architecture.md §4.4):
- Subprocess via `multiprocessing` ("spawn" context for cross-platform parity)
- `signal.setitimer(ITIMER_REAL, ...)` inside the subprocess for `act()` wall-time
- Parent-side `Pipe.poll(timeout)` for `init`/episode wall-time
- `resource.setrlimit(RLIMIT_AS, ...)` for memory cap (best-effort on macOS)
- `_DeniedImportFinder` on `sys.meta_path` enforces AGENTS.md §3.2
  denied-imports list (0.1.0a1; full list in ``DENIED_IMPORTS``)
- No network blocking yet (post-MVP)

A new Sandbox is spawned per submit and torn down after. The child holds
exactly one Policy instance; all episodes in a submit share that instance,
matching SPEC.md §2 ("Policy persists across episodes within a submit").

Failure → verdict mapping (SPEC.md §4.1, §4.4):

    Stage in submit       | Failure                  | Category
    ----------------------|--------------------------|------------------
    Sandbox.init_policy() | child died before ack    | import_error
                          | denied import attempted  | denied_import
                          | Policy.__init__ raised   | init_error
                          | Policy() exceeded        | init_timeout
                          |   init_wall_s            |
    Sandbox.run_episode() | policy.act raised        | act_error (per-ep)
                          | policy.act over          | act_timeout (per-ep)
                          |   act_wall_s             |
                          | env.reset raised         | reset_error (per-ep)
                          | child died unexpectedly  | oom (submit-level —
                          |                          |   parent surfaces it)

Failures classified as "per-episode" are reported on the returned
`EpisodeRecord` (ended_with_error=True with the category); the sandbox
itself does NOT raise on these — SubmitHandler keeps submitting more
episodes. Submit-level failures raise `SandboxInitError` or
`SandboxDead` for the caller to translate into a verdict.
"""

from __future__ import annotations

import contextlib
import io
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


# ----------------------------- denied imports ------------------------------


#: AGENTS.md §3.2 default denied list (enforcement landed in 0.1.0a1).
#: Matched against the importing module name and every dotted prefix,
#: so listing ``"urllib"`` also blocks ``urllib.request``. Listing
#: ``"google.genai"`` blocks the sub-package without blocking the
#: ``google`` namespace (which other allowed libs may need).
#:
#: Limitation (v0.1.0a1): module-level only. AGENTS.md §3.2 also lists
#: ``os.system`` / ``os.exec*`` as forbidden, but those are function
#: attributes on the (allowed) ``os`` module. Function-level blocking
#: requires post-import monkeypatching and is deferred to a future
#: hardening pass.
DENIED_IMPORTS: frozenset[str] = frozenset({
    # Pretrained ML — trivially load model weights, violates Core Invariant.
    "transformers", "huggingface_hub", "timm", "diffusers",
    # External LLM APIs — bypass submit interface entirely.
    "openai", "anthropic", "google.genai", "cohere",
    # RL frameworks bundling checkpoints / parallel rollout workers.
    "stable_baselines3", "ray", "rllib", "cleanrl", "sb3_contrib",
    # Network access — violates AGENTS.md §3.1.
    "urllib", "requests", "socket", "httpx", "aiohttp",
    # Sandbox escape vectors.
    "subprocess",
})


class _DeniedImport(ImportError):
    """Raised by ``_DeniedImportFinder`` for a denied module name.

    Subclass of ``ImportError`` so policy code that wraps imports in
    ``try: import x; except ImportError`` still catches it (no leakage
    of the finder implementation). The sandbox catches this subclass
    specifically and maps to the ``denied_import`` verdict per SPEC §4.1.
    """


class _DeniedImportFinder:
    """``sys.meta_path`` finder that vetoes denied imports.

    Installed once at child startup before ``system/`` is added to
    ``sys.path``, so it sees every import the policy attempts. Returns
    ``None`` for allowed modules to let the regular finders handle them.
    """

    def __init__(self, denied: frozenset[str]) -> None:
        self._denied = denied

    def find_spec(
        self, name: str, path: Any = None, target: Any = None,
    ) -> None:
        # Block name and every dotted prefix.
        parts = name.split(".")
        for i in range(len(parts)):
            prefix = ".".join(parts[: i + 1])
            if prefix in self._denied:
                raise _DeniedImport(
                    f"Module {name!r} is on the denied list (see AGENTS.md §3.2)"
                )
        return None  # let regular finders handle



# ----------------------------- child worker --------------------------------


class _ActTimeout(Exception):
    """Raised inside the child by SIGALRM when act() exceeds act_wall_s."""


def _sigalrm_handler(_signum: int, _frame: Any) -> None:
    raise _ActTimeout()


class _StreamCapture:
    """Replace ``sys.stdout`` / ``sys.stderr`` with ``io.StringIO`` so the
    child can hand the captured text back to the parent for SPEC §4.5
    ``stdout.txt`` / ``stderr.txt`` files.

    The capture buffers are NOT reset by ``install()`` — anything written
    between install and the first ``swap()`` (i.e., Policy.__init__
    output) is folded into the first episode's capture, matching the
    SPEC requirement that "anything Policy.__init__ prints goes into the
    first episode's stdout.txt".
    """

    def __init__(self) -> None:
        self._real_stdout: Any = None
        self._real_stderr: Any = None
        self._stdout_buf = io.StringIO()
        self._stderr_buf = io.StringIO()

    def install(self) -> None:
        self._real_stdout = sys.stdout
        self._real_stderr = sys.stderr
        sys.stdout = self._stdout_buf
        sys.stderr = self._stderr_buf

    def swap(self) -> tuple[str, str]:
        """Return current buffer contents, install fresh buffers."""
        out = self._stdout_buf.getvalue()
        err = self._stderr_buf.getvalue()
        self._stdout_buf = io.StringIO()
        self._stderr_buf = io.StringIO()
        sys.stdout = self._stdout_buf
        sys.stderr = self._stderr_buf
        return out, err

    def uninstall(self) -> None:
        if self._real_stdout is not None:
            sys.stdout = self._real_stdout
        if self._real_stderr is not None:
            sys.stderr = self._real_stderr


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


def _child_main(
    conn: Any,
    snapshot_dir: Path,
    env_factory: Any,
    env_meta: dict[str, Any],
    act_wall_s: float,
    max_rss_bytes: int | None,
    record_obs: bool,
    reward_components: dict[str, str] | None,
    denied_imports: frozenset[str],
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
        with contextlib.suppress(ValueError, OSError):
            # not supported on this platform / value; ignore
            resource.setrlimit(resource.RLIMIT_AS, (max_rss_bytes, max_rss_bytes))

    # SIGALRM handler for act() wall-time.
    signal.signal(signal.SIGALRM, _sigalrm_handler)

    # Install denied-import hook BEFORE adding system/ to sys.path so
    # the policy can't slip imports past it via load-time aliasing.
    #
    # Limitation: Python's import machinery checks sys.modules before
    # calling meta_path finders, so denied modules that Python startup
    # already imported (`subprocess` via multiprocessing, transitively
    # `urllib`/`socket` via SSL/email) bypass our finder when re-imported.
    # We don't evict them — eviction breaks gymnasium and other env
    # plumbing that legitimately uses those stdlib modules. The finder
    # cleanly blocks third-party pretrained / external-API modules
    # (transformers, requests, httpx, …) and submodule paths
    # (`urllib.request` triggers because the submodule isn't cached);
    # full network blocking is a deeper hardening pass.
    if denied_imports:
        sys.meta_path.insert(0, _DeniedImportFinder(denied_imports))

    # Make `system/policy.py` importable as bare `policy`. SPEC §2: the
    # submitted code lives in system/ and is the only writable dir.
    system_dir = str((snapshot_dir / "system").resolve())
    sys.path.insert(0, system_dir)

    # Install stdout/stderr capture right before the policy runs.
    # __init__ output gets folded into the first episode's capture per
    # SPEC §4.5 (we don't swap between init_done and the first
    # run_episode, so the same buffer keeps accumulating).
    capture = _StreamCapture()
    capture.install()

    # 1. Initialize: import policy module + construct Policy().
    try:
        import policy as _policy_module  # type: ignore[import-not-found]
        PolicyClass = _policy_module.Policy
        inner_policy = PolicyClass(
            obs_space=env_meta.get("obs_space"),
            action_space=env_meta.get("action_space"),
            env_meta=env_meta,
        )
    except _DeniedImport:
        # Order matters: _DeniedImport is an ImportError subclass, must
        # be caught first to override the generic import_error verdict.
        # Uninstall capture so the parent's logs aren't poisoned by
        # whatever we send back next.
        capture.uninstall()
        conn.send(("init_error", {
            "category": "denied_import",
            "traceback_str": traceback.format_exc(),
        }))
        return
    except ImportError:
        capture.uninstall()
        conn.send(("init_error", {
            "category": "import_error",
            "traceback_str": traceback.format_exc(),
        }))
        return
    except Exception:
        capture.uninstall()
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
                # Drain the capture buffers into the record. First
                # iteration also picks up any Policy.__init__ output.
                stdout_text, stderr_text = capture.swap()
                rec.stdout_captured = stdout_text
                rec.stderr_captured = stderr_text
                conn.send(("ok", {"event": "episode_done", "record": rec}))
                continue

            conn.send(("ok", {"event": "unknown_command", "cmd": cmd}))
    finally:
        capture.uninstall()
        with contextlib.suppress(Exception):
            env.close()
        conn.close()


# ----------------------------- parent --------------------------------------


@dataclass
class SandboxConfig:
    """Tunables for a Sandbox; all wall-times in seconds."""

    init_wall_s: float = 30.0
    act_wall_s: float = 1.0
    episode_wall_s: float = 60.0
    #: Address-space cap (RLIMIT_AS) for the policy subprocess.
    #:
    #: Default ``None`` (disabled) is intentional. Two reasons RLIMIT_AS
    #: is a poor knob to enforce by default:
    #:   - macOS: Apple's malloc generally ignores RLIMIT_AS — limit is
    #:     silently no-op, not actually enforced.
    #:   - Linux: RLIMIT_AS limits *virtual address space*, not RSS;
    #:     loading numpy + gymnasium + box2d + mujoco can mmap many GiB
    #:     of shared-lib pages even when actual RSS stays low. A 1 GiB
    #:     cap that "looks reasonable" actually kills child processes
    #:     during ``import`` (visible upstream as ``SandboxDead: child
    #:     closed pipe before init ack``).
    #:
    #: Operators wanting real memory caps should use a cgroup or set
    #: this to a comfortably large value (e.g. ``8 << 30`` for 8 GiB)
    #: based on their machine. None by default = trust the OS. The
    #: ``submit_peak_rss_bytes`` SPEC field is exposed via ``GET /info``
    #: for downstream observability — the actual enforcement of
    #: per-submit RSS via psutil polling is post-MVP.
    max_rss_bytes: int | None = None
    #: Module names blocked at import time (AGENTS.md §3.2). Default is the
    #: full SPEC list; tests can pass ``frozenset()`` to disable.
    denied_imports: frozenset[str] = DENIED_IMPORTS


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
                self._config.denied_imports,
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
            # Capture child exit code + a hint about likely causes before
            # terminating — saves operators from chasing this blind.
            self._proc.join(timeout=1.0)
            exitcode = self._proc.exitcode
            self._terminate()
            hint = ""
            if exitcode is not None and exitcode < 0:
                # On POSIX, negative exitcode = -signum (process killed
                # by signal). SIGKILL=9 typically = OS OOM-killer or
                # RLIMIT_AS hit. SIGSEGV=11 = native-lib crash.
                signum = -exitcode
                if signum == 9:
                    hint = (
                        " (exitcode=-9 SIGKILL — likely RLIMIT_AS too "
                        "tight or OS OOM-killer; try Server(config_overrides="
                        "{'max_rss_bytes': None}) or a larger value)"
                    )
                elif signum == 11:
                    hint = " (exitcode=-11 SIGSEGV — native lib crash; check env install)"
                else:
                    hint = f" (exitcode={exitcode} signal={signum})"
            elif exitcode is not None and exitcode != 0:
                hint = f" (exitcode={exitcode})"
            raise SandboxDead(f"child closed pipe before init ack{hint}") from e

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
        `error_category` ∈ {"act_error", "act_timeout", "reset_error"} —
        those are per-episode failures, NOT sandbox failures, so this
        method does not raise on them.

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
            with contextlib.suppress(BrokenPipeError, OSError):
                self._parent_conn.send(("close", {}))
            self._proc.join(timeout=2.0)

        self._terminate()

    def _terminate(self) -> None:
        """Force-kill the child if still alive, then close the pipe."""
        with contextlib.suppress(Exception):
            self._parent_conn.close()
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1.0)
