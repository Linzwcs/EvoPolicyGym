"""Harness loop runner.

Owns the harness-side state machine:

  1. Spin up an in-process ``Server`` for the requested env.
  2. Compose the initial prompt from the env's TASK.md + GET /info.
  3. Hand the first turn to the agent.
  4. Loop:
       - inspect ``Server.info()`` — has the agent submitted? finalized?
       - if budget exhausted but not finalized: nudge the agent to
         POST /finalize on the next turn,
       - if max_turns reached: force-finalize and exit,
       - otherwise: build a continuation prompt and run another turn.
  5. Write ``<run_dir>/logs/harness_runner.json`` summarising the run
     (turn count, final score, per-turn duration, agent verdicts).

The runner is deliberately separated from the CLI entry point so tests
can inject a fake agent (see ``tests/test_harness_runner.py``)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from hlbench.core.server import FinalResult, Server
from hlbench_harness.prompts import (
    AGENTS_MD_EXCERPT,
    compose_continuation_prompt,
    compose_initial_prompt,
)
from hlbench_harness.state import TurnObservation, observe

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


class AgentLike(Protocol):
    """Minimal agent surface the runner depends on. Both
    ``ClaudeAgent`` and the test-only ``FakeAgent`` satisfy this."""

    @property
    def session_id(self) -> str: ...
    @property
    def turn_count(self) -> int: ...

    def run_turn(self, prompt: str) -> Any: ...


@dataclass
class TurnLogEntry:
    """One row in ``harness_runner.json:turns``."""

    turn_index: int
    duration_seconds: float
    exit_code: int
    timed_out: bool
    text_len: int
    state_after: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Returned by ``HarnessRunner.run()`` and persisted to
    ``logs/harness_runner.json``.

    ``termination_reason`` is one of (priority order if multiple apply):

      - ``budget_exhausted`` — preferred natural termination; remaining
        budget hit 0 after the most recent turn.
      - ``agent_finalized`` — defensive: the agent called POST /finalize
        despite the prompt telling it not to. Rare.
      - ``consecutive_failures`` — N back-to-back failed agent turns.
      - ``max_turns`` — turn cap hit before budget exhausted (fallback).

    The harness always calls ``Server.finalize()`` after the loop ends,
    regardless of reason, so ``run.json`` is always written."""

    session_id: str
    n_turns: int
    termination_reason: str
    final_result: dict[str, Any] | None
    turns: list[TurnLogEntry] = field(default_factory=list)
    started_at_monotonic: float = 0.0
    ended_at_monotonic: float = 0.0

    def to_record(self) -> dict[str, Any]:
        out = asdict(self)
        out["wall_time_seconds"] = round(
            self.ended_at_monotonic - self.started_at_monotonic, 3
        )
        del out["started_at_monotonic"]
        del out["ended_at_monotonic"]
        return out


# ---------------------------------------------------------------------------


class HarnessRunner:
    """Drives one full agent run end-to-end.

    Lifecycle::

        runner = HarnessRunner(server, agent, http_url="http://...",
                                max_turns=12)
        summary = runner.run()       # blocks until finalize or exit cond

    The runner does NOT start the HTTP server — that's the caller's job
    (CLI wires it via ``HlbenchHTTPServer`` background thread). The
    runner only needs the URL so it can embed it in the agent's prompts.
    """

    def __init__(
        self,
        *,
        server: Server,
        agent: AgentLike,
        http_url: str,
        max_turns: int = 12,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._server = server
        self._agent = agent
        self._http_url = http_url
        self._max_turns = max_turns
        self._max_consec_failures = max_consecutive_failures

    def run(self) -> RunSummary:
        """Drive the loop until terminal, then auto-finalize.

        Termination priority (after each turn):

          1. ``remaining_budget == 0`` → ``budget_exhausted``. Preferred
             natural termination.
          2. Server reports ``is_finalized == true`` → ``agent_finalized``.
             Defensive: agent shouldn't be calling /finalize per the prompt.
          3. N consecutive failed agent turns → ``consecutive_failures``.
          4. ``max_turns`` reached without any of the above → ``max_turns``.

        The harness always calls ``Server.finalize()`` after the loop
        regardless of reason — ``run.json`` always exists."""
        started = time.monotonic()
        turns: list[TurnLogEntry] = []
        termination_reason = "max_turns"  # default if loop exhausts naturally
        consecutive_failures = 0

        for turn_idx in range(self._max_turns):
            prompt = self._build_prompt(turn_idx)
            log.info("turn %d: invoking agent", turn_idx)
            try:
                result = self._agent.run_turn(prompt)
            except Exception:  # pragma: no cover (subprocess-level surprise)
                log.exception("turn %d: agent raised; aborting loop", turn_idx)
                consecutive_failures += 1
                if consecutive_failures >= self._max_consec_failures:
                    termination_reason = "consecutive_failures"
                    break
                continue

            # Snapshot live server state immediately after the turn so
            # the entry captures what the agent's actions produced.
            post_info = self._server.info()
            entry = TurnLogEntry(
                turn_index=turn_idx,
                duration_seconds=getattr(result, "duration_seconds", 0.0),
                exit_code=getattr(result, "exit_code", 0),
                timed_out=getattr(result, "timed_out", False),
                text_len=len(getattr(result, "text", "")),
                state_after={
                    "remaining_budget": post_info["state"]["remaining_budget"],
                    "n_submits": post_info["state"]["n_submits"],
                    "n_successful_submits": post_info["state"]["n_successful_submits"],
                    "last_submit_status": post_info["state"]["last_submit_status"],
                    "is_finalized": post_info["state"]["is_finalized"],
                },
            )
            turns.append(entry)

            ok = getattr(result, "ok", entry.exit_code == 0 and not entry.timed_out)
            if not ok:
                consecutive_failures += 1
                log.warning(
                    "turn %d: agent unhealthy (exit=%d timed_out=%s); failures=%d/%d",
                    turn_idx, entry.exit_code, entry.timed_out,
                    consecutive_failures, self._max_consec_failures,
                )
                if consecutive_failures >= self._max_consec_failures:
                    termination_reason = "consecutive_failures"
                    break
            else:
                consecutive_failures = 0

            # Defensive — the prompt tells the agent NOT to call /finalize,
            # but if it did anyway we honor the request.
            if entry.state_after["is_finalized"]:
                termination_reason = "agent_finalized"
                log.info("turn %d: agent unexpectedly finalized", turn_idx)
                break

            # Primary natural termination: budget exhausted.
            if entry.state_after["remaining_budget"] <= 0:
                termination_reason = "budget_exhausted"
                log.info("turn %d: budget exhausted; harness will finalize", turn_idx)
                break

        # Always finalize. ``Server.finalize()`` is idempotent so the
        # ``agent_finalized`` branch above is harmless.
        post_info = self._server.info()
        if not post_info["state"]["is_finalized"]:
            log.info("auto-finalize: termination_reason=%s", termination_reason)
        final: FinalResult = self._server.finalize()

        summary = RunSummary(
            session_id=self._agent.session_id,
            n_turns=len(turns),
            termination_reason=termination_reason,
            final_result=_final_result_to_dict(final),
            turns=turns,
            started_at_monotonic=started,
            ended_at_monotonic=time.monotonic(),
        )

        self._persist(summary)
        return summary

    # ---------------- private -------------------------------------------

    def _build_prompt(self, turn_idx: int) -> str:
        info = self._server.info()
        if turn_idx == 0:
            return compose_initial_prompt(
                workspace=str(self._server.workspace_dir),
                http_url=self._http_url,
                info=info,
                task_md=self._server.task_md_text(),
                agents_md_excerpt=AGENTS_MD_EXCERPT,
                max_turns=self._max_turns,
            )
        obs: TurnObservation = observe(
            turn_index=turn_idx,
            info=info,
            workspace=self._server.workspace_dir,
        )
        return compose_continuation_prompt(obs, max_turns=self._max_turns)

    def _persist(self, summary: RunSummary) -> None:
        log_dir = self._server.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "harness_runner.json"
        out_path.write_text(json.dumps(summary.to_record(), indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _final_result_to_dict(final: FinalResult) -> dict[str, Any]:
    """``FinalResult`` is a frozen dataclass; expose enough for the
    summary JSON without leaking the ``Path`` (jsonified separately)."""
    return {
        "status": final.status,
        "final_score": final.final_score,
        "held_out_mean_return": final.held_out_mean_return,
        "held_out_std_return": final.held_out_std_return,
        "final_submit_index": final.final_submit_index,
        "error": final.error,
        "run_json_path": str(final.run_json_path),
    }


__all__ = ["AgentLike", "HarnessRunner", "RunSummary", "TurnLogEntry"]
