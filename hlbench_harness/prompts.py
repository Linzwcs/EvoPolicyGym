"""Prompt composition for the Claude Code automation harness.

Two prompt shapes:

- ``compose_initial_prompt`` — sent once on the first ``claude --print``
  call. Embeds the env's TASK.md, the full ``GET /info`` body, the AGENTS.md
  highlights, the workspace path, the HTTP URL, and the operating
  instructions. After this turn the agent has everything it needs; it
  never has to re-fetch /task or /info to know the rules of the game
  (it MAY re-fetch /info to refresh dynamic state).

- ``compose_continuation_prompt`` — sent on every subsequent
  ``--resume`` call. Terse: just the delta state (remaining budget,
  last verdict) and a nudge. Claude's resumed conversation already
  carries the full task context.

Both are pure functions: state in → string out. No I/O. Tests pin the
exact wording so prompt drift is visible in diffs.
"""

from __future__ import annotations

import json
from typing import Any

from hlbench_harness.state import TurnObservation

# ---------------------------------------------------------------------------
# Operational instructions — same wording in every initial prompt so the
# agent knows the protocol.
# ---------------------------------------------------------------------------

_OPS_INSTRUCTIONS = """\
## How you interact with the harness

You are the *agent* in an hlbench-pro run. Your goal: write a Python
``Policy`` class at ``workspace/system/policy.py`` that scores well on
the held-out evaluation that runs after you call ``POST /finalize``.

You have these tools available:

  - **Edit** / **Write** — modify ``workspace/system/policy.py`` (or
    add helper modules anywhere under ``workspace/system/``)
  - **Read** — inspect ``workspace/feedback/submit_NNN/`` after each
    submit. summary.json has the per-submit aggregate; per-episode
    ``trajectory.jsonl`` has step-by-step data
  - **Bash** — call the HTTP server. Examples:
      curl -s $HLBENCH_URL/info
      curl -sX POST $HLBENCH_URL/submit -H 'Content-Type: application/json' \\
           -d '{"env_instances": [0,1,2,3]}'
      curl -sX POST $HLBENCH_URL/finalize

## The loop you should run, in your head and in code

  1. Examine the task. Sketch a policy strategy.
  2. Write ``workspace/system/policy.py``. The class MUST be named
     ``Policy`` with ``__init__(obs_space, action_space, env_meta)``,
     ``reset(episode_index)``, ``act(obs) -> action`` methods.
  3. POST /submit with a small batch (4–8 env_instances) to probe.
  4. Read ``workspace/feedback/submit_<latest>/summary.json``. Look at
     ``mean_return``, ``std_return``, ``returns`` per episode, plus
     ``timeouts`` / ``errors`` if any.
  5. Refine the policy and resubmit. Use larger batches once you trust
     the design (8–16 for high-confidence comparison).
  6. When you're satisfied, or remaining_budget reaches 0, POST
     ``/finalize``. The held-out score is then computed; you will NOT
     see it (it's hidden by design).

## Rules of the game

  - You cannot see held-out seeds, the held-out pool size, or the
    expert/random baselines used in scoring. Optimize on what you see.
  - Each submit consumes ``len(env_instances)`` episodes from the
    budget, regardless of pass/fail. Budget bounds checked on Phase 1
    rejection are free (no budget consumed).
  - Allowed imports: numpy, scipy (except scipy.optimize.minimize and
    friends), math + stdlib, torch/jax/flax. Forbidden: anything that
    loads pretrained weights, opens network sockets, or spawns
    subprocesses.
  - Address env instances by integer ID in [0, n_env_instances). The
    real seed is server-internal.

## How to know when you're done

  - Budget exhausted (remaining_budget reaches 0), OR
  - You explicitly decide to stop iterating, OR
  - The harness will not call you again after ``is_finalized`` becomes
    true.

Use POST /finalize to lock in the most recent successful submit as the
final policy. After /finalize, no more submits are accepted.
"""

# ---------------------------------------------------------------------------


def compose_initial_prompt(
    *,
    workspace: str,
    http_url: str,
    info: dict[str, Any],
    task_md: str,
    agents_md_excerpt: str,
    max_turns: int,
) -> str:
    """Build the initial prompt for the first ``claude --print`` invocation.

    Args:
        workspace: Absolute path to the run's ``workspace/`` dir. This
            is also the cwd of the spawned ``claude`` process, so
            relative paths like ``system/policy.py`` work.
        http_url: Base URL of the running hlbench server. Exposed to
            the agent as the env var ``HLBENCH_URL`` (set by the
            runner before spawning ``claude``).
        info: The ``GET /info`` body at run start. Embedded verbatim
            so the agent can read budget, env_meta, etc. without an
            HTTP round-trip on turn 0.
        task_md: The env's ``TASK.md`` text (from ``GET /task``).
        agents_md_excerpt: A short excerpt from AGENTS.md emphasizing
            the rules that matter most for the loop. The full AGENTS.md
            is in the workspace if the agent wants more.
        max_turns: How many harness turns the agent gets in total
            (informational; budget is the hard limit).
    """
    info_json = json.dumps(info, indent=2, sort_keys=False)
    return _INITIAL_PROMPT_TEMPLATE.format(
        workspace=workspace,
        http_url=http_url,
        info_json=info_json,
        task_md=task_md.strip(),
        agents_md_excerpt=agents_md_excerpt.strip(),
        max_turns=max_turns,
        ops_instructions=_OPS_INSTRUCTIONS.strip(),
    )


_INITIAL_PROMPT_TEMPLATE = """\
# hlbench-pro: automated evaluation turn

You are running inside the hlbench-pro automated evaluation harness.
The harness has set up a workspace and an HTTP server. Each call into
you is one "turn"; you have up to {max_turns} turns total.

## Run context

  - Working directory: ``{workspace}`` (this is your cwd)
  - HTTP server URL: ``{http_url}`` (also exposed as ``$HLBENCH_URL``)
  - Max turns: {max_turns}

## Task (from GET /task)

{task_md}

## Effective config (from GET /info — already fetched for you)

```json
{info_json}
```

## Rules (excerpt from AGENTS.md)

{agents_md_excerpt}

{ops_instructions}

## What to do this turn

Start the loop. On turn 0: read the task, design a starter policy,
write ``system/policy.py``, POST a small probe submit, read the
feedback. Future turns will resume this same session, so you can refer
back to what you did this turn.

When you reply, end with a one-line status of what you did so the
harness can log it.
"""


# ---------------------------------------------------------------------------


def compose_continuation_prompt(obs: TurnObservation, *, max_turns: int) -> str:
    """Build the prompt for a ``--resume`` turn.

    The resumed Claude session already has the task description and
    rules from the initial prompt. We just tell it the current state
    delta and nudge it to continue.
    """
    last = obs.last_submit
    if last is None:
        last_block = "  - No submits recorded yet (last turn did not call POST /submit)."
    else:
        last_block = _format_last_submit(last)

    if obs.remaining_budget <= 0 and not obs.is_finalized:
        nudge = (
            "Budget exhausted (remaining_budget == 0). You can no longer "
            "submit. Call ``POST /finalize`` now to lock in the most "
            "recent successful submit as the final policy."
        )
    elif obs.is_finalized:
        nudge = (
            "The run is already finalized (is_finalized == true). Nothing "
            "more to do; respond with a one-line summary and the harness "
            "will exit."
        )
    else:
        nudge = (
            "Continue iterating: read the latest feedback, refine "
            "``system/policy.py``, submit again. If you think the policy "
            "has converged you may call ``POST /finalize`` early."
        )

    return _CONTINUATION_PROMPT_TEMPLATE.format(
        turn_index=obs.turn_index,
        max_turns=max_turns,
        remaining_budget=obs.remaining_budget,
        last_submit_block=last_block,
        nudge=nudge,
    )


_CONTINUATION_PROMPT_TEMPLATE = """\
# Turn {turn_index}/{max_turns}

## Current state

  - remaining_budget: {remaining_budget}

{last_submit_block}

## Next step

{nudge}

End your reply with a one-line summary of what you did this turn.
"""


def _format_last_submit(last: dict[str, Any]) -> str:
    """Two-to-four lines summarising the latest submit so the agent
    doesn't have to re-read summary.json just to know the headline."""
    status = last.get("status", "?")
    idx = last.get("submit_index", "?")
    n_ep = last.get("n_episodes", "?")
    mean = last.get("mean_return")
    std = last.get("std_return")
    if mean is None:
        # Failed submit
        return (
            f"  - Last submit: #{idx} status={status} "
            f"(no returns — read submit_{idx:03d}/errors.txt for cause)"
        )
    mean_str = f"{mean:.2f}"
    std_str = "n/a" if std is None else f"{std:.2f}"
    timeouts = last.get("timeouts") or []
    errors = last.get("errors") or []
    extras = ""
    if timeouts or errors:
        extras = f" timeouts={timeouts} errors={errors}"
    return (
        f"  - Last submit: #{idx} status={status} n_episodes={n_ep} "
        f"mean_return={mean_str} std={std_str}{extras}"
    )


# ---------------------------------------------------------------------------
# AGENTS.md excerpt for the initial prompt. Kept inline so prompt tests
# don't depend on AGENTS.md disk contents (which may evolve). The full
# AGENTS.md remains accessible to the agent via the workspace.
# ---------------------------------------------------------------------------

AGENTS_MD_EXCERPT = """\
Hard rules (full text in ``workspace/AGENTS.md``):

  - All information about the environment must flow through the
    submit/feedback loop. The submitted policy must generalize to the
    held-out pool (size + seeds hidden).
  - Forbidden imports: ``transformers``, ``huggingface_hub``,
    ``stable_baselines3``, ``ray``, ``openai``, ``anthropic``,
    ``urllib``, ``requests``, ``socket``, ``httpx``, ``aiohttp``,
    ``subprocess`` (full list in AGENTS.md §3.2).
  - Per-call wall times: ``act()`` 10 ms by default, ``__init__`` 1 s,
    per submit 5 min. ``system/`` capped at 50 KB by default. Effective
    values are in ``GET /info:resource_limits``.
  - No reading outside ``workspace/system/`` from inside ``policy.py``.
    No network from ``policy.py``.
"""
