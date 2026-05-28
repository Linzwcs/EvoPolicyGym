# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is a **docs-only repository at the design stage** — no source code exists yet. The five markdown files below define the protocol, behavior, and a 2-week MVP implementation plan. The plan targets Pendulum-v1 end-to-end as the first working slice.

When asked to "start implementing", follow `docs/architecture.md` Day 1: scaffold `src/hlbench/` per its package layout (§3), then the env registry + Pendulum env (Day 2). Don't go off-plan without flagging it.

## Document map

Each document has a single audience and purpose; don't conflate them.

| File | Audience | Purpose |
|---|---|---|
| `README.md` | First-time reader | Project pitch, gap statement, quick start |
| `AGENT.md` | The benchmark's runtime agent | Rules of the game (sandbox, anti-hack, submit protocol from agent's POV) |
| `SPEC.md` | Harness implementer | Wire-level contract: workspace layout, `/info` endpoint, Policy interface, feedback schemas, scoring, held-out details |
| `docs/output.md` | Analyst / leaderboard tooling | What `runs/<model>/<env>/<exp-id>/` looks like on disk (runtime output layout) |
| `docs/submit-protocol.md` | Implementer + sophisticated agent | Submit lifecycle (7 phases, 11 verdicts), anti-cheating provisions |
| `docs/architecture.md` | Whoever writes the first line of code | MVP implementation plan: package layout, component sketches, 2-week schedule, validation checklist |

When changing one of these docs, **scan the others for cross-references**. The user has been bitten multiple times by orphan references after rename/restructure (e.g., `total_episode_budget` → `episode_budget`, deleting `_run.json`, deleting `method_profile.json`).

## Non-negotiable design invariants

These have been argued through and resettled multiple times. Don't relitigate without explicit user instruction:

1. **Method neutrality.** The benchmark does not prescribe heuristic vs. learned methods. Frame docs around "iterative code refinement under tight rollout budget", never "Heuristic Learning paradigm". `torch`/`jax` are allowed; `transformers`/`huggingface_hub`/`stable_baselines3` are denied (block pretrained-model shortcuts).
2. **Held-out is fully invisible.** Agent never sees: held-out size, held-out seeds, individual held-out returns, `expert_baseline`, `random_baseline`, or `final_score` during the run. Held-out lives in env-side `heldout.json` and the server runs it only after `finalize`.
3. **`env_instance` addressing.** Agents address envs by integer ID `[0, n_env_instances)`. The mapping ID → real seed lives in env-side `train.json` and is server-internal. Never expose real seeds to agents.
4. **`/info` is the source of truth for config.** No `_run.json` file in workspace. The per-run server's `GET /info` serves merged static config + live dynamic state.
5. **Workspace contains exactly 4 things:** `TASK.md`, `AGENT.md`, `system/`, `feedback/`. The first two are static (delivered at run start). `system/` is the only agent-writable directory. `feedback/` is populated directly by the server (shared filesystem) after each submit.
6. **`system/` is a Python package.** Agents organize code freely under `system/`; `policy.py` is the required entry point. `sys.path[0] = workspace/system/` during policy execution. Size limit counts source files only (excludes `__pycache__`, `.pytest_cache`, etc.).
7. **Per-run server, not multi-tenant.** One `Server` instance per (model, env, exp-id). Sandboxes don't multiplex runs.
8. **Shared-host architecture, HTTP control plane.** Server and agent run on the same host, sharing the workspace directory. Agent issues commands (submit, info, finalize) **via HTTP only** (sync, no long-poll, no tar bundle); server writes feedback files directly to the shared `workspace/feedback/`. Agent reads feedback as local files. Remote / distributed deployment is **not** a goal.
9. **Lib-first internals.** Core is `hlbench.core.Server` (Python class); the HTTP layer is a thin FastAPI wrapper. Tests, dev tooling, and orchestration may use the lib directly. Agents cannot — they must go through HTTP.

## Recurring debate points (with the settled answer)

The user explores design space by raising the same questions repeatedly. Each time they've been resolved, the answer is below:

- **"Do we need `_run.json`?"** → No. Deleted. Use `GET /info`.
- **"Do we need `TASK.md` in workspace?"** → Yes (kept). Static prose, delivered by server at run start.
- **"Do we need `AGENT.md` in workspace?"** → Yes (kept). Client-side protocol document.
- **"Do we need `method_profile.json`?"** → No. Implementer-dependent metrics are not comparable; removed.
- **"Do we need `diff_from_prev.json`?"** → No. Agent already knows what they changed; AST diffs are implementer-dependent.
- **"Should `index_width` be in `/info`?"** → No. Derived from `episode_budget`, agents compute themselves.
- **"Should `expert_baseline`/`random_baseline` be exposed?"** → No. Forces agents to optimize without targeting a known threshold.
- **"Should feedback be served via API endpoints or files?"** → Files. Server writes directly to shared `workspace/feedback/`; agent reads as local files. (Earlier we considered tar-bundle delivery for remote deployment — abandoned. Shared host is the only target.)
- **"Sync or async submit?"** → Sync. Local HTTP, no timeout pressure, episodes are seconds-to-minutes.
- **"HTTP or Python lib for agent control?"** → HTTP only for agents. Lib mode exists for tests/dev but is not an agent-facing interface.
- **"Hash verification on seed files?"** → No. `env_version` already binds the seed files; hash is paranoia without a public leaderboard.

## When implementing (per architecture.md)

The MVP plan in `docs/architecture.md` deliberately defers many real spec items:
- No `denied_imports` enforcement (Pendulum + numpy doesn't need it)
- No network blocking (same)
- No `checkpoints/` directory
- No `agent.jsonl` logging
- No container (subprocess sandbox only)

(HTTP wrapper **is** in MVP — it's the agent's only control channel.
Async + long-poll + tar bundle are **permanently** out of scope per
the shared-host commitment, not just deferred.)

These are **real spec items**, just not needed to validate the core loop on Pendulum. When extending past MVP (HalfCheetah → CarRacing → etc.), each addition forces filling in one of these. Don't try to implement the full spec in one go.

## Style notes for doc edits

- Use sharp, focused prose. The user dislikes vague "should consider..." language.
- When proposing options, give 2-4 with trade-offs explicit. Recommend one. Mark `(推荐)`.
- For multi-file refactors, grep for stale references after the main edits.
- The user reads Chinese; technical terms stay in English. Mixed Chinese-English in prose is fine.
- Avoid emoji unless the user uses them first.
