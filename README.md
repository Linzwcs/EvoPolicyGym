# hlbench-pro

A benchmark for evaluating how well coding agents iteratively
synthesize and refine control policies under environmental feedback,
with tight rollout budgets.

## What it measures

Given a control task, a tight rollout budget, and a workspace that
persists across iterations, **how well can an agent improve a
submitted policy?** The benchmark scores the final policy on a
held-out pool of episodes drawn from a separate, hidden seed pool.

This sits in a gap between existing benchmarks:

| Benchmark | Closed loop? | Environmental feedback? | Code iteration? |
|---|---|---|---|
| SWE-bench | no | only test pass/fail | one-shot fix |
| MLE-bench | no | validation score | one-shot script |
| AgentBench / GAIA | partial | tool returns | varies |
| HumanEval / GPQA | no | none | none |
| RL benchmarks | n/a (measures algorithms, not agents) | dense rewards | n/a |
| **hlbench-pro** | **yes** | **rich replays** | **yes, budgeted** |

## Method neutrality

hlbench-pro does not prescribe how to solve a task. Heuristic
controllers, classical control, search-and-planning, networks
trained from scratch, or any combination are permitted. The rollout
budget is intentionally tight; methods that need many trials
typically underperform methods that extract more information per
episode, but that's an outcome of the budget, not a rule.

`torch` and `jax` are available; `transformers`, `huggingface_hub`,
and `stable_baselines3` are blocked to prevent pretrained-model
shortcuts. See [`AGENTS.md`](./AGENTS.md) for the full list.

## 60-second quick start

```bash
# 1. Install (Python 3.12+; uv recommended)
uv venv --python 3.12 .venv
.venv/bin/pip install -e .

# 2. Create a run dir at runs/<model>/<env>/<exp-id>/
.venv/bin/hlbench init --env pendulum --model reference-pd --exp-id demo

# 3. Drop the reference policy into the workspace
RUN_DIR=./runs/reference-pd/pendulum/demo
cp agents/pd_pendulum/policy.py $RUN_DIR/workspace/system/policy.py

# 4. Start the server (foreground, in one terminal)
.venv/bin/hlbench serve --run-dir $RUN_DIR --env pendulum

# 5. From another terminal, submit and finalize
.venv/bin/hlbench info
.venv/bin/hlbench submit --env-instances 0-7
.venv/bin/hlbench finalize
# → run.json at $RUN_DIR/run.json with outcome.final_score
```

For a deeper walkthrough including the lib API, see
[`docs/quickstart.md`](./docs/quickstart.md).

## How it works

```
        agent                       harness
        ┌─────┐                     ┌──────┐
        │     │                     │      │
        │ edit│──── system/ ───────▶│ snap │
        │     │                     │      │
        │     │   POST /submit      │ run  │
        │     │  ──────────────────▶│ N ep │
        │     │                     │      │
        │ read│◀──── feedback/ ─────│ write│
        │     │                     │      │
        └─────┘                     └──────┘

                  ↑
       loop until episode budget exhausted

         ▼
   ┌────────────┐
   │ held-out   │   M episodes, fresh hidden seeds
   │ evaluation │   (agent never sees these)
   └────────────┘
         ▼
     run.json   (final_score + auxiliary metrics)
```

The agent reads only what it produces (`system/`) and what the
harness returns (`feedback/`). It commits a specific count of
episodes per submit, drawn from a single budget. Held-out seeds
are hidden during and after the run.

## Workspace layout

```
workspace/
├── AGENTS.md    protocol rules (delivered by server at run start)
├── system/      agent-writable; system/policy.py is required
└── feedback/    populated by server after every submit
```

Run config (budget, limits, env metadata, dynamic state) is served
by the server's `GET /info` endpoint; the env's human-readable task
description by `GET /task` (text/markdown). No workspace config or
task files. Held-out results live in `run.json` outside the workspace,
written once at finalize.

## What you write

Minimum: `system/policy.py` defining a `Policy` class with `reset`,
`act`, and (optionally) `on_episode_end`. You may add helper modules
anywhere under `system/`. See [`SPEC.md §2`](./SPEC.md) for the
interface and [`agents/pd_pendulum/policy.py`](./agents/pd_pendulum/policy.py)
for a working reference.

## Scoring

```
final_score = clip((mean_held_out - random) / (expert - random), 0, 1.2) * 100
```

Auxiliary metrics (`auc_in_loop`, `episodes_to_50pct`, `held_out_gap`,
…) are reported alongside but don't affect the headline. See
[`SPEC.md §5`](./SPEC.md) for the full list.

`expert_baseline` and `random_baseline` are env-internal; the agent
never sees their numerical values during or after the run, so it
optimizes without targeting a known threshold.

## Document map

| File | Audience | Purpose |
|---|---|---|
| [`README.md`](./README.md) | First-time reader | Project pitch, quick start |
| [`AGENTS.md`](./AGENTS.md) | Benchmark agent | Rules: sandbox, anti-hack, submit protocol |
| [`SPEC.md`](./SPEC.md) | Harness implementer | Wire-level contract: workspace, /info, Policy interface, feedback, scoring |
| [`docs/quickstart.md`](./docs/quickstart.md) | First-time benchmark user | Working walkthrough |
| [`docs/findings.md`](./docs/findings.md) | Spec maintainer | Calibration analysis (Day 14) |
| [`docs/architecture.md`](./docs/architecture.md) | Implementer | Package layout, build sequence, validation checklist |
| [`docs/output.md`](./docs/output.md) | Analyst / leaderboard | What `runs/<...>/` looks like on disk |
| [`docs/submit-protocol.md`](./docs/submit-protocol.md) | Implementer + agent | Submit lifecycle (7 phases, 11 verdicts) |
| [`CHANGELOG.md`](./CHANGELOG.md) | Reviewer | What shipped in this release |

## Status

MVP — Pendulum-v1 only. The harness end-to-end pipeline (init → submit
→ finalize → run.json) is working. Held-out evaluation, normalized
scoring, and all auxiliary metrics ship.

Reference PD on Pendulum-v1: `final_score = 98.3`, held-out mean
return = -168 (random ~ -1200, expert ~ -150). See
[`docs/findings.md`](./docs/findings.md) for the calibration sweep.

Post-MVP roadmap (deferred): denied-import enforcement, network
sandbox, RSS-poll OOM, video / external-obs storage, additional envs
(HalfCheetah, CarRacing, Atari).
