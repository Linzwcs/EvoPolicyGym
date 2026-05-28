# hlbench-pro

A benchmark for evaluating how well coding agents iteratively
synthesize and refine control policies under environmental feedback,
with tight rollout budgets.

## What it measures

Given a control task, a tight rollout budget, and a workspace that
persists across iterations, **how well can an agent improve a
submitted policy?** The benchmark scores the final policy on held-out
episodes drawn from a disjoint, hidden seed pool.

This capability sits in a gap between existing benchmarks:

| Benchmark | Closed loop? | Environmental feedback? | Code iteration? |
|---|---|---|---|
| SWE-bench | no | only test pass/fail | one-shot fix |
| MLE-bench | no | validation score | one-shot script |
| AgentBench / GAIA | partial | tool returns | varies |
| HumanEval / GPQA | no | none | none |
| RL benchmarks | n/a (measures algorithms, not agents) | dense rewards | n/a |
| **hlbench-pro** | **yes** | **rich replays** | **yes, budgeted** |

No existing benchmark sits in the bottom row.

## Method neutrality

hlbench-pro does not prescribe how you solve a task. Heuristic
controllers, classical control, search-and-planning, neural networks
trained from scratch, or any combination are permitted. The rollout
budget is intentionally tight; methods that need many trials will
typically underperform methods that extract more information per
episode, but this is an outcome of the budget, not a rule.

See `AGENT.md §6` for details.

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Create a workspace for a task (starts a per-run server)
hlbench init --task halfcheetah --dir ./my_run

# 3. Inspect the run config — budget and limits live on the server
hlbench info                 # calls GET /info under the hood

# 4. Write or edit policy
$EDITOR my_run/system/policy.py

# 5. Submit (agent picks env instance IDs; each consumes 1 episode)
cd my_run && hlbench submit --env-instances 0-7

# 6. Inspect feedback, iterate, pick next IDs strategically
ls feedback/submit_000/
$EDITOR system/policy.py
hlbench submit --env-instances 5,42,100        # cheap probe of 3 specific instances
hlbench submit --env-instances 0-15            # high-confidence eval over 16

# 7. After episode budget is exhausted, see final score
hlbench score
```

A workspace contains exactly one task. The workspace and its
`system/` directory persist across submits within a run. The
agent decides which env instances each submit runs (and thereby
how many episodes); the server's `GET /info` exposes the effective
budget, env instance count, and limits.

## How it works

```
        agent                       harness
        ┌─────┐                     ┌──────┐
        │     │                     │      │
        │ edit│──── system/ ───────▶│ snap │
        │     │                     │      │
        │     │  --env-instances    │ run  │
        │     │ ──────────────────▶│ N ep │
        │     │                     │      │
        │ read│◀──── feedback/ ─────│ write│
        │     │                     │      │
        └─────┘                     └──────┘

                  ↑
       loop until episode budget exhausted

         ▼
   ┌────────────┐
   │ held-out   │   100 episodes, fresh hidden seeds
   │ evaluation │   (agent never sees these)
   └────────────┘
         ▼
     final score
```

The agent only ever sees what it produces (in `system/`) and what the
harness returns from submits (in `feedback/`). The agent commits a
specific number of episodes per submit, drawn from a single total
budget. Held-out evaluation seeds are hidden.

## Scoring

The headline score is the normalized held-out return:

```
score = clip((mean_held_out - random) / (expert - random), 0, 1.2) * 100
```

Auxiliary metrics (AUC over consumed episodes, episodes-to-threshold,
in-loop vs held-out gap, mean episodes per submit) are reported
alongside but do not affect the headline. See `SPEC.md §5`.

## Workspace layout

```
workspace/
├── TASK.md          obs / action / reward / eval spec (delivered by server at start)
├── AGENT.md         protocol rules (shipped with agent harness)
├── system/          agent-writable; policy.py is required
└── feedback/        populated by server directly into shared workspace
```

Run config (budget, limits, env metadata, dynamic state) is served by
the per-run server's `GET /info` endpoint — no workspace file.

## What you write

Minimum: `system/policy.py` defining a `Policy` class. You may add
helper modules, tests, analysis scripts, and memory files anywhere
under `system/`, subject to the size limit served by `GET /info`
(default 50 KB total). See `SPEC.md §2` for the interface.

## What you cannot do

- Load pretrained model weights (any source).
- Access the network during policy execution.
- Read held-out evaluation seeds or results before the run ends.
- Read files outside the workspace.

Full list of constraints in `AGENT.md §3` and `§5`.

## Tasks

v1 ships with 12 control tasks across three tiers (sanity, discrimination,
frontier), spanning classic control, Atari (RAM), and MuJoCo. See
`tasks/README.md` for the list and per-tier budgets.

## Documents

- [`AGENT.md`](./AGENT.md) — what agents may and may not do.
- [`SPEC.md`](./SPEC.md) — workspace, policy interface, submit protocol,
  feedback format, scoring.
- [`tasks/`](./tasks/) — individual task descriptions.

## Status

Pre-release. The v1 task suite and harness are under active development.
The protocol described in `AGENT.md` and `SPEC.md` is considered stable
for v1; tasks added in v1.x will conform to it.
