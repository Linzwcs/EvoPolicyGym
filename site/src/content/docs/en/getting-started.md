---
locale: en
page: getting-started
section: start
title: "Getting started"
navTitle: "Getting started"
description: "Install EvoPolicyGym 0.3 and run a current control or game Benchmark."
lead: "Install the portable Kernel, choose an independently distributed Benchmark, and inspect committed Feedback or semantic replay."
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## Requirements

- Python `>=3.12,<3.13`
- [`uv`](https://docs.astral.sh/uv/) `0.11.16`
- A local checkout of the repository
- Only trusted Policy and Agent code

> **Safety boundary.** The current `ProcessExecution` setting launches local
> subprocesses with the authority of your operating-system user. It is not a
> sandbox.

## Install the Kernel

```console
git clone https://github.com/Linzwcs/EvoPolicyGym
cd EvoPolicyGym
uv sync
uv run evopolicygym --version
```

Expected version output:

```text
evopolicygym 0.3.0
```

The base package contains the portable evaluation and Program-evolution
Kernel. Environment implementations are independently installable Benchmark
distributions.

## Install a Benchmark

CartPole is the compact reference distribution, Acrobot adds sparse-reward
swing-up control, the two Mountain Car distributions contrast discrete and
continuous control, Pendulum completes the Classic Control set, and Balatro is
the long-horizon game distribution:

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv sync --project environments/gymnasium/classic_control/acrobot --extra dev
uv sync --project environments/gymnasium/classic_control/mountain_car --extra dev
uv sync --project environments/gymnasium/classic_control/mountain_car_continuous --extra dev
uv sync --project environments/gymnasium/classic_control/pendulum --extra dev
uv sync --project environments/jackdaw/balatro --extra dev
```

They install as `evopolicygym-benchmark-cartpole`,
`evopolicygym-benchmark-acrobot`, `evopolicygym-benchmark-mountain-car`,
`evopolicygym-benchmark-mountain-car-continuous`,
`evopolicygym-benchmark-pendulum`, and
`evopolicygym-benchmark-balatro`. Their public import packages are `cartpole`,
`acrobot`, `mountain_car`, `mountain_car_continuous`, `pendulum`, and
`balatro`, respectively.

## Evaluate the baseline

Evaluate the packaged baseline over five deterministic validation Episodes:

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole evaluate \
  --episodes 5 \
  --allow-unsafe-process
```

The command prints one JSON object containing the Benchmark ID, immutable
Program digest, scalar score, and Benchmark-defined Feedback content.

The acknowledgement flag is required because local execution is unisolated.
It does not add containment or change the execution profile.

Acrobot and Mountain Car also use mean Episode return, but their normal rewards
are non-positive. A Policy failure therefore contributes the task's complete
Episode floor—`-500` for Acrobot and `-200` for Mountain Car—rather than zero.
Both distributions translate Gymnasium arrays into named semantic dictionaries
before an observation crosses the Policy boundary.

Continuous Mountain Car instead uses a finite floating-point Action from
`-1.0` to `1.0`. Its zero-force baseline returns `0` without success, while a
velocity-direction strategy earns about `89`. Policy failure is scored at
`-100`, below the theoretical minimum complete Episode return.

Pendulum always runs for 200 steps and has no success termination. Its reward
is a negative angle, angular-velocity, and torque cost with a maximum of zero.
Policy failure is scored at `-3300`, below the theoretical minimum complete
Episode return.

## Inspect Balatro

The Balatro Benchmark evaluates one complete Red Deck, White Stake run per
Episode. Its score is a 1000-point win bonus plus one point for every Blind
cleared. Policy decisions cover hands, discards, Blind selection, shops,
Jokers, consumables, packs, and antes.

The public `replay.jsonl` artifact preserves the complete semantic observation
that the Policy received at every retained step. The site player renders a
readable subset without changing the underlying artifact.

- [Read the Balatro Benchmark contract →](../../environments/#balatro)
- [Open the baseline game replay →](../../environments/balatro/replay/)

## Run a Coding Agent

After authenticating the Codex CLI, start a small development Run:

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole run \
  --model gpt-5.5 \
  --record-to runs/cartpole-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --allow-unsafe-process
```

The Agent decides each Submission's Episode count by default. Add
`--max-episodes-per-submission N` only when you want an extra cap.

Balatro also publishes an optional Policy-optimization skill. It is disabled
by default; pass `--benchmark-skill` to `scripts/run_balatro_codex.py` when the
Run should expose it read-only as `workspace/skill/SKILL.md`.

The Agent edits only `runs/cartpole-001/workspace/program/`. Committed public
Feedback is materialized under the adjacent `workspace/feedback/` directory.
Host-side Programs, artifacts, events, and Agent logs are retained separately.

## What happened

1. The initial Policy directory became an immutable, content-addressed
   `Program`.
2. The Coding Agent received a fixed workspace, Benchmark specification, and
   finite submission authority.
3. Every requested Evaluation planned deterministic Episodes.
4. Every Episode created a fresh Environment and fresh Policy process.
5. A completed Submission atomically published its Program, Feedback, Episode
   summaries, and optional artifacts.
6. The Agent selected one fully published Submission as the final Program.

## Next steps

- [Read the core concepts →](../concepts/)
- [Read the Policy ABI →](../policy/)
- [Understand Evaluation and Runs →](../evaluation/)
- [Inspect the Environment collection →](../../environments/)
