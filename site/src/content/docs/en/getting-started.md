---
locale: en
page: getting-started
section: start
title: "Getting started"
navTitle: "Getting started"
description: "Install EvoPolicyGym 0.3 and evaluate the CartPole reference Benchmark."
lead: "Install the portable Kernel, run an independently distributed Benchmark, and inspect committed Feedback."
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

## Install CartPole

The current reference distribution is located under `environments/cartpole`:

```console
uv sync --project environments/cartpole --extra dev
```

It installs as `evopolicygym-benchmark-cartpole` and imports as
`evopolicygym_cartpole`.

## Evaluate the baseline

Evaluate the packaged baseline over five deterministic validation Episodes:

```console
uv run --project environments/cartpole \
  evopolicygym-cartpole evaluate \
  --episodes 5 \
  --allow-unsafe-process
```

The command prints one JSON object containing the Benchmark ID, immutable
Program digest, scalar score, and Benchmark-defined Feedback content.

The acknowledgement flag is required because local execution is unisolated.
It does not add containment or change the execution profile.

## Run a Coding Agent

After authenticating the Codex CLI, start a small development Run:

```console
uv run --project environments/cartpole \
  evopolicygym-cartpole run \
  --model gpt-5.5 \
  --record-to runs/cartpole-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --allow-unsafe-process
```

The Agent decides each Submission's Episode count by default. Add
`--max-episodes-per-submission N` only when you want an extra cap.

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
