---
locale: en
page: getting-started
section: start
title: "Getting started"
navTitle: "Getting started"
description: "Install EvoPolicyGym 0.3 and complete the first Evaluation or Program-evolution Run."
lead: "Install the Kernel, evaluate an example Program, and start a bounded coding-agent Run."
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## Requirements

- Python `>=3.12,<3.13`
- [`uv`](https://docs.astral.sh/uv/) `0.11.16`
- A local checkout of the repository
- Trusted Policy and Agent code

> `ProcessExecution` launches Policy and Agent subprocesses with the authority
> of your operating-system user.

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

The Kernel supplies the Evaluation and Program-evolution lifecycle. Benchmark
distributions supply environments, Policy contracts, and scoring.

## Choose a Benchmark

Choose a distribution from the
[Environment collection](../../environments/). The commands below use the
small CartPole distribution:

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
```

## Complete one Evaluation

Evaluate the example distribution's packaged Program over five deterministic
validation Episodes:

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole evaluate \
  --episodes 5 \
  --allow-unsafe-process
```

The command prints one JSON object containing the Benchmark identity,
immutable Program digest, scalar score, and public Feedback. Feedback content
follows the selected Benchmark contract.

`--allow-unsafe-process` acknowledges execution under the current local user.

## Run a Coding Agent (optional)

After authenticating the Codex CLI, let the Agent revise the example Program
within a small development budget:

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole run \
  --model gpt-5.5 \
  --record-to runs/quickstart-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --allow-unsafe-process
```

The Agent chooses the Episode count for each Submission. Its Program workspace
is `runs/quickstart-001/workspace/program/`, committed Feedback appears under
`workspace/feedback/`, and Host records remain in the Run directory.

## What the Run does

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
- [Choose and configure an Environment →](../../environments/)
