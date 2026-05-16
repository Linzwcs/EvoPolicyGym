# HLBench

HLBench is a benchmark for evaluating language and coding agents as iterative
heuristic optimizers. An agent receives a constrained workspace, reads task and
feedback files, runs train-only experiments, edits an executable policy, and is
then evaluated by the benchmark on train, validation, and heldout splits.

The core question is not whether a model can write one good policy in a single
response. HLBench measures whether a model can repeatedly improve a maintained
heuristic system under fixed budgets and strict visibility boundaries.

## Current Status

The repository contains the refactored HLBench v1 skeleton:

- generic Gymnasium environment backend;
- scenario contracts and generated task contracts;
- train / validation / heldout seed split support;
- learner workspace generation;
- command-based agent execution;
- epoch loop, scoring, reports, and learning curves;
- smoke scenarios for Classic Control and Box2D.

Current built-in scenarios:

```text
cartpole_balance
mountain_car
pendulum_swingup
acrobot_swingup
lunar_lander
```

The target benchmark suite is documented in
[docs/benchmark/environment_roadmap.md](docs/benchmark/environment_roadmap.md).

## Repository Layout

```text
src/hlbench/
  core/        Scenario, seed, task, policy, and artifact contracts
  envs/        Environment backends and observation/action schemas
  rollout/     Policy rollout engine and CLI
  harness/     Agent execution, epoch loop, scoring, and evaluation
  reports/     Metrics and HTML report generation
  scenarios/   Built-in benchmark scenarios
  workspace/   Learner workspace creation and feedback publishing

docs/          Benchmark, protocol, and architecture documentation
scripts/       Convenience run scripts
tests/         Unit and protocol tests
runs/          Local run artifacts, ignored by git
```

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

During development, run commands with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Common Commands

Validate a scenario:

```bash
PYTHONPATH=src python -m hlbench scenario validate --scenario mountain_car
```

Run a train rollout with a scenario baseline policy:

```bash
PYTHONPATH=src python -m hlbench.rollout.cli \
  --scenario lunar_lander \
  --split train \
  --episodes 2 \
  --output-dir /tmp/hlbench-lunar-smoke
```

Run a no-op one-epoch harness smoke test:

```bash
PYTHONPATH=src python -m hlbench run \
  --scenario mountain_car \
  --model-name local-smoke \
  --agent-preset none \
  --preset smoke \
  --epochs 1
```

Run a Codex-agent experiment with the shared 32-train-episode script:

```bash
SCENARIO=mountain_car EPOCHS=8 TRAIN_EPISODES=32 \
  ./scripts/run_codex_scenario_32.sh
```

## Evaluation Boundary

Learner workspaces expose train feedback and aggregate validation summaries.
Heldout data and heldout scores never enter the workspace. Validation and
heldout evaluations do not produce replay, per-episode, seed, or failure-detail
artifacts.

Run outputs are written to:

```text
runs/<model_name>/<env_name>/<run_id>/
```

The main reported metrics are final heldout score, best heldout score, heldout
AUC over epochs, train episode budget, agent wall time, and failure counts.

## Documentation

Start with [docs/README.md](docs/README.md). The most relevant protocol docs are:

- [Evaluation protocol](docs/benchmark/evaluation_protocol.md)
- [Environment roadmap](docs/benchmark/environment_roadmap.md)
- [Visibility boundary](docs/benchmark/visibility_boundary.md)
- [Scenario validation](docs/code/scenario-validation.md)
- [Task contract](docs/code/task-contract.md)

## Contributing

See [AGENTS.md](AGENTS.md) for repository guidelines, coding style, testing
commands, and pull request expectations.
