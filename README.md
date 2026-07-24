# EvoPolicyGym 0.3

This repository contains the clean-slate portable Kernel for evaluating
immutable Policy Programs and letting a Coding Agent improve them through
bounded submissions. Its import name is `evopolicygym`; there are no `v2`,
`next`, `legacy`, or compatibility namespaces.

## Current status

Implemented:

- bounded-carrier Policy ABI, including dense tensors;
- case-independent `PolicyContext` with a separate Policy random seed;
- detached, content-addressed `Program` snapshots;
- structural Benchmark and Environment protocols;
- trusted Episode specifications and records;
- sanitized Feedback, Artifact, Evaluation, Submission, and Run values;
- immutable Evaluation and Run configuration;
- direct evaluation with one fresh Policy subprocess per Episode;
- typed Policy failure versus trusted Environment failure behavior;
- explicit acknowledgement of unsafe local-process execution;
- an in-memory outer-Agent Session with bounded submission authority;
- Agent-facing `submit` and `finish` CLI commands over a private local socket;
- retained Run directories with `workspace/{program,feedback}`;
- Host-owned initial and per-Submission Program snapshots;
- a terminal `run-record/v1` manifest;
- distinct Agent exit classifications for natural exit, timeout, start
  failure, and Host stop after Session terminal state;
- atomic public Feedback and Artifact materialization;
- physically separate Host-side Agent stdout, stderr, and event logs;
- a first-party Codex configuration and executable `run()` workflow;
- an ephemeral, machine-readable Codex CLI invocation contract;
- deterministic Benchmark fixture replay.

Not implemented or exported yet:

- Docker or virtual-machine execution settings;
- persistence, recovery, catalogs, releases, and formal operators.

Remaining executable entry points will be exported only after their complete
semantics exist. The package contains no functions that merely raise
`NotImplementedError`.

## Stable interface roles

| Namespace | Audience |
| --- | --- |
| `evopolicygym` | common caller-facing values |
| `evopolicygym.policy` | submitted Policy ABI |
| `evopolicygym.program` | immutable Program snapshots |
| `evopolicygym.benchmark` | caller-facing Benchmark identity and metadata |
| `evopolicygym.authoring` | complete external Benchmark authoring and conformance SPI |
| `evopolicygym.artifacts` | bounded public Artifact values |
| `evopolicygym.results` | public Feedback and result values |
| `evopolicygym.evaluation` | direct Program evaluation |
| `evopolicygym.execution` | explicit unsafe-process acknowledgement |
| `evopolicygym.run` | executable Coding Agent runs and configuration |
| `evopolicygym.agents` | CodingAgent integration template and providers |
| `evopolicygym.agents.codex` | Codex provider configuration |
| `evopolicygym.errors` | sanitized public failures |

Evaluation and Program-Evolution internals, execution-setting implementations,
storage, subprocess, and protocol details are not public extension surfaces.

## Policy Program

A Program directory contains a fixed `policy.py:make_policy` entry point:

```python
from evopolicygym.policy import PolicyContext, PolicyValue


def make_policy(context: PolicyContext):
    return MyPolicy(context)


class MyPolicy:
    def __init__(self, context: PolicyContext):
        self._seed = context.policy_seed

    def act(self, observation: PolicyValue) -> PolicyValue:
        return 0
```

Each Episode receives a fresh Policy process, instance, and scratch directory.
State may persist between `act()` calls in that Episode. There is no
`learn()`, `reset()`, `update()`, Submission, or Feedback method.

Capture a caller-owned directory before evaluation:

```python
from evopolicygym import Program

program = Program.from_directory("policy/")
print(program.digest)
```

The snapshot retains no Host source path. Later source mutations cannot change
its files or digest.

## Direct evaluation

Local subprocess execution is intentionally explicit:

```python
from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution

program = Program.from_directory("policy/")

result = evaluate(
    program,
    benchmark,
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(
        split="validation",
        episodes=100,
        seed=42,
    ),
)
```

`ProcessExecution` is not a sandbox. Submitted Policy code has the authority of
the current operating-system user.

For every Episode the evaluator:

1. obtains a trusted `EpisodeSpec`;
2. opens and resets a fresh Environment;
3. derives a separate Policy seed;
4. materializes the immutable Program into fresh scratch;
5. starts a fresh Policy process and constructs `make_policy(context)`;
6. preserves that instance across the Episode's `act()` calls;
7. records strict, unmodified Actions and trusted Steps;
8. closes and reaps the Policy process and Environment on every exit path.

Policy exceptions, timeouts, malformed outputs, and invalid Actions produce
sanitized Policy failures. Environment, Benchmark, process-control, and cleanup
faults abort evaluation and never become Policy score penalties.

## Coding Agent integrations

`run()` accepts any structural `CodingAgent`. The Kernel owns the complete
`AgentTask`, including workspace rules, budget, submit/finish commands, and the
public Benchmark specification. A provider only translates that task into its
validated command invocation:

```python
from dataclasses import dataclass

from evopolicygym.agents import (
    AgentInvocation,
    AgentTask,
    resolve_executable,
)


@dataclass(frozen=True)
class ExampleAgent:
    executable: str

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        executable = resolve_executable(self.executable)
        return AgentInvocation(
            command=(executable, task.instructions),
            recorded_command=(executable, "@agent/instructions.md"),
            identity={"provider": "example"},
            instructions=task.instructions,
        )
```

The returned invocation must retain the Host task unchanged. Process startup,
workspace selection, logging, timeout, and process-tree cleanup remain
Kernel-owned. Codex is the first implementation of this same interface.

## Codex development Run

The caller supplies an immutable initial Program, a Benchmark, and a new Run
directory:

```python
from evopolicygym import Program, RunConfig, run
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution

result = run(
    Program.from_directory("policy/"),
    benchmark,
    agent=Codex(model="gpt-5.5"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/run-001",
    config=RunConfig(
        max_submissions=20,
        episode_budget=1_000,
        max_episodes_per_submission=100,
    ),
)
```

Codex runs from the fixed `workspace/` child. Its only editable and
submittable Policy Program is `workspace/program/`; authorized evaluation
output is visible beside it under `workspace/feedback/`:

```text
<run directory>/
├── run.json                        # written last on normal termination
├── events.jsonl                    # Host lifecycle events
├── workspace/
│   ├── program/                    # Agent-writable Program source
│   └── feedback/                   # Benchmark-authorized Agent view
│       ├── latest.json
│       └── submissions/<id>/
│           ├── feedback.json
│           └── artifacts/...
├── initial/
│   └── program/                    # retained initial snapshot
├── submissions/
│   └── <id>/
│       ├── program/                # retained submitted snapshot
│       ├── feedback.json
│       └── artifacts/...
├── agent/
│   ├── instructions.md             # exact retained Codex task
│   ├── invocation.json
│   ├── stdout.log                  # Codex JSONL event stream
│   └── stderr.log
└── control/                        # present only while the Run is active
    └── session.sock
```

The Host injects `EVOPOLICYGYM_WORKSPACE` and
`EVOPOLICYGYM_SESSION_SOCKET`. From an active Session, the Agent uses:

```console
evopolicygym submit program --episodes 10
evopolicygym finish submission-000002
```

`feedback.json` has one uniform Kernel-generated envelope: submission
accounting, the Benchmark's scalar score, and sanitized per-Episode `status`,
`reward`, `steps`, and failure code. Its `content` field is entirely defined by
the Benchmark and may be any `PolicyValue`; the Kernel does not prescribe or
interpret its keys. JSON-native mappings, lists, and scalars are the simplest
choice, while Host paths and custom Python objects are rejected. The Benchmark
may also add public Artifact files. Artifact content is likewise open-ended and
has no declared content schema; the Agent inspects each file's name, media
type, and contents. This lets a Benchmark publish domain-specific
explanations, traces, diagnostics, images, or reports without teaching the
Kernel their semantics.

The Kernel still owns safe publication constraints. One Artifact is limited to
16 MiB; one Feedback may reference at most 64 files and 64 MiB total. Artifact
names are safe relative POSIX paths, and every published file is committed with
its size and SHA-256 digest. Oversized or invalid Benchmark output fails before
publication instead of producing a partial bundle.

Private process protocols use bounded, length-prefixed strict JSON-object
frames; malformed UTF-8, non-object payloads, and nonstandard numeric constants
such as `NaN` are rejected. CLI and Host frames are identified as
`agent-session/v1`; this transport is private to matching library versions.
Published Feedback uses the internal `evopolicygym/feedback/v1` envelope
version; this does not constrain Benchmark-defined `content` or Artifact
contents.

`submit` accepts only the fixed workspace `program/` directory. It never sends
a Host path through the Session protocol. Its stdout is a compact receipt; the
complete authorized Feedback, sanitized Episode summaries, and Artifact files
are read from `feedback/`.

Submission accounting has one explicit commit point:

1. freeze the current `program/` tree into an immutable `Program`;
2. reserve and deduct the requested Episode budget;
3. evaluate the snapshot;
4. atomically retain the Program, Feedback, and Artifacts under the Host
   `submissions/` record;
5. independently materialize and atomically publish the Agent-facing
   `workspace/feedback/` copy;
6. advance `latest.json` and admit that submission ID into the in-memory set
   accepted by `finish`.

Invalid Program capture does not start evaluation and does not consume Episode
budget. Once evaluation starts, its reserved budget is not refunded: Policy
failure is a committed scored result, while trusted Environment or
infrastructure failure closes the Run as `evaluation_failed`. `finish` can
select only a fully published submission, and the returned final `Program`
must equal that submission's retained immutable `Program`. Every
`SubmissionResult` also carries its detached Program rather than only a digest.

The workspace remains in the Run bundle after termination, including any
unsubmitted final edits. It is observational and never determines the final
Program: `run.json.final_submission_id` selects the authoritative retained
Submission. Agent-facing Feedback and Host records are independent files and
never hard links.

Agent stdout, stderr, retained instructions, invocation metadata, and
structured execution events are siblings of `workspace/`; the Agent does not
receive their paths as part of its workspace contract. `invocation.json`
records environment variable names but never their credential values.

The caller uses Codex's
[non-interactive `exec` mode](https://learn.chatgpt.com/docs/non-interactive-mode)
with ephemeral session storage and JSONL output. Because the current Session
socket is Host-owned outside `workspace/`, this initial `ProcessExecution`
integration launches Codex with `danger-full-access` and approvals disabled.
It is explicitly non-isolated: an unsafe local Agent can traverse to sibling
Host paths and has the authority of the current operating-system user. Use it
only with trusted Benchmarks and Programs. A future whole-Run container will
preserve the same workspace layout while reducing Host authority.

Session state is intentionally volatile: there is no SQLite store, crash
recovery, or resume protocol in this version. `run.json` is a terminal export
written last; a partial directory without it is diagnostic evidence, not
resumable state.

## Benchmark

External packages implement `Benchmark` and `Environment` structurally:

```python
from collections.abc import Sequence

from evopolicygym.authoring import (
    BenchmarkSpec,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
    Step,
)


class CounterEnvironment:
    def reset(self):
        return 0

    def step(self, action):
        return Step(
            observation=1,
            reward=1.0,
            terminated=True,
            metrics={"private_success": True},
        )

    def close(self):
        pass


class CounterBenchmark:
    @property
    def spec(self):
        return BenchmarkSpec(
            id="example/counter-v1",
            description="A deterministic counter.",
            observation_space={"type": "integer"},
            action_space={"enum": [0, 1]},
            metadata={},
            max_episode_steps=10,
            primary_metric="reward",
            score_direction="maximize",
        )

    def episodes(self, split, *, seed, count):
        return tuple(
            EpisodeSpec(environment_seed=seed + index)
            for index in range(count)
        )

    def make_environment(self, episode):
        return CounterEnvironment()

    def feedback(self, episodes: Sequence[EpisodeRecord]):
        return Feedback(
            score=sum(episode.total_reward for episode in episodes),
            content={
                "message": "Evaluation complete.",
                "episodes": len(episodes),
            },
        )
```

`EpisodeSpec.scenario`, its Environment seed, the Policy seed, and per-Step
metrics stay on the trusted Benchmark side. Only the explicit `Feedback`
content and its public Artifacts may reach a caller or Coding Agent.

Simple Benchmarks need only Environment seeds. A structured scenario is
available for dataset examples, maps, horizons, or task configurations that
are semantically distinct from randomness.

## Conformance

```python
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    check_benchmark,
)

report = check_benchmark(
    benchmark,
    fixtures=(
        BenchmarkFixture(
            episode=EpisodeSpec(environment_seed=0),
            actions=(1,),
        ),
    ),
)
report.raise_for_errors()
```

The initial checker verifies structural compatibility, cleanup, bounded value
carriers, and deterministic replay. Additional fault-injection and publication
checks remain additive work.

## Development checks

Run the active Kernel from the repository root:

```console
uv sync --extra dev
uv run python -m unittest discover -s tests
uv run ruff check src tests
uv run mypy
uv run evopolicygym --version
```

Independently installable Benchmark distributions live under `environments/`.
The active CartPole Benchmark is at `environments/cartpole/`. Superseded
implementations, tests, catalogs, and experimental virtualization products are
intentionally excluded from the active repository and dependency graph.
