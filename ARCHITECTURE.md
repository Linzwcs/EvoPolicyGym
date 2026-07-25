# EvoPolicyGym kernel architecture

This document is authoritative for the current clean-slate Kernel. It describes
the local, non-durable product only. Docker, recovery, and remote execution are
not design inputs for this version.

## Domain language

- A `Program` is one immutable, content-addressed Policy source snapshot.
- An `Evaluation` runs one Program through a bounded set of Episodes.
- A `Submission` commits one Program and its public Feedback.
- A finished candidate set is an ordered, bounded tuple of published
  Submissions handed from the Agent to the Host in one atomic request.
- `Validation` is a Host-only, post-Agent Evaluation of every finished
  candidate on identical Episodes. It selects the final Submission without
  publishing evidence back to the Agent workspace.
- `Assessment` is a Host-only held-out Evaluation of the already selected
  final Program. It measures the result but never participates in selection.
- `Feedback` has a Kernel-required scalar score, Benchmark-defined public
  content, and optional public Artifact files.
- A `ProgramEvolutionRun` is one bounded outer loop in which a Coding Agent
  edits Programs, submits candidates, reads Feedback, and hands candidates to
  Host-side final selection.
- A `RunEvent` is immutable, Host-side observation delivered only after its
  matching lifecycle event is persisted.
- An `Experiment` is reserved for a future collection of comparable Runs.
- An `AgentRunner` starts and reaps one Coding Agent. It does not own Submission
  accounting, Feedback publication, or Run records.

## Package ownership

```text
evopolicygym/
├── __init__.py                 lazy common workflow exports
├── policy.py                   submitted Policy ABI; stdlib-only leaf
├── program.py                  immutable Program snapshots
├── benchmark.py                caller-facing Benchmark identity
├── results.py                  detached Feedback and result values
├── artifacts.py                bounded public Artifact values
├── errors.py                   sanitized public failures
├── authoring/                  external Benchmark authoring and conformance SPI
│
├── agents/                     Coding Agent providers
│   ├── base.py                 Agent contract and shared command helpers
│   └── codex.py                Codex selection and CLI translation
│
├── evaluation/                 complete direct-Evaluation use case
│   ├── __init__.py             EvaluationConfig and evaluate()
│   └── _service.py             Episode rules and narrow runtime contracts
│
├── run/                        complete Program-Evolution use case
│   ├── __init__.py             Run/Validation/Assessment configs and run()
│   ├── progress.py             public Run events, observer, and console reporter
│   ├── _service.py             Run coordination and process-setting assembly
│   ├── _session.py             Submission budget and atomic finish admission
│   ├── _validation.py          post-Agent candidate evaluation and selection
│   ├── _assessment.py          held-out final-Program measurement
│   ├── _directory.py           workspace, events, invocation, and run.json
│   ├── _feedback.py            Feedback and Artifact publication
│   ├── _json.py                retained public-value JSON projection
│   ├── _socket.py              active Agent Session transport
│   └── _task.py                provider-independent Agent instructions
│
├── execution/                  public execution selections
│   ├── __init__.py             explicit unsafe ProcessExecution acknowledgement
│   └── process/                private process implementation
│       ├── policy/
│       │   ├── runtime.py      Host Policy-process controller
│       │   ├── worker.py       Episode-local guest entry point
│       │   └── stream.py       blocking frame I/O
│       └── agent/
│           └── runner.py       command Agent process lifecycle
│
├── _protocol/
│   ├── _framing.py             shared bounded JSON frame mechanism
│   ├── policy.py               Policy process framing and PolicyValue codec
│   └── session.py              versioned Agent Session framing
└── cli.py                      Agent-facing Session presentation
```

There is no parallel private shadow package for a public use case.
`evopolicygym.evaluation`, `evopolicygym.run`, and
`evopolicygym.execution` are both their stable public entry points and their
implementation ownership boundaries.

## Environment distributions

Independently installable Benchmark distributions live under the repository's
top-level `environments/` directory. Each child owns its Environment,
deterministic Episode planning, scoring, Feedback, baseline, dependencies, and
tests.

An Environment distribution depends only on the supported public SDK and
`evopolicygym.authoring` SPI. It is not included in the base wheel, the Kernel
does not import it, and sibling distributions do not import one another.
`environments/cartpole/`, for example, builds
`evopolicygym-benchmark-cartpole` and imports as `cartpole`.

## Dependency direction

Public values are the shared vocabulary. A public use-case `__init__.py` owns
its configuration and performs lazy selection only when the operation is
called. Its private service owns the workflow.

```text
caller
  │
  ▼
public use case ───────▶ provider integration
  │                            │
  ▼                            ▼
service and rules ─────▶ execution implementation
  │                            │
  └──────────────▶ pure protocol codecs
```

The rules are:

- importing public configurations or `policy.py` does not load a private
  runtime, process implementation, or protocol codec;
- `evaluation/_service.py` declares the Policy-runtime capabilities it
  consumes and never selects an execution setting or Agent provider;
- `run/_session.py` owns budgets, admission, publication ordering, and the
  atomic transfer of an ordered candidate set without depending on an
  execution setting or provider;
- `run/_validation.py` owns deterministic final selection. It starts only
  after the Agent runner has reaped the process tree and the Session gateway
  has closed;
- `run/_assessment.py` evaluates only the selected Submission, uses its own
  seed domain, and cannot replace or rerank candidates;
- `run/` owns Run directories, Feedback publication, and Session transport;
  these responsibilities do not live under process execution;
- `run/progress.py` owns non-authoritative observation and terminal
  presentation; observers never participate in Run state transitions and
  receive no private Case, seed, Policy exchange, or Host path;
- `evaluation/_service.py` reports only sanitized Episode completion through a
  narrow callback and performs no terminal or file I/O;
- `execution/process` owns only generic process mechanisms and never imports a
  Codex, Claude, or other provider integration;
- `agents.base.CodingAgent` is the supported structural integration template:
  the Host supplies an `AgentTask`, and the provider returns a validated
  `AgentInvocation`;
- `run/_task.py` owns workspace, submit, finish, budget, and Benchmark
  instructions, so provider implementations do not duplicate Kernel semantics;
- a Benchmark may publish one bounded `agent_skill` in its `BenchmarkSpec`;
  an opted-in Run retains it read-only at `workspace/skill/SKILL.md`, while
  default Runs, direct Evaluation, and the Policy boundary never receive it;
- provider packages translate the task into their own invocation but do not
  author the task or start and supervise the process themselves;
- a small provider integration remains one cohesive module until its own
  responsibilities justify a package;
- `_protocol` is pure bytes/value transformation and performs no I/O;
- `policy.py` remains a stdlib-only leaf for submitted code.

Narrow `Protocol` contracts are colocated with the service that consumes them.
There is no global `ports.py`, global adapter namespace, or global composition
root.

## Run phase and evidence boundaries

```text
Agent search
  submit → public Feedback → edit → ... → finish(candidate IDs)
                                             │
                                             ▼
                              close Session and reap Agent
                                             │
                                             ▼
                                Host-only Validation
                                             │
                                             ▼
                                      final selection
                                             │
                                             ▼
                              held-out Assessment (optional)
                                             │
                                             ▼
                                        Run commit
```

`finish` uses `agent-session/v2` and accepts a non-empty
`submission_ids` list. The Host rejects malformed, duplicate, unknown, or
over-limit candidates before changing Session state. A successful request
closes Agent authority; it does not select a final Program inside the Session.
Without Validation, exactly one candidate is allowed.

Validation uses a Run-seed-derived domain-separated seed and one identical
`EvaluationConfig` for every candidate. Selection compares primary score in
the Benchmark's declared direction, then Policy-failure count, then finish
argument order. A trusted fault terminates the Run as `validation_failed`
without a partial final selection or automatic retry.

Assessment uses an independent Run-seed-derived domain and evaluates only the
selected Submission. A trusted fault terminates as `assessment_failed` while
retaining that final Program; it never falls back to another candidate and
does not create a partial report.

The Agent-visible `workspace/` contains only editable `program/`, public
`feedback/`, and an optional Benchmark skill. `validation/` and `assessment/`
are not created until after Agent cleanup. Successful phases retain only
aggregate scores and Policy-failure counts in their reports; private Episodes,
seeds, cases, traces, and execution evidence do not cross into Feedback.
`run.json` uses `evopolicygym/run-record/v3` and references the available
aggregate reports.

This is a logical lifecycle and publication boundary, not a security boundary:
`ProcessExecution` remains non-isolated. A Benchmark that requires
cryptographically hidden Cases still needs future remote execution or
whole-Run virtualization.

## Migration status

The parallel `_evaluation`, `_evolution`, and `_execution` shadow packages and
the global `_composition.py` root have been eliminated. Earlier architecture
roles such as `_local`, `_engine`, `_adapters`, `_wire`, `_wiring`, and
`settings` remain prohibited. Reintroducing a compatibility or
version-suffixed namespace for removed behavior is also prohibited.
