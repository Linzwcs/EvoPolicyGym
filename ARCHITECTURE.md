# EvoPolicyGym kernel architecture

This document is authoritative for the current clean-slate Kernel. It describes
the local, non-durable product only. Docker, recovery, and remote execution are
not design inputs for this version.

## Domain language

- A `Program` is one immutable, content-addressed Policy source snapshot.
- An `AgentSkill` is one immutable, content-addressed, pathless directory
  snapshot explicitly selected as a Coding Agent input to a Run.
- Environment parameters are the public, Case-independent values bound to one
  configured Benchmark before Evaluation. Their canonical digest distinguishes
  otherwise identical Benchmark IDs with different task configurations.
- An `Evaluation` runs one Program through a bounded set of Episodes.
- A training Episode pool is one immutable Run-local tuple of trusted
  `EpisodeSpec` and Policy-seed pairs, addressed publicly only by integer
  indices.
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
├── skills.py                   immutable explicit Agent Skill snapshots
├── errors.py                   sanitized public failures
├── authoring/                  external Benchmark authoring and conformance SPI
│
├── agents/                     Coding Agent providers
│   ├── base.py                 Agent contract and shared command helpers
│   └── codex.py                Codex selection and CLI translation
│
├── evaluation/                 complete direct-Evaluation use case
│   ├── __init__.py             EvaluationConfig and evaluate()
│   ├── _plan.py                exact trusted Episode and Policy-seed inputs
│   └── _service.py             Episode rules and narrow runtime contracts
│
├── run/                        complete Program-Evolution use case
│   ├── __init__.py             Run/Validation/Assessment configs and run()
│   ├── progress.py             public Run events, observer, and console reporter
│   ├── _service.py             Run coordination and process-setting assembly
│   ├── _episode_pool.py        fixed training pool and seed derivation
│   ├── _session.py             Submission budget and atomic finish admission
│   ├── _validation.py          post-Agent candidate evaluation and selection
│   ├── _assessment.py          held-out final-Program measurement
│   ├── _directory.py           workspace, events, invocation, and run.json
│   ├── _feedback.py            Feedback and Artifact publication
│   ├── _json.py                retained public-value JSON projection
│   ├── _socket.py              active Agent Session transport
│   ├── _session_cli.py         Agent-facing Session presentation
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
└── cli.py                      Host/operator presentation over the public SDK
```

There is no parallel private shadow package for a public use case.
`evopolicygym.evaluation`, `evopolicygym.run`, and
`evopolicygym.execution` are both their stable public entry points and their
implementation ownership boundaries.

The `evopolicygym` executable is the Host/operator command surface. The
separate `evopolicygym-session` executable is projected into an active Agent
Run and owns only `submit` and `finish`; it is a presentation over
`agent-session/v3`, not an operator workflow or Benchmark registry.

## Environment distributions

Independently installable Benchmark distributions live under the repository's
top-level `environments/` directory. Each child owns its Environment,
deterministic Episode planning, scoring, Feedback, baseline, dependencies, and
tests.

An Environment distribution depends only on the supported public SDK and
`evopolicygym.authoring` SPI. It is not included in the base wheel, the Kernel
does not import it, and sibling distributions do not import one another.
`environments/gymnasium/classic_control/cartpole/`, for example, builds
`evopolicygym-benchmark-cartpole` and imports as `cartpole`.

First-party reusable Coding Agent workflows live independently under the
repository's top-level `skills/` directory. A Benchmark distribution may
document a compatible Skill, but does not package, load, or reference it from
`BenchmarkSpec`.

## Environment configuration boundary

The caller constructs a configured Benchmark using the distribution's typed
API. The distribution validates those values, retains them on the Benchmark,
publishes the exact applied values through
`BenchmarkSpec.environment_parameters`, and uses that bound configuration from
`make_environment(episode)`.

```text
distribution-owned typed parameters
                 │
                 ▼
       configured Benchmark
          │              │
          ▼              ▼
 public BenchmarkSpec   make_environment(EpisodeSpec)
 parameters + digest        uses bound parameters
```

The Kernel never accepts an untyped bag of simulator keyword arguments and
does not pass a second configurable value into `make_environment()`. This
keeps validation and application under one owner and prevents two competing
configuration sources.

Environment parameters are static, public, and visible to the Coding Agent and
the Episode-local Policy through `PolicyContext`. `EpisodeSpec.scenario` and
`environment_seed` remain trusted per-Episode inputs and never cross the Policy
boundary. Run budgets, timeouts, rendering, and execution settings are not
Environment parameters.

The canonical Environment digest is derived from the exact bounded
`PolicyValue` mapping, including carrier types. Evaluation results carry that
digest; Run submission, Validation, and Assessment rules reject results whose
Benchmark ID or Environment digest differs from the Run's selected
`BenchmarkSpec`.

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
- `policy/v2` provides the Case-independent public Environment parameters
  separately from descriptive Benchmark metadata;
- `evaluation/_service.py` declares the Policy-runtime capabilities it
  consumes and never selects an execution setting or Agent provider;
- `run/_session.py` owns budgets, admission, publication ordering, and the
  mapping from submitted public Episode indices to the fixed training pool,
  plus the atomic transfer of an ordered candidate set, without depending on
  an execution setting or provider;
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
- callers explicitly compose zero or more immutable `AgentSkill` snapshots
  into a Run; the Run retains each complete directory read-only at
  `workspace/skills/<name>/`, records its content digest, and never exposes a
  Skill to direct Evaluation or the Policy boundary;
- provider packages translate the task into their own invocation but do not
  author the task or start and supervise the process themselves;
- provider-specific experiment inputs such as the Codex model and reasoning
  effort belong to the provider selection and retained Agent identity, not
  `RunConfig` or `BenchmarkSpec`;
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
  submit(Episode indices) → public Feedback → edit → ... → finish(candidate IDs)
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

Before Agent execution, the Host derives one training-pool seed from
`RunConfig.seed` under `evopolicygym/training-pool/v1`, asks the Benchmark for
exactly `episode_pool_size` `EpisodeSpec` values once, and derives one stable
Policy seed per index under `evopolicygym/training-policy/v1`. The default pool
size equals `episode_budget`, but callers may configure a larger selectable
pool without increasing total authority.

`submit` uses `agent-session/v3` and carries an explicit, non-empty,
strictly-increasing `episode_indices` list. The CLI expands singleton and
half-open range unions such as `"0:2,4:8"` to
`[0, 1, 4, 5, 6, 7]`. Duplicate, overlapping, malformed, out-of-pool, and
over-budget selections are rejected before Program capture. Reusing an index
across Submissions is valid and consumes budget again. The mapping is fixed
for the Run, while Evaluation still creates a fresh Environment, Policy
process, Policy instance, and scratch directory for every selected index.

`finish` also uses `agent-session/v3` and accepts a non-empty
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
`feedback/`, and zero or more explicitly selected read-only Skill directories
under `skills/`. `validation/` and `assessment/` are not created until after
Agent cleanup. Successful phases retain only aggregate scores and
Policy-failure counts in their reports; private Episodes, seeds, cases, traces,
and execution evidence do not cross into Feedback. Submission Feedback uses
`evopolicygym/feedback/v2` and maps every public Episode summary to its
Run-local training index. `run.json` uses
`evopolicygym/run-record/v6`, retains the pool size and derivation protocol,
the public Environment parameters and canonical digest beside the Benchmark
ID, selected index sets, selected Skill names, digests, and snapshot paths,
and references the available aggregate reports.

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
