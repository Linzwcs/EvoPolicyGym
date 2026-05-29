# docs/intro.md — What HLBench-Pro Measures (and Why)

This document is the project's conceptual introduction: **what capability we
are trying to measure, why no existing benchmark measures it cleanly,
and how the protocol's three design pillars together produce a valid
measurement**. Read this before `SPEC.md`, `AGENTS.md`, or `docs/envs.md` —
those documents describe HOW the system works; this one describes WHY
it exists.

---

## TL;DR

HLBench-pro measures one specific LLM capability: **policy evolution
under environmental feedback with bounded resources**.

The question is: *given a control task, a tight episode budget, and rich
step-level rollout feedback, how well does a given LLM iteratively refine
a Python policy that must generalize to held-out instances?*

That capability is unmeasured by current benchmarks. SWE-bench is one-shot;
MLE-bench is one-shot model-building; HumanEval is one-shot function
completion; Voyager is closed-loop but in a single environment with no
held-out test; Eureka is closed-loop but for reward design, not policy.
The intersection — **multi-task + closed-loop + held-out + policy-code +
budget discipline** — is empty.

Three design pillars together fill the gap:

1. **Capability framing.** We define policy evolution as a distinct LLM
   capability axis and design the benchmark around measuring it.
2. **Budget protocol.** A single resource (episode budget),
   agent-determined allocation, commit-after-snapshot semantics, and
   held-out invisibility — making "iteration costs something" a
   first-class testable constraint.
3. **Anti-cheat package.** Held-out invisibility (size / seeds /
   baselines / results), seed indirection, denied imports enforced at
   `sys.meta_path`, and a formal 10-verdict state machine.

The three pillars are not orthogonal features — each is a *necessary
condition* for the measurement to be valid. Remove budget discipline and
iteration becomes free; remove anti-cheat and "evolution gains" inflate
from leakage; remove the capability framing and the rigor becomes
ungrounded engineering.

---

## §1. The capability axis

### 1.1 Policy evolution is a missing capability axis

LLM evaluation has matured into a set of recognized capability axes,
each with at least one canonical benchmark:

| Capability                       | Canonical benchmark      | Year | Cited  |
|----------------------------------|--------------------------|------|--------|
| Mathematical reasoning           | MATH                     | 2021 | 5000+  |
| Function-level code synthesis    | HumanEval                | 2021 | 3000+  |
| Abstract reasoning               | ARC                      | 2019 | 2000+  |
| Real-world code repair           | SWE-bench                | 2023 | 1500+  |
| Tool use                         | AgentBench               | 2023 | 800+   |
| Long-context retrieval           | Needle-in-haystack       | 2023 | 500+   |
| Vision-language understanding    | MMMU                     | 2024 | 1000+  |
| **Closed-loop policy evolution** | —                        | —    | —      |

The last row is what HLBench-pro fills. By "closed-loop policy evolution"
we mean: given an environment, episode-level feedback, and a tight
rollout budget, can the LLM iteratively write Python code (a `Policy`
class) that performs well on held-out instances?

### 1.2 How it differs from neighbors

| Capability                  | Iteration?  | Code as artifact? | Held-out test?  | Multi-task?   |
|-----------------------------|-------------|-------------------|-----------------|---------------|
| One-shot code synthesis     | No          | Yes               | No              | Yes           |
| In-context learning         | Per-call    | No                | Sometimes       | Yes           |
| Tool use                    | Within task | No (API calls)    | No              | Yes           |
| RL training                 | Yes         | No (weights)      | Sometimes       | Sometimes     |
| Reward design (Eureka-like) | Yes         | Yes (reward)      | No              | Sometimes     |
| **Policy evolution (ours)** | **Yes**     | **Yes (policy)**  | **Yes (hidden)** | **Yes**       |

The distinguishing property: the LLM's product is **executable control
code** that must work on **previously unseen instances**, arrived at
through **iterative observation of execution feedback** under a
**bounded budget**.

### 1.3 Why it matters

This capability surfaces in real deployments:

- **Autonomous coding assistants** (Claude Code, Cursor, Devin) iteratively
  refine code under test feedback. This benchmark measures the iteration
  discipline directly.
- **LLM-driven research automation** (FunSearch, Eureka, OPRO) requires
  the model to propose, evaluate, and revise candidates under bounded
  compute.
- **Closed-loop post-training pipelines** use LLMs to generate task-
  specific code (reward functions, policies, training recipes) and iterate
  based on observed training results.
- **Model selection for agentic deployments** currently lacks a
  comparison metric on closed-loop iteration capability. Labs need it;
  none exists.

### 1.4 Why we expect inter-model differences

Preliminary intuition (to be confirmed empirically):

- Some models are strong at **one-shot policy quality**.
- Some are strong at **incorporating feedback into revisions**.
- Some **regress** when patching one failure mode breaks another (the
  classic iterative-refinement failure pattern).
- Some **plateau in local optima** after the first few submits.

These four sub-dimensions are likely *uncorrelated across frontier
models* — meaning a benchmark that isolates "iteration capability" can
produce model rankings that differ from those of any existing
benchmark, and that differ across the four sub-dimensions. This is the
empirical bet.

---

## §2. The three pillars

### Pillar 1 — Capability framing

The benchmark is positioned around *what is being measured* first;
environment selection follows from that.

The 16 environments in `docs/envs.md` are chosen not because they are
interesting RL problems in isolation, but because they collectively
probe the capability axis at multiple combinations of:

- Observation modality (10 visual / 6 state)
- Action space (continuous / discrete)
- Reward density (sparse / dense)
- Distribution structure (procedural / fixed instance pools)

Each environment must satisfy two design pressures:

1. **Policy synthesis is the bottleneck.** Running a textbook algorithm
   verbatim does not max the score; iteration on observed failures must
   measurably help.
2. **Held-out generalization matters.** Train and held-out instances
   differ in ways that defeat in-loop memorization.

An environment that fails either pressure after empirical calibration
is cut from the suite, not patched.

### Pillar 2 — Budget protocol

The benchmark uses **a single resource** — episode budget — to govern
all iteration cost. Default: 256 episodes per run.

Design choices, with rationale:

| Choice                                          | Why                                                                                                       |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Single resource (episode count)**             | Eliminates metric gaming via cheap-step exploitation; makes cross-model comparison direct                 |
| **Unit is episode, not step or wall time**      | Step is too fine-grained (hides strategy); wall time is gameable via parallel jobs                        |
| **Agent-determined per-submit allocation**      | Cheap probe (1–3) vs high-confidence batch (16–32) — the choice itself is a tested capability             |
| **Commit-after-snapshot semantics**             | Once `system/` is snapshotted, the full `N` is charged regardless of outcome — no speculative submits     |
| **Free retry only on malformed requests**       | `budget_invalid` / `invalid_env_instance` are agent *errors*, not agent *decisions* — they cost nothing   |
| **Held-out budget invisible**                   | Agent has no knowledge of held-out pool size; cannot reserve budget; cannot strategize around it          |

Each choice is traceable to a measurement requirement. Together they
make "iteration costs something" a first-class testable constraint —
without them, evolution gain is hard to measure cleanly because
iteration is effectively free.

Compare with adjacent budget approaches:

| Benchmark      | Resource              | Granularity | Agent decides allocation? | Held-out invisible? |
|----------------|-----------------------|-------------|---------------------------|---------------------|
| Atari 100k     | env steps             | step        | No                        | No                  |
| Procgen 200M   | env steps             | step        | No                        | Partial             |
| MLE-bench      | wall time             | 24 h        | Yes (loose)               | No                  |
| Voyager        | none (until converge) | —           | n/a                       | n/a                 |
| Eureka         | function evals        | call        | No                        | No                  |
| SWE-bench      | one-shot              | —           | n/a                       | n/a                 |
| AgentBench     | per-task action limit | action      | No                        | No                  |
| **HLBench-pro**| **episode budget**    | **episode** | **Yes**                   | **Yes**             |

### Pillar 3 — Anti-cheat package

A closed-loop benchmark is meaningless if the agent can leak its way
to the answer. We enforce:

| Anti-cheat measure          | Mechanism                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------|
| **Held-out invisibility**   | Held-out size / seeds / returns / expert_baseline / random_baseline never exposed      |
| **Seed indirection**        | Agent addresses envs by integer ID; mapping ID → real seed is server-internal           |
| **Denied imports**          | Enforced at `sys.meta_path` in the policy child process before `system/` joins the path|
| **Workspace isolation**     | Agent reads only `workspace/system/` and `workspace/feedback/`; no run-dir access      |
| **Snapshot determinism**    | Every submit snapshotted to `checkpoints/submit_NNN/` — full audit, no mid-flight swap |
| **Formal verdict enum**     | 10 verdicts named; every failure mode reachable from exactly one phase                 |

Each measure addresses a specific leakage attack:

- *Seed memorization* → defeated by held-out invisibility + seed indirection
- *Pretrained shortcuts* → defeated by denied imports
- *Baseline reverse-engineering* → defeated by held-out invisibility
- *Checkpoint snooping* → defeated by workspace isolation
- *Mid-flight gaming* → defeated by snapshot-commit semantics

The **completeness** of the package is itself a contribution. Most
LLM-agent benchmarks address one or two of these but not all; the
gaps inflate reported capability.

---

## §3. Why the pillars are coherent

The three pillars are not three independent design decisions stacked
into a feature list. **Each is necessary for the others to be
meaningful**:

```
            Want to measure policy evolution capability
                              │
                ┌─────────────┴─────────────┐
                │                           │
        Iteration must cost          Generalization must
        something (otherwise         be the target (otherwise
        evolution is free)           in-loop overfit dominates)
                │                           │
        ┌───────┴────────┐                  │
        │                │                  │
   Episode-budget   Snapshot-commit    Held-out invisible
   discipline       semantics          + seed indirection
   (single unit,    (no speculation)   + denied imports
   agent decides)
        │                │                  │
        └───────┬────────┘                  │
                │                           │
                ▼                           ▼
         Budget protocol          Anti-cheat package
                │                           │
                └────────────┬──────────────┘
                             │
                             ▼
              Valid measurement of capability axis
```

Remove any component and the measurement loses meaning:

- Without budget discipline → iteration is free → "evolution gain" is
  just brute force, not a capability signal.
- Without snapshot-commit → agent can speculate (abort bad submits) →
  reported scores measure attempts, not committed decisions.
- Without held-out invisibility → agent overfits to known evaluation →
  scores inflate without measuring generalization.
- Without seed indirection → agent memorizes specific in-loop instances →
  in-loop performance ≠ held-out performance.
- Without denied imports → agent loads a pretrained model → the
  capability measurement collapses into "is HuggingFace available".

This **design coherence** is what elevates the protocol beyond a
feature list to a measurement framework. It mirrors the structure of
the most-cited capability benchmarks (HumanEval's `pass@k`, MATH's
problem format, SWE-bench's containerized verification): every
mechanism serves the same measurement goal.

---

## §4. What HLBench-Pro is not

To prevent miscalibrated expectations:

- **Not a general-purpose RL benchmark.** Policy quality is the test
  material, not the object of study. The object of study is LLM
  capability.
- **Not a code-completion benchmark.** Iteration is the test; one-shot
  scores are not the headline.
- **Not a tool-use benchmark.** The agent's output is a complete
  policy, not a sequence of API calls.
- **Not a reasoning benchmark.** Output is executable code, not
  chain-of-thought.
- **Not a SWE-bench competitor.** Different capability axis. We do
  not claim SWE-bench's task-source scale.
- **Not an MLE-bench competitor.** MLE-bench is one-shot model-building
  on Kaggle competitions; HLBench is closed-loop policy authoring on
  RL tasks. Adjacent, not overlapping.

---

## §5. Reading order

| If you want to                         | Read                                                |
|----------------------------------------|-----------------------------------------------------|
| Run the benchmark                      | `README.md` → `docs/quickstart.md`                  |
| See the environment list               | `docs/envs.md`                                      |
| Write an agent                         | `AGENTS.md`                                         |
| Implement the harness                  | `SPEC.md` → `docs/submit-protocol.md` → `docs/architecture.md` |
| Understand on-disk output              | `docs/output.md`                                    |
| Drive Claude Code end-to-end           | `docs/dogfood.md`                                   |
| Audit what shipped                     | `CHANGELOG.md`                                      |
| Understand the **why**                 | **this document**                                   |

---

## §6. The empirical bet

The novelty claim laid out above is contingent on what the empirical
evaluation shows. The bet is that, when 5+ frontier LLMs are run
across the 16 environments under this protocol:

1. **Cross-model differences on evolution gain will be significant** —
   not within noise. If Claude / GPT / Gemini all show the same
   "post-iteration minus first-submit" delta, the capability axis is
   not separating models and the benchmark is weaker.
2. **First-submit ranking will not predict evolution-gain ranking** —
   meaning the capability is genuinely distinct from one-shot ability.
3. **Held-out gap will be quantifiable and non-trivial** — proving
   that iteration on in-loop seeds risks overfitting that
   held-out detects.
4. **Regression rate (iteration that makes things worse) will be
   measurable** — and will differ across models, capturing the
   "iteration discipline" sub-capability.

Any one of these findings is paper-worthy. All four together would
make this the canonical benchmark for the capability axis.

The benchmark is the measurement instrument. The paper is the first
measurement. The capability axis is the contribution.
