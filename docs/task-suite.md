# Task Suite

## Selection Criteria

A task is suitable for heuristic-learning training if it has:

- an executable environment;
- a clear scalar metric;
- reproducible seeds;
- failure cases that can be summarized;
- a policy that can be written as normal code;
- enough structure that iterative improvements are meaningful;
- low enough cost for repeated patch evaluation.

## MVP Suite

### 1. Adaptive ODE Solver Heuristic

Goal:

```text
choose timestep, method switch, retry, and tolerance adjustment policies
```

Inputs:

- current error estimate;
- accepted/rejected step history;
- stiffness indicators;
- compute budget;
- target accuracy.

Metrics:

- final error;
- number of function evaluations;
- rejection rate;
- timeout rate;
- robustness across equation families.

Why this is useful:

Numerical solvers already rely on explicit engineering heuristics. The task is
cheap, deterministic, and easy to run across many variants.

### 2. Toy Control Switching

Goal:

```text
choose high-level controller mode, gains, and recovery behavior
```

Candidate environments:

- cartpole swing-up;
- double integrator with saturation;
- simple quadrotor attitude simulator;
- HVAC temperature control.

Metrics:

- stability violations;
- settling time;
- overshoot;
- energy cost;
- recovery success after disturbances.

Constraint:

The learned heuristic may switch or tune controllers but should not bypass the
simulator or inspect hidden state.

### 3. Scheduling and Packing Heuristics

Goal:

```text
select, combine, and repair constructive heuristics
```

Candidate tasks:

- bin packing;
- job-shop scheduling;
- vehicle routing toy instances;
- cloud resource placement.

Metrics:

- objective value;
- constraint violations;
- runtime;
- performance gap to classical baselines.

Why this is useful:

These tasks have strong prior heuristic families and are cheap enough for
best-of-k sampling.

## Article-Inspired Expansion Suite

These tasks follow the spirit of the learning-beyond-gradients article but
should not be first in the training loop.

### Breakout or MinAtar Breakout

Use this after the MVP pipeline is stable.

Focus:

- state detection;
- paddle and ball heuristics;
- regression against known miss modes;
- sample efficiency under fixed frame budget.

### MuJoCo Ant or HalfCheetah

Use this after control-switching tasks work.

Focus:

- gait primitive selection;
- gain scheduling;
- residual correction;
- disturbance recovery.

### VizDoom

Use this only after cheaper visual or symbolic tasks are stable.

Focus:

- mode switching;
- map-specific failure modes;
- policy simplification after score improvements.

## Evaluation Splits

Each task should define:

- train seeds visible to the sampler;
- validation seeds used for candidate selection;
- held-out seeds used for reporting;
- held-out environment variants;
- hidden failure traces not shown to the sampler.

## Recommended Starting Point

Start with:

```text
adaptive_ode_v0
control_switching_v0
bin_packing_v0
```

This mix tests numerical, control, and combinatorial heuristic learning without
requiring expensive visual environments.
