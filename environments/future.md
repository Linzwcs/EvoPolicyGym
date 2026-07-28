# Future Environment Candidates

This note records prospective environments for Coding Agent optimization of
executable Policy systems. It is a research priority list, not an
integration-status claim. Runnable and deferred integrations remain
authoritative in [STATUS.md](STATUS.md).

## Provenance and adoption rule

Candidates are named and organized by their authoritative upstream ecosystem,
as required by [the environment taxonomy](README.md). EdgeBench is a secondary
discovery and comparison source: it repackages upstream games, contest
problems, benchmarks, and APIs as long-running workspace-plus-Judge tasks. An
EdgeBench task ID identifies that derived task, not the owner of the underlying
Environment.

EvoPolicyGym will not import, vendor, or depend on EdgeBench task workspaces,
container images, prompts, baselines, generators, testers, Judge programs,
hidden cases, score anchors, feedback traces, or SForge. For each candidate we
instead:

- locate the official upstream specification, repository, release, and assets;
- verify the upstream code, data, and asset licenses independently;
- use an official runtime directly when it already exposes interaction;
- otherwise implement an independent adapter from the public upstream rules;
  and
- reject the candidate if no authoritative and redistributable upstream can be
  established.

EdgeBench's CC BY 4.0 task-metadata license does not establish redistribution
rights for the upstream software, contest materials, games, ROMs, or datasets
referenced by a task.

## Verified provenance map

The EdgeBench column documents what EdgeBench changed so that the distinction
is auditable. Those changes are not implementation inputs for EvoPolicyGym.

| Upstream candidate | Authoritative upstream | EdgeBench-derived task and alteration | EvoPolicyGym decision |
| --- | --- | --- | --- |
| BipedalWalker | [Gymnasium `BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) | `bipedalwalker_locomotion_rl`: trains a checkpoint and scores only the submitted artifact | Keep the existing direct Gymnasium integration |
| Treant's Forest | [AtCoder AHC054](https://atcoder.jp/contests/ahc054/tasks/ahc054_a) | `treant_forest`: adds a Python baseline, local tooling, fixed Judge cases, and score normalization | Keep the independent direct integration; no EdgeBench or AtCoder tool content is packaged |
| Warehouse Manager | [CodeChef `WAREHOUS`](https://www.codechef.com/problems/WAREHOUS) | `warehouse_forklift_routing`: adds a Python workspace, baseline, local generator/tester, hidden evaluation, and rescaling | Keep the independent direct integration; one atomic Action preserves the official complete-output contract, and no CodeChef content is packaged |
| Apple Incremental Game | [AtCoder AHC058](https://atcoder.jp/contests/ahc058/tasks/ahc058_a) | `apple_incremental_game`: packages the contest task with a baseline, local tooling, fixed Judge cases, and rescaling | Keep the independent direct integration with one official turn per Action and no AtCoder tool content |
| NetHack | [NetHack](https://github.com/NetHack/NetHack) and maintained [NLE](https://github.com/NetHack-LE/nle) | `nethack_dungeon_agent`: fixes an observation/policy scaffold and scores multiple generated runs | Defer to a pinned engine execution profile: NLE 1.3.0 supports Python 3.12 and Linux wheels, but native macOS installation requires a CMake 3.28+ source toolchain not owned by the current distribution model |
| VRPTW | [Solomon VRPTW benchmark](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/) | `vehicle_routing_time_windows`: turns solver output into hidden-instance and best-known-solution scoring | Defer until the instance files have explicit redistribution terms and a constructive interaction contract is accepted; do not repackage EdgeBench's hidden instances or score anchors |
| Molecules | [AtCoder AHC057](https://atcoder.jp/contests/ahc057/tasks/ahc057_a) | `molecular_self_assembly`: adds a Python baseline, local tooling, fixed Judge cases, and rescaling | Keep the independent direct integration with atomic bond sets, public motion rules, and no AtCoder tool content |
| Battle for Wesnoth | [Wesnoth Lua AI API](https://wiki.wesnoth.org/LuaAPI/ai) | `wesnoth_tactical_ai`: selects tactical maps, objectives, an opponent, and a scoring Judge | Defer to an engine execution profile that owns a pinned headless Wesnoth process, Lua bridge, scenario data, timeouts, and cleanup |
| OpenRCT2 | [OpenRCT2](https://github.com/OpenRCT2/OpenRCT2) | `openrct2_theme_park_ai`: defines a JavaScript automation plugin task and hidden scenarios | Defer: the engine is open source but normal play requires separately licensed RCT2 files |
| OpenTTD | [OpenTTD NoAI API](https://docs.openttd.org/ai-scripting/ai-api/) | `openttd_transport_ai`: defines an AI-script objective, generated maps, and company-value scoring | Defer to an engine execution profile that pins the OpenTTD binary, base graphics, NoAI bridge, generated maps, and long-running process lifecycle |
| Dungeon Crawl Stone Soup | [DCSS](https://github.com/crawl/crawl) | `dcss_dungeon_ai`: fixes a Lua bot interface, character build, time budget, repeated runs, and mean score | Defer to an engine execution profile that pins a headless DCSS build and validates a supported machine-control surface |

The public [EdgeBench task metadata](https://huggingface.co/datasets/ByteDance-Seed/EdgeBench)
and [paper](https://edge-bench.org/paper.pdf) are retained only as evidence of
the derived task designs.

## Recommended roadmap

| Priority | Upstream candidate | Policy-system value | Main concern |
| --- | --- | --- | --- |
| 0 | Gymnasium BipedalWalker | Fast continuous-control calibration; already integrated | Useful as a baseline, not a flagship systems task |
| 1 | AtCoder AHC054 Treant's Forest | Native turn-by-turn interaction, state tracking, constraint handling, and replanning | Integrated independently with a bounded turn horizon |
| 2 | CodeChef WAREHOUS Warehouse Manager | Constructive routing, storage strategy, sliding-state planning, and cost optimization | Integrated independently over the full official size range with an atomic complete-solution Action |
| 3 | AtCoder AHC058 Apple Incremental Game | Investment timing, horizon estimation, and phase-dependent decisions | Integrated independently with one official turn per Action over the complete 500-turn horizon |
| 4 | NLE NetHack | Partial observability, map memory, exploration, inventory, combat, and risk | Deferred until a pinned NLE engine profile is portable across the supported execution matrix |
| 5 | Solomon VRPTW | Constructive search, feasibility management, and route repair | Deferred pending explicit instance redistribution terms and a genuine constructive interaction contract |
| 6 | AtCoder AHC057 Molecules | Temporal scheduling and constrained constructive planning | Integrated independently across all 300 points and 1,000 official turns with bounded bond-event diagnostics |
| Later | Battle for Wesnoth | Tactical planning, centralized multi-unit control, and opponent response | Deferred to a pinned headless engine/Lua execution profile |
| Later | OpenTTD | Network design, capital allocation, expansion, and recovery | Deferred to a pinned NoAI engine profile with long-Run lifecycle ownership |
| Later | DCSS | Exploration, combat, inventory, and survival strategy | Deferred to a pinned headless engine profile and validated control surface |
| Deferred | OpenRCT2 | Hierarchical resource allocation and long-term management | Original RCT2 data files are a separate asset dependency |

Treant's Forest, Warehouse Manager, Molecules, and Apple Incremental Game are
completed independent integrations. They deliberately retain their different
task semantics through two accepted Policy interaction shapes: the three AHC
integrations expose one meaningful simulated turn per Action, while Warehouse
Manager preserves its complete-output contract and validates one instruction
Program atomically. NetHack is the strongest flagship candidate once the
runtime and long-horizon evaluation workflow are ready.

## Suggested capability gradient

```text
continuous control
└── Gymnasium BipedalWalker

native online interaction
└── AtCoder AHC054 Treant's Forest

fast long-horizon decisions
└── AtCoder AHC058 Apple Incremental Game

constructive planning
├── CodeChef WAREHOUS Warehouse Manager
├── Solomon VRPTW
└── AtCoder AHC057 Molecules

partially observable systems
├── NLE NetHack
└── DCSS

hierarchical strategy
├── Battle for Wesnoth
└── OpenTTD
```

This gradient supports experiments on how Coding Agents progress from tuning
reactive controllers to authoring Policies with explicit memory, search,
subsystems, and hierarchical planning. The first four runnable steps are
represented by existing Gymnasium distributions plus the three AtCoder and one
CodeChef distributions above. Engine-backed steps remain roadmap entries, not
runnable claims.

## Conditional and rejected leads

[AtCoder AHC056 Grid Turing Robot](https://atcoder.jp/contests/ahc056/tasks/ahc056_a)
and [CodeChef `TRICOL`](https://www.codechef.com/problems/TRICOL) are verified
upstreams for EdgeBench's `grid_turing_robot` and
`triangulation_coloring_optimization`. They remain conditional because their
official form grades a completed design. They qualify only if individual
Actions expose meaningful construction state rather than an artificial
one-step Episode. The current roadmap does not admit them; one-shot support by
itself is not sufficient evidence of useful Policy interaction.

The EdgeBench-only leads `vibrating_path_graph_coloring`,
`order_addition_permutation_optimization`, and its particular `smt_solver`
workspace are not candidates without an independently verified upstream. A
future SMT environment must begin from the official
[SMT-LIB](https://smt-lib.org/) standards and benchmark ecosystem, not from the
EdgeBench scaffold.

The text adventures `anchorhead_text_adventure`,
`trinity_text_adventure`, and `tryst_text_adventure` refer to existing
interactive-fiction games exposed by EdgeBench through an HTTP wrapper. Any
reconsideration must start from each game's publisher, distribution rights, and
interpreter ecosystem. Fixed worlds also make walkthrough hard-coding difficult
to distinguish from general Policy improvement.

Systems engineering, formal-proof, document-production, and large-workspace
tasks should not be presented as EvoPolicyGym Environments. They grade
repositories, proofs, checkpoints, or documents rather than closed-loop
Policies. Faithful support would require a separate Artifact/Workspace Judge
use case instead of weakening the current `Environment.reset()` /
`Environment.step()` contract.

## Integration requirements

Every accepted candidate must be an independent Benchmark distribution using
the public authoring SPI described in
[ARCHITECTURE.md](../ARCHITECTURE.md). It must provide:

- a recorded authoritative upstream URL, version or commit, and license;
- explicit code, data, model, ROM, scenario, and asset provenance;
- no runtime or build dependency on EdgeBench or SForge;
- a fresh Environment and Policy lifecycle for every Episode;
- strict, unrepaired Action validation;
- deterministic split-scoped Episode generation;
- a scalar objective plus bounded diagnostic Feedback;
- private Validation and held-out Assessment coverage; and
- real upstream smoke tests and cleanup tests.

No `environments/edgebench/` namespace is planned. EdgeBench task IDs appear
only in the provenance map to document how a secondary benchmark transformed
the corresponding upstream material.
