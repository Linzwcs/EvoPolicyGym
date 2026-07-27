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
| Treant's Forest | [AtCoder AHC054](https://atcoder.jp/contests/ahc054/tasks/ahc054_a) | `treant_forest`: adds a Python baseline, local tooling, fixed Judge cases, and score normalization | Implement the official interactive rules independently; do not use the EdgeBench workspace |
| Warehouse Manager | [CodeChef `WAREHOUS`](https://www.codechef.com/problems/WAREHOUS) | `warehouse_forklift_routing`: adds a Python workspace, baseline, local generator/tester, hidden evaluation, and rescaling | Recreate a stepwise simulator only from the official problem contract, after license review |
| Apple Incremental Game | [AtCoder AHC058](https://atcoder.jp/contests/ahc058/tasks/ahc058_a) | `apple_incremental_game`: packages the contest task with a baseline, local tooling, fixed Judge cases, and rescaling | Implement the public turn dynamics and our own split-scoped Episode generator |
| NetHack | [NetHack](https://github.com/NetHack/NetHack) and [NLE](https://github.com/facebookresearch/nle) | `nethack_dungeon_agent`: fixes an observation/policy scaffold and scores multiple generated runs | Integrate NLE directly and define EvoPolicyGym-owned observations, Actions, horizons, and scoring |
| VRPTW | [Solomon VRPTW benchmark](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/) | `vehicle_routing_time_windows`: turns solver output into hidden-instance and best-known-solution scoring | Use independently obtained benchmark instances; admit only a genuine constructive interaction design |
| Molecules | [AtCoder AHC057](https://atcoder.jp/contests/ahc057/tasks/ahc057_a) | `molecular_self_assembly`: adds a Python baseline, local tooling, fixed Judge cases, and rescaling | Implement the public motion and bonding rules with independent Episode generation |
| Battle for Wesnoth | [Wesnoth Lua AI API](https://wiki.wesnoth.org/LuaAPI/ai) | `wesnoth_tactical_ai`: selects tactical maps, objectives, an opponent, and a scoring Judge | Integrate the official engine/API directly and author independent scenarios |
| OpenRCT2 | [OpenRCT2](https://github.com/OpenRCT2/OpenRCT2) | `openrct2_theme_park_ai`: defines a JavaScript automation plugin task and hidden scenarios | Defer: the engine is open source but normal play requires separately licensed RCT2 files |
| OpenTTD | [OpenTTD NoAI API](https://docs.openttd.org/ai-scripting/ai-api/) | `openttd_transport_ai`: defines an AI-script objective, generated maps, and company-value scoring | Integrate the official engine and NoAI API directly with independently defined profiles |
| Dungeon Crawl Stone Soup | [DCSS](https://github.com/crawl/crawl) | `dcss_dungeon_ai`: fixes a Lua bot interface, character build, time budget, repeated runs, and mean score | Integrate the official game and supported control surface directly |

The public [EdgeBench task metadata](https://huggingface.co/datasets/ByteDance-Seed/EdgeBench)
and [paper](https://edge-bench.org/paper.pdf) are retained only as evidence of
the derived task designs.

## Recommended roadmap

| Priority | Upstream candidate | Policy-system value | Main concern |
| --- | --- | --- | --- |
| 0 | Gymnasium BipedalWalker | Fast continuous-control calibration; already integrated | Useful as a baseline, not a flagship systems task |
| 1 | AtCoder AHC054 Treant's Forest | Native turn-by-turn interaction, state tracking, constraint handling, and replanning | Requires an independent tester and a redistribution review |
| 2 | CodeChef WAREHOUS Warehouse Manager | Long-horizon routing, memory, storage strategy, and recovery | The upstream task emits a complete command string; the Policy form must be stepwise |
| 3 | AtCoder AHC058 Apple Incremental Game | Investment timing, horizon estimation, and phase-dependent decisions | Must expose one official turn per Policy Action |
| 4 | NLE NetHack | Partial observability, map memory, exploration, inventory, combat, and risk | Linux-oriented runtime and expensive sparse-reward Episodes |
| 5 | Solomon VRPTW | Constructive search, feasibility management, and route repair | Risks becoming a one-shot solver benchmark |
| 6 | AtCoder AHC057 Molecules | Temporal scheduling and constrained constructive planning | Needs useful intermediate observations and bounded diagnostics |
| Later | Battle for Wesnoth | Tactical planning, centralized multi-unit control, and opponent response | Large state/action surface and engine bridge |
| Later | OpenTTD | Network design, capital allocation, expansion, and recovery | Very long simulations and substantial engine integration |
| Later | DCSS | Exploration, combat, inventory, and survival strategy | Long wall-clock Episodes and control-surface validation |
| Deferred | OpenRCT2 | Hierarchical resource allocation and long-term management | Original RCT2 data files are a separate asset dependency |

Treant's Forest is the strongest first new investigation because its
authoritative upstream is already an interactive protocol. Warehouse Manager
remains attractive, but requires more semantic re-authoring from batch command
output into a Policy loop. NetHack is the strongest flagship candidate once the
runtime and long-horizon evaluation workflow are ready.

## Suggested capability gradient

```text
continuous control
└── Gymnasium BipedalWalker

native online interaction
└── AtCoder AHC054 Treant's Forest

fast long-horizon decisions
├── AtCoder AHC058 Apple Incremental Game
└── CodeChef WAREHOUS Warehouse Manager

constructive planning
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
subsystems, and hierarchical planning.

## Conditional and rejected leads

[AtCoder AHC056 Grid Turing Robot](https://atcoder.jp/contests/ahc056/tasks/ahc056_a)
and [CodeChef `TRICOL`](https://www.codechef.com/problems/TRICOL) are verified
upstreams for EdgeBench's `grid_turing_robot` and
`triangulation_coloring_optimization`. They remain conditional because their
official form grades a completed design. They qualify only if individual
Actions expose meaningful construction state rather than an artificial
one-step Episode.

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
