# docs/envs.md — v1 Environment Roster

This document is the **authoritative roster** for the hlbench-pro v1
environment suite: **16 environments across 6 categories**, deliberately
selected to test "closed-loop, budget-constrained, LLM-driven policy
synthesis with held-out generalization."

The selection is calibrated against two design pressures:

1. **Policy synthesis must be the bottleneck.** Each env is chosen
   such that "the right code is non-obvious" — running a textbook
   algorithm doesn't max the score. The LLM has to look at rollouts,
   diagnose failure modes, and write structurally new code.
2. **Held-out generalization must matter.** Train and held-out
   instances differ in ways that defeat memorization or
   overfitting to in-loop seeds. For procedural envs this is
   automatic; for the others it's engineered via parameter ranges
   or curated instance pools.

For each env we document:

- **What it is** — observation/action spaces, episode dynamics, reward
- **Role in the suite** — what agent capability it probes
- **Current solutions** — known baselines (classical / RL / heuristic)
- **Policy-synthesis testability** — does writing the policy actually
  matter, or is it just recalling a textbook algorithm?
- **Expected discrimination** — how much we expect frontier models
  (Claude / GPT / Gemini / etc.) to differ on this env
- **Implementation cost** — single-developer estimate

---

## Composition at a glance

| #  | Env                          | Category                | Obs type                 | Discrimination | Cost     |
|----|------------------------------|-------------------------|--------------------------|----------------|----------|
| 1  | CarRacing-Pixel              | Visual control          | pixel 96×96×3            | **High**       | 2 d      |
| 2  | MiniGrid-Pixel-Hard          | Visual control          | pixel partial-view       | **High**       | 2 d      |
| 3  | Pendulum-from-Pixels         | Visual control          | pixel 64×64×3            | Medium         | 1 d      |
| 4  | Procgen-CoinRun              | Procedural visual       | pixel 64×64×3            | **High**       | 1 d      |
| 5  | Procgen-BigFish              | Procedural visual       | pixel 64×64×3            | **High**       | 1 d      |
| 6  | Procgen-Maze                 | Procedural visual       | pixel 64×64×3            | **High**       | 1 d      |
| 7  | Sokoban-Pixel                | Spatial reasoning       | pixel grid               | **High**       | 4 d      |
| 8  | Hidden-Maze                  | Spatial reasoning       | pixel revealed-map       | **High**       | 3 d      |
| 9  | Snake-Pixel                  | Visual game             | pixel grid               | Medium-High    | 2 d      |
| 10 | Frogger-Mini                 | Visual game             | pixel grid               | Medium         | 2 d      |
| 11 | Cache-Replacement            | Online algorithm        | state (cache + request)  | Medium         | 2 d      |
| 12 | K-Server                     | Online algorithm        | state (server pos + req) | Medium-High    | 1.5 d    |
| 13 | Online-Bipartite-Matching    | Online algorithm        | state (graph view)       | Medium         | 1.5 d    |
| 14 | Pendulum-Hardcore            | Hardcore control        | state Box(3)             | **High**       | 0.5 d    |
| 15 | Lunar-Hardcore               | Hardcore control        | state Box(8)             | **High**       | 0.5 d    |
| 16 | Bipedal-Hardcore             | Hardcore control        | state Box(24)            | **High**       | 0.5 d    |
| —  | **Total**                    | 6 categories            | 10 visual + 6 state      | —              | **~26 d**|

**Visual / state split**: 10 visual + 6 state. The state-based envs
are not filler — they serve as a deliberate cross-paradigm control
group, enabling the claim "visual envs are harder for LLMs by X%"
to be quantitatively defended.

---

## Category 1 — Visual control from pixels (3)

### 1. CarRacing-Pixel

**What it is.** Top-down 2D racing on a procedurally generated
track. Agent observes a 96×96×3 RGB frame and outputs continuous
controls (steering ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1]).
Reward: -0.1 per frame + 1000/N per visited track tile. Episode
terminates when all tiles visited or 1000 steps elapsed.

**Spaces.** obs: `Box(0, 255, (96, 96, 3), uint8)`, action:
`Box([-1, 0, 0], [1, 1, 1], (3,), float32)`, horizon: 1000.

**Role.** Visual control with continuous action — the canonical
"see image, output torque" task. Tests pixel→continuous-action
policy synthesis.

**Current solutions.** PPO/SAC from pixels (Gymnasium baseline ~900
score). Classical: HSV color segmentation to find road centerline
+ PD on lateral error (reaches ~700 with ~80 lines of code).

**Policy-synthesis testability.** **High.** An LLM can write a
competitive classical color-segmentation policy in one submit.
Iteration matters because the naive version fails on sharp turns,
off-track recovery, and shadow patches — each failure mode requires
a specific code addition.

**Expected discrimination.** **High.** Vision quality varies
considerably across frontier models; the gap between "extracts
road correctly from a few frames" and "writes a hard-coded turn
left every step" is a real model-capability test.

**Implementation cost.** ~2 days (wrap Gymnasium `CarRacing-v3` +
`observations.npy` storage). Wraps a stable Gymnasium env.

---

### 2. MiniGrid-Pixel-Hard

**What it is.** Gridworld navigation in a partially-observed
multi-room environment. Agent sees a 7×7 partial-view pixel patch
(its surroundings) and outputs one of 7 discrete actions
(forward, turn-left, turn-right, pickup, drop, toggle, done).
Goal: navigate from start to a target through doors that may
require picking up keys.

**Spaces.** obs: `Box(0, 255, (56, 56, 3), uint8)` (7×7 grid
upscaled), action: `Discrete(7)`, horizon: 200.

**Role.** Partial observability + visual symbol recognition
(key, door, goal colors) + multi-step planning. The agent must
maintain memory across steps because no full map is observable.

**Current solutions.** Recurrent PPO (RIMs, LSTM-based) reaches
near-optimal. Classical: frontier-based exploration + symbol
detection from pixel colors + BFS on inferred grid.

**Policy-synthesis testability.** **High.** The Policy instance
must maintain explicit state (visited cells, key inventory,
current goal) across `act()` calls — this stresses the
"persistent state in Policy attributes" pattern. Iteration
exposes loops, deadlocks, missed pickups.

**Expected discrimination.** **High.** Combines vision +
memory + planning; all three differ across frontier models.

**Implementation cost.** ~2 days. Wraps `minigrid` package
(needs adding to deps).

---

### 3. Pendulum-from-Pixels

**What it is.** Classic Pendulum-v1 (swing up + stabilize an
inverted pendulum) but the observation is a 64×64×3 RGB render
of the pendulum instead of the state vector (cos θ, sin θ, dθ).
Action: continuous torque.

**Spaces.** obs: `Box(0, 255, (64, 64, 3), uint8)`, action:
`Box(-2, 2, (1,), float32)`, horizon: 200.

**Role.** Visual extraction of physics state — the LLM must
infer angle (one frame) and angular velocity (two frames or
internal state) from images.

**Current solutions.** DQN/SAC with CNN from pixels (millions
of steps to converge). Classical: compute pendulum-arm angle
via image moments, then PD on angle (~50 lines of numpy).

**Policy-synthesis testability.** **High.** Naive
single-frame policies cannot compute velocity, so they
oscillate or undershoot. Iteration: observe oscillation,
add 2-frame state, get the PD working.

**Expected discrimination.** **Medium.** Pendulum dynamics
are simple, so once an LLM extracts angle correctly the
policy is short. The discriminator is "can the LLM write
image moments correctly" — borderline for weaker models.

**Implementation cost.** ~1 day. Existing pendulum + render
+ swap obs.

---

## Category 2 — Procedural visual generalization (3)

The defining feature of this category: **held-out seeds produce
visually different levels** with different layouts, enemy
placements, and obstacle distributions. Memorizing in-loop levels
fails on held-out — a clean generalization story.

### 4. Procgen-CoinRun

**What it is.** Side-scrolling platformer. Collect the coin at
the right end of the level, avoid saws and enemies. Each seed
generates a unique procedural level (varying platform layout,
enemy positions, gaps). Discrete actions (15 combinations of
left/right/jump/etc.).

**Spaces.** obs: `Box(0, 255, (64, 64, 3), uint8)`, action:
`Discrete(15)`, horizon: 1000.

**Role.** Visual generalization across procedurally generated
levels. The benchmark's headline generalization-test env.

**Current solutions.** PPO with IMPALA-CNN (Procgen paper
baseline ~5-7 average score; humans ~10). Heuristic: detect
platforms via color, detect enemies, time jumps from contour
analysis.

**Policy-synthesis testability.** **High.** A policy that
memorizes specific in-loop level layouts will fail on
held-out levels — the LLM is forced to write
distribution-invariant code (e.g., "always jump when there's
an obstacle below in the next 20 pixels").

**Expected discrimination.** **High.** Generalization to
held-out is the discriminator. Procgen was specifically
designed for this (Cobbe et al., 2019).

**Implementation cost.** ~1 day. Wraps `procgen` package
(needs adding to deps; gymnasium-compatible wrapper exists).

---

### 5. Procgen-BigFish

**What it is.** Agent controls a fish in a 2D ocean; eat
smaller fish to grow, avoid larger fish. Procedurally
generated each seed.

**Spaces.** obs: `Box(0, 255, (64, 64, 3), uint8)`, action:
`Discrete(15)`, horizon: 1000.

**Role.** Visual relative-size judgment + dynamic
obstacle avoidance.

**Current solutions.** PPO baseline (~3-5 score, humans ~5-10).
Heuristic: estimate sizes from pixel-area, approach smaller,
flee larger.

**Policy-synthesis testability.** **High.** "Estimate size
from pixels" is a non-trivial visual reasoning task; the
greedy heuristic has many failure modes (occlusion,
boundary effects) that iteration can address.

**Expected discrimination.** **High.**

**Implementation cost.** ~1 day.

---

### 6. Procgen-Maze

**What it is.** Navigate a procedurally generated 2D maze to
reach a cheese. Discrete actions (4 movement directions).

**Spaces.** obs: `Box(0, 255, (64, 64, 3), uint8)`, action:
`Discrete(15)` (only 4 effective), horizon: 500.

**Role.** Visual maze-topology extraction + path planning.

**Current solutions.** RL with CNN (~5 PPO). Classical:
extract maze structure from pixels via thresholding + run
BFS on the inferred grid.

**Policy-synthesis testability.** **High.** "Extract maze
structure correctly from pixels + run BFS" is a clean
two-stage problem with measurable subgoals; LLM iterations
on each stage are visible in rollouts.

**Expected discrimination.** **High.**

**Implementation cost.** ~1 day.

---

## Category 3 — Spatial reasoning & search (2)

### 7. Sokoban-Pixel

**What it is.** Classic Sokoban: push boxes onto target squares
without getting them stuck against walls or in corners. Pixel
rendering (each cell is a 16×16 sprite; board sizes 8×8 to 12×12).
Curated puzzle set: train pool = 10000 random-generated puzzles
of varying difficulty (verified solvable); held-out = 256
hand-curated harder puzzles.

**Spaces.** obs: `Box(0, 255, (192, 192, 3), uint8)`, action:
`Discrete(4)` (up/down/left/right), horizon: 200.

**Role.** Spatial planning under combinatorial complexity —
the canonical test for "can the policy plan ahead and avoid
irreversible mistakes."

**Current solutions.** Classical: BFS on state space (works
on tiny puzzles), A* with heuristics (e.g., box-target
Manhattan distance), specialized solvers (Festival, Patrick
Spettel) that use deadlock detection + pattern databases.
Naive BFS times out on >6 boxes.

**Policy-synthesis testability.** **High.** This is the
benchmark's most ambitious env for iteration: BFS alone
fails on hard puzzles → LLM must add deadlock detection
(corner detection, 2×2 trap detection, freeze deadlock,
etc.) → each pattern is a code change that improves on
specific failed puzzles. Highly visible iteration.

**Expected discrimination.** **High.** Sokoban is known
hard for LLMs — recent DeepMind work has shown Gemini
struggles with Sokoban-like planning. Differences between
frontier models on deadlock-detection sophistication are
real.

**Implementation cost.** ~4 days. Need a custom Sokoban
engine (cheap), puzzle generator (medium), and curated
held-out set (expensive — needs solvability verification).

---

### 8. Hidden-Maze

**What it is.** A 2D grid maze where walls are revealed only
as the agent steps adjacent to them. The obs shows the
revealed map so far + the agent's position. Action: discrete
movement.

**Spaces.** obs: `Box(0, 255, (128, 128, 3), uint8)`, action:
`Discrete(4)`, horizon: 300.

**Role.** Visual exploration + map building under
uncertainty + planning with incomplete information.

**Current solutions.** Frontier-based exploration (classical
robotics), POMDP solvers (overkill).

**Policy-synthesis testability.** **High.** The Policy must
maintain an explicit internal map data structure across
`act()` calls (Python dict, numpy grid, whatever). This
stresses cross-step state — many naive policies will fail
to remember explored cells and re-traverse them.

**Expected discrimination.** **High.**

**Implementation cost.** ~3 days. Custom env (no off-the-shelf
fits exactly).

---

## Category 4 — Visual game (2)

### 9. Snake-Pixel

**What it is.** Classic Snake: eat food to grow, avoid
self-collision and walls. Pixel rendering of an N×N grid
(N = 20).

**Spaces.** obs: `Box(0, 255, (160, 160, 3), uint8)`,
action: `Discrete(4)` (up/down/left/right), horizon: 500.

**Role.** Simple visual game with growing-state complexity
— early game is trivial (BFS to food works), late game
requires lookahead (BFS-to-food traps the snake).

**Current solutions.** BFS to nearest food (works early),
Hamiltonian cycle (optimal but unrewarding), greedy +
1-step lookahead, A* with safety check.

**Policy-synthesis testability.** **High.** Naive
BFS-to-food fails late game when the snake is large.
LLM observes "snake died trying to reach food in a
corner" → must add lookahead or safety check. Several
incremental improvements possible.

**Expected discrimination.** **Medium-High.** Score range
is wide (10 to 100+ apples) — clear separator.

**Implementation cost.** ~2 days. Custom simulator.

---

### 10. Frogger-Mini

**What it is.** Cross multiple lanes of moving traffic and
hop across logs in a river to reach the top of the screen.
Pixel rendering of a 9×11 grid.

**Spaces.** obs: `Box(0, 255, (176, 144, 3), uint8)`,
action: `Discrete(5)` (up/down/left/right/wait), horizon: 200.

**Role.** Visual hazard avoidance + timing + multi-step
planning. Tests "extract dynamic obstacles from pixels,
predict, time crossings."

**Current solutions.** Extract traffic positions and
velocities from frame-pairs + greedy with safety margin.

**Policy-synthesis testability.** **High.** Naive
"always move up" fails (gets hit). LLM must extract
traffic state + reason about future positions.

**Expected discrimination.** **Medium.** Most frontier
models will eventually crack this; differences are in
how quickly.

**Implementation cost.** ~2 days. Custom simulator.

---

## Category 5 — Online algorithms (3, state-based)

These three envs are deliberately state-based (not pixels) for
two reasons: (a) the problems are intrinsically combinatorial,
not visual; (b) they form the state-based cross-paradigm
comparison group for the visual envs.

### 11. Cache-Replacement

**What it is.** Stream of memory accesses (object IDs).
Cache capacity C; on a miss the policy chooses which
cached item to evict. Reward: +1 per cache hit. Trace
distributions vary per seed (Zipfian / scan-heavy /
LRU-friendly / LRU-hostile mixtures).

**Spaces.** obs: state dict `{"cache": list[int],
"access": int, "history_window": list[int]}`, action:
`Discrete(C)` (index in cache to evict), horizon: trace
length (typically 10000 accesses).

**Role.** Online algorithm design under adversarial
distributions — tests "can the policy recognize the
access pattern type and switch strategy."

**Current solutions.** LRU / LFU / ARC / 2Q / LIRS
(textbook). ML-based: LeCar, Glider, Cacheus. Belady's
OPT (oracle upper bound).

**Policy-synthesis testability.** **Medium.** This is
the env most at risk of saturation — frontier LLMs will
implement ARC and likely all hit similar scores.
Mitigated by including trace distributions where ARC
underperforms (scan-heavy), forcing the LLM to either
recognize the pattern or accept the loss.

**Expected discrimination.** **Medium.** Saturation
risk; needs adversarial trace tuning to keep
discriminating.

**Implementation cost.** ~2 days. Trace generator +
simulator.

---

### 12. K-Server

**What it is.** k servers on a metric space (here:
points in 2D Euclidean plane). Requests arrive at
points; the policy must dispatch one server to that
point. Reward: -distance moved. Standard hard problem
in competitive analysis.

**Spaces.** obs: state `{"servers": ndarray(k, 2),
"request": ndarray(2)}`, action: `Discrete(k)`,
horizon: 500 requests.

**Role.** Online optimization with competitive ratio
context — tests "can the LLM design a policy that
beats greedy without going full-WFA."

**Current solutions.** Work Function Algorithm (WFA;
optimal competitive ratio but O(k! · n) compute),
Double Coverage (2-competitive on trees), greedy
(closest server; not competitive in general),
randomized algorithms.

**Policy-synthesis testability.** **High.** WFA is
intractable in 10ms act_wall; simple heuristics
suboptimal. Real space for LLM to design an
intermediate strategy (e.g., greedy with anticipation
penalty).

**Expected discrimination.** **Medium-High.**

**Implementation cost.** ~1.5 days.

---

### 13. Online-Bipartite-Matching

**What it is.** Bipartite graph with N left vertices
fixed at the start; right vertices arrive online with
edges to a subset of left vertices. The policy must
match each arriving right vertex immediately (or skip)
to one of its neighbors that isn't matched yet. Reward:
+1 per match (or sum of edge weights for weighted
variant).

**Spaces.** obs: state `{"left_matched": ndarray(N, bool),
"arrival": dict{"neighbors": list[int], "weights": list[float]}}`,
action: `Discrete(N+1)` (which left vertex to match, or N
for skip), horizon: M right-vertex arrivals.

**Role.** Online algorithm with strong theoretical
baselines — tests "can the LLM beat the (1-1/e)
competitive ratio of RANKING on adversarial inputs."

**Current solutions.** RANKING algorithm (Karp-Vazirani-
Vazirani 1990, (1-1/e)-competitive), greedy, water-filling
for fractional variant.

**Policy-synthesis testability.** **Medium.** RANKING
is well-known; LLM may default to it. Real test: graph
distributions where RANKING's gap to OPT is large
(stochastic graphs where structure can be exploited).

**Expected discrimination.** **Medium.**

**Implementation cost.** ~1.5 days.

---

## Category 6 — Hardcore state-based control (3)

These three are deliberately the **cross-paradigm comparison
group**. State-based, no vision, classical control territory —
the LLM's strongest setting in principle. Including them
enables the paper finding: "Frontier models score X on
state-based control but Y on visual envs; the gap measures
visual reasoning capability."

### 14. Pendulum-Hardcore

**What it is.** Pendulum-v1 but with mass, length, and gravity
sampled per seed across wide ranges. Training pool: mass ∈
[0.3, 3.0], length ∈ [0.5, 2.0], g ∈ [5.0, 15.0]. Held-out
pool: OOD ranges (mass ∈ [3.0, 5.0], etc.).

**Spaces.** obs: `Box(-inf, inf, (3,), float32)` (cos θ, sin θ,
dθ/dt) + `env_meta` declares the (mass, length, g) ranges;
action: `Box(-2, 2, (1,), float32)`, horizon: 200.

**Role.** Domain-randomized control — tests "can the LLM
write an adaptive controller that reads the env's parameter
range and adjusts gains, vs. a fixed PD that fails on
held-out OOD."

**Current solutions.** Fixed-gain PD (fails on extremes),
gain-scheduled PD (reads env_meta, sets gains as a function
of mass/length), MPC (heavier but works), adaptive control
(estimates params online from initial swings).

**Policy-synthesis testability.** **High.** Naive PD with
in-loop-tuned gains will overfit. Held-out OOD reveals
this immediately — the LLM must either read env_meta and
schedule gains, or do online identification.

**Expected discrimination.** **High.** Read-the-env_meta vs
hardcoded-gains is a clear separator.

**Implementation cost.** ~0.5 days. Modify existing
`pendulum` env: change seed generator to sample
(mass, length, g), expose ranges in env_meta.

---

### 15. Lunar-Hardcore

**What it is.** LunarLanderContinuous-v3 with randomized
terrain (varying landing-pad position + crater profiles),
wind, and gravity. Train pool: standard ranges; held-out
pool: extreme conditions.

**Spaces.** obs: `Box(8,)` (position, velocity, angle,
contact flags), action: `Box(2,)` (main engine, side
engine), horizon: 1000.

**Role.** Higher-dim domain-randomized control with
discrete-event reward structure (landing bonus,
crashing penalty).

**Current solutions.** Hand-tuned PID with mode switching
(descend → align → land). MPC. PPO/SAC trained from
scratch.

**Policy-synthesis testability.** **High.** Multiple
control modes (descent, alignment, touchdown) need to be
coded explicitly; failure modes are visible (crash, drift
out, run out of fuel) and each suggests a code change.

**Expected discrimination.** **High.**

**Implementation cost.** ~0.5 days. Modify existing
`lunar_lander_continuous` env: add wind / terrain
randomization, regenerate pools.

---

### 16. Bipedal-Hardcore

**What it is.** Gymnasium's `BipedalWalkerHardcore-v3` —
biped locomotion over irregular terrain with ladders,
stumps, and pits. Pre-existing hard variant of
BipedalWalker.

**Spaces.** obs: `Box(24,)` (joint angles, velocities,
lidar), action: `Box(4,)` (joint torques), horizon: 2000.

**Role.** Complex high-DoF locomotion; the most
mechanically challenging env in the suite. Known hard
for hand-crafted controllers.

**Current solutions.** PPO/SAC trained for millions of
steps (canonical), specialized hand-crafted gaits (very
rare to write competitively).

**Policy-synthesis testability.** **High.** Hand-crafting
a competitive gait is genuinely hard. Most LLM attempts
will score low — but the FAILURE PATTERNS will differ
across models (some write open-loop CPG, some try
reactive sensor-based, some default to constant torques).
The failure-mode diversity itself is a discrimination
signal.

**Expected discrimination.** **High.** Floor is low
but the spread is wide.

**Implementation cost.** ~0.5 days. Wrap Gymnasium's
existing `BipedalWalkerHardcore-v3` (currently we have
`BipedalWalker-v3`).

---

## Cross-cutting capability matrix

Which agent capabilities each env stresses (✓ = primary,
~ = secondary):

| Env | Vision | Memory | Search | Online decision | Generalization | Code complexity |
|---|---|---|---|---|---|---|
| 1. CarRacing-Pixel | ✓ | ~ |  |  |  | ~ |
| 2. MiniGrid-Pixel-Hard | ✓ | ✓ | ✓ |  |  | ✓ |
| 3. Pendulum-from-Pixels | ✓ | ~ |  |  |  |  |
| 4. Procgen-CoinRun | ✓ |  |  |  | ✓ |  |
| 5. Procgen-BigFish | ✓ |  |  | ✓ | ✓ |  |
| 6. Procgen-Maze | ✓ |  | ✓ |  | ✓ |  |
| 7. Sokoban-Pixel | ✓ | ✓ | ✓ |  |  | ✓ |
| 8. Hidden-Maze | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| 9. Snake-Pixel | ✓ |  | ~ | ✓ |  |  |
| 10. Frogger-Mini | ✓ | ~ |  | ✓ |  |  |
| 11. Cache-Replacement |  | ✓ |  | ✓ | ~ |  |
| 12. K-Server |  | ~ | ~ | ✓ |  |  |
| 13. Online-Bipartite-Matching |  | ~ |  | ✓ |  |  |
| 14. Pendulum-Hardcore |  |  |  |  | ✓ |  |
| 15. Lunar-Hardcore |  | ~ |  |  | ✓ | ~ |
| 16. Bipedal-Hardcore |  |  |  |  |  | ✓ |

**Coverage check.** Every capability has ≥4 envs probing
it, satisfying the per-category statistical power floor for
"X-capability findings" in the paper.

---

## Migration from v0 (current 5 envs)

| v0 env (current) | v1 fate | Notes |
|---|---|---|
| `pendulum` | → #14 `Pendulum-Hardcore` | Add parameter randomization to seed generator |
| `lunar_lander_continuous` | → #15 `Lunar-Hardcore` | Add wind / terrain randomization |
| `bipedal_walker` | → #16 `Bipedal-Hardcore` | Swap to `BipedalWalkerHardcore-v3` |
| `acrobot` | **Removed from v1 roster** | Kept in codebase as legacy test env; not part of the scored suite |
| `mountain_car_continuous` | **Removed from v1 roster** | Same as `acrobot` |

Both `acrobot` and `mountain_car_continuous` lack genuine
iteration depth — a trivial policy already approaches
expert. They remain in the repo for backward compatibility
and as integration-test envs but are not part of the v1
scored suite.

---

## Implementation prerequisites

Before any visual env (#1–#10) can be implemented, the harness
needs three pieces of infrastructure that are currently absent
or under-spec'd:

1. **`observations.npy` writer** (deferred from 0.1.0a1,
   `SPEC §4.6`). Per-episode pixel observations are too large
   for inline JSON in `trajectory.jsonl` — they must go to a
   binary side-car. The `obs_storage: "external"` mechanism is
   spec'd but not implemented.
2. **Per-env `act_wall_ms` override.** Current default of 10 ms
   is hostile to any vision processing. Visual envs need ~100 ms
   per `act()`. This requires lifting the "later layers may
   only tighten" rule in `AGENTS.md §8` to allow per-env
   relaxation, and exposing the effective value in `/info`.
3. **Anti-cheat extension for vision libs.** `denied_imports`
   must add pretrained-vision packages: `clip`, `open_clip`,
   `transformers` (already there), `timm` (already there),
   `dinov2`, `sam`, `ultralytics` (pretrained YOLO), and any
   other pretrained ViT distributions.

Estimated infrastructure cost: ~3 weeks (item 1 is the bulk).
Visual env implementation can begin in parallel with item 3.

## Build cost summary

- **Infrastructure** (items 1–3 above): ~3 weeks
- **State-based envs** (#11–#16): ~6 days, can start immediately
  (no infra dependency)
- **Visual envs** (#1–#10): ~20 days, gated on infrastructure
- **Frontier-model evaluation matrix** (5 models × 16 envs):
  ~1 week (after envs done)
- **Calibration sweep** (random/expert baselines per env):
  ~1 week

**Total time-to-v1**: ~10–12 weeks single-developer, in line
with the M1–M3 estimate in the v0.1 roadmap.

---

## Discrimination ceiling claim

If the suite is built as specified and frontier models are
evaluated:

- We expect **at least 10 of 16 envs** to show clean
  inter-model gradient (>20 score-point spread between
  Claude 4.7 / GPT-5 / Gemini 2.5).
- Categories most likely to discriminate cleanly: **Visual
  control, Procedural visual, Spatial reasoning**.
- Categories most at risk of saturation: **Online algorithms**
  (textbook baselines may dominate), **Bipedal-Hardcore**
  (floor effect — most models near zero).
- If saturation happens on >6 envs after the first eval pass,
  the suite needs a calibration pass before paper submission
  (swap adversarial distributions, harder puzzle pools, etc.).

This document is the contract: each env in the roster must
satisfy the "policy synthesis is the bottleneck" and
"held-out generalization matters" tests before being
counted toward the v1 suite. Envs failing this check after
implementation should be cut from the roster, not patched.
