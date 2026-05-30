# paper/table.md — HLBench-Pro v1 Scored Suite (16 envs, 4 categories)

The v1 evaluation suite is **16 environments across four Gymnasium
categories**, structured to mirror the `hlbench` reference paper's
table (ICLR 2026 submission, `benchmark.tex` Table 2) but
deliberately substituted with **harder variants** wherever Gymnasium
ships one. All envs use the latest Gymnasium APIs (Box2D v3, MuJoCo
v5, MiniGrid v1 where available).

This document is the authoritative roster for paper Table 1
("Implemented HLBench-Pro environments") and for the calibration
harness (`scripts/run_matrix.py`).

---

## Suite at a glance

| # | Category | hlbench-pro env id | Gymnasium env (latest) | Obs | Action | Horizon | Hardness rationale |
|---|---|---|---|---|---|---|---|
| 1 | Classic Control | `cartpole_balance` | `CartPole-v1` | Box(4) state | Discrete(2) | 500 | **EASY anchor** — angle-based PD hits the 500 ceiling; calibration baseline for score-distribution diagnostics |
| 2 | Classic Control | `pendulum` | `Pendulum-v1` | Box(3) state | Box(1) cont. | 200 | **MEDIUM** — textbook PD/LQR baseline, continuous action |
| 3 | Classic Control | `acrobot` | `Acrobot-v1` | Box(6) state | Discrete(3) | 500 | **MEDIUM-HARD** — under-actuated swing-up, 2-link non-linear, sparse-style reward |
| 4 | Classic Control | `mountain_car_continuous` | `MountainCarContinuous-v0` | Box(2) state | Box(1) cont. | 999 | **HARD** — sparse-reward exploration, random rarely solves |
| 5 | Box2D | `lunar_hardcore` | `LunarLanderContinuous-v3` + wind/turbulence DR | Box(8) state | Box(2) cont. | 1000 | **wind enabled with per-seed strength** — disjoint OOD held-out wind power |
| 6 | Box2D | `bipedal_hardcore` | `BipedalWalkerHardcore-v3` | Box(24) state | Box(4) cont. | 2000 | **Gymnasium's hardcore variant** (stumps, ladders, pits) — vs the gentler `BipedalWalker-v3` |
| 7 | Box2D | `car_racing` | `CarRacing-v3` + 16×16 downsample | Box(16,16,3) uint8 | Box(3) cont. | 1000 | pixel obs, but downsampled to fit inline — lite color-segmentation task |
| 8 | Box2D | `car_racing_pixel` | `CarRacing-v3` (full) | Box(96,96,3) uint8 | Box(3) cont. | 1000 | **full-resolution** pixel obs via `observations.npy` side-car (SPEC §4.6) |
| 9 | MuJoCo | `half_cheetah` | `HalfCheetah-v5` | Box(17) state | Box(6) cont. | 1000 | canonical locomotion — periodic gait required, no balance constraint |
| 10 | MuJoCo | `hopper` | `Hopper-v5` | Box(11) state | Box(3) cont. | 1000 | **single-leg balance** — fall terminates; harder than `half_cheetah` |
| 11 | MuJoCo | `walker2d` | `Walker2d-v5` | Box(17) state | Box(6) cont. | 1000 | **bipedal walking** — fall terminates; alternating gait needed |
| 12 | MuJoCo | `ant` | `Ant-v5` | Box(105) state | Box(8) cont. | 1000 | **highest dim obs in the suite** (105-D), 4-legged locomotion |
| 13 | MiniGrid | `minigrid_doorkey` | `MiniGrid-DoorKey-16x16-v0` | Box(148) uint8 flat | Discrete(7) | 2560 | 16×16 large grid — exploration + key+door manipulation |
| 14 | MiniGrid | `minigrid_keycorridor` | `MiniGrid-KeyCorridorS6R3-v0` | Box(148) uint8 flat | Discrete(7) | 1080 | corridor maze with hidden key — POMDP memory required |
| 15 | MiniGrid | `minigrid_lavacrossing` | `MiniGrid-LavaCrossingS11N5-v0` | Box(148) uint8 flat | Discrete(7) | 880 | **fatal hazard** (lava terminates) — 5 lava strips to cross |
| 16 | MiniGrid | `minigrid_obstructedmaze` | `MiniGrid-ObstructedMaze-2Dlhb-v1` | Box(148) uint8 flat | Discrete(7) | 2304 | **hardest MiniGrid variant** — doors blocked by boxes (multi-step manipulation: drop key, push box, retrieve key, unlock) |

---

## Comparison with the `hlbench` reference table

`hlbench`'s 16-scenario table (ICLR 2026 sub) uses base Gymnasium
envs throughout. We substitute as follows:

| Slot | `hlbench` env | Our v1 env | Why we differ |
|---|---|---|---|
| Classic 1 | `cartpole_balance` (CartPole-v1) | `cartpole_balance` (CartPole-v1) | same — kept as the EASY anchor for score-distribution diagnostics |
| Classic 2 | `mountain_car` (MountainCar-v0 discrete) | `mountain_car_continuous` | continuous variant is provably harder (sparse + no boost on near-goal) |
| Classic 3 | `acrobot_swingup` | `acrobot` | same env, different naming |
| Classic 4 | `pendulum_swingup` | `pendulum` | same env, different naming. (Note: `pendulum_hardcore`, the DR variant, lives in "additional registered envs" below as a v2 extension demo.) |
| Box2D 1 | `lunar_lander` (LunarLander-v3 discrete) | `lunar_hardcore` | continuous + wind disturbance |
| Box2D 2 | `lunar_lander_continuous` | `bipedal_hardcore` | **swap to Hardcore terrain** (Gymnasium ships this variant) |
| Box2D 3 | `bipedal_walker` (BipedalWalker-v3) | `car_racing_pixel` | **full 96×96 pixel obs** (tests `observations.npy` infra) |
| Box2D 4 | `car_racing` (CarRacing-v3) | `car_racing` (downsampled) | lite version for inline-obs comparison; full version is row 8 |
| MuJoCo 1 | `reacher` (Reacher-v5) | `half_cheetah` | reacher is too easy; HalfCheetah is the canonical locomotion baseline |
| MuJoCo 2 | `inverted_pendulum` (InvertedPendulum-v5) | `hopper` | InvertedPendulum is trivial; Hopper requires single-leg balance |
| MuJoCo 3 | `hopper` | `walker2d` | added Walker2d (bipedal walking, harder than Hopper) |
| MuJoCo 4 | `half_cheetah` | `ant` | added Ant (105-D obs, hardest in MuJoCo suite for hand-crafted policies) |
| MiniGrid 1 | `minigrid_doorkey_16x16` | `minigrid_doorkey` | same |
| MiniGrid 2 | `minigrid_keycorridor_s6r3` | `minigrid_keycorridor` | same |
| MiniGrid 3 | `minigrid_obstructedmaze_2dlhb` (v1) | `minigrid_obstructedmaze` (v1) | same — we match the v1 update from v0 |
| MiniGrid 4 | `minigrid_lavacrossing_s11n5` | `minigrid_lavacrossing` | same |

**Net direction**: our suite preserves a deliberate **easy/medium/
medium-hard/hard difficulty spread** in Classic Control, while going
strictly harder than `hlbench` on the other three categories
(Hardcore Box2D, heavier MuJoCo, hardest MiniGrid variants where
applicable). The two new variants (Hardcore Box2D, full-pixel
CarRacing) also test protocol mechanisms (DR wrapper, external obs
storage) that `hlbench`'s base envs do not exercise.

---

## Per-environment detail

The complete spec — train/heldout pool composition, expert/random
baselines, anti-cheat allow-list, success criteria — lives in each
env's `TASK.md`. Below we surface the fields most relevant to paper
analysis: baselines (for normalized score interpretation), held-out
generalization design, and the policy-synthesis bottleneck claim.

### 1. `pendulum` — Pendulum-v1 baseline

- **Baselines**: random ≈ −1200; expert (PD/LQR) ≈ −150.
- **Held-out**: same env config; 256 random seeds for initial-state
  diversity. Generalization claim: "policy works on unseen initial
  conditions of the same physics".
- **Bottleneck**: textbook PD reaches ~98.3 final_score. *Iteration
  is not load-bearing here*; kept as the calibration baseline.

### 2. `pendulum_hardcore` — Domain-randomized Pendulum

- **Baselines**: random ≈ −1700; adaptive expert ≈ −200.
- **Held-out**: train pool draws `(mass, length, gravity)` from
  nominal ranges `[0.5,2.0] × [0.7,1.5] × [8,12]`; held-out from
  **disjoint OOD** `[2.0,3.5] × [1.5,2.2] × [4,8]`. A fixed-gain PD
  that wins train provably fails held-out.
- **Bottleneck**: must read `env_meta` ranges and write gain-scheduled
  or adaptive controller; naive PD demonstrates the in-loop vs
  held-out gap that justifies the paper's headline finding.

### 3. `acrobot` — Acrobot-v1

- **Baselines**: random ≈ −500 (never reaches target); expert ≈ −80.
- **Held-out**: 256 random initial conditions.
- **Bottleneck**: under-actuated swing-up requires energy pumping
  + switching control law near target — non-trivial code.

### 4. `mountain_car_continuous` — MountainCarContinuous-v0

- **Baselines**: random ≈ −50 (never reaches goal under sparse
  reward); expert ≈ +90.
- **Held-out**: 256 random initial positions.
- **Bottleneck**: **sparse-reward exploration** — naive
  random/greedy never solves; bang-bang energy-building strategy
  required.

### 5. `lunar_hardcore` — Wind-disturbed lander

- **Baselines**: random ≈ −200; adaptive expert ≈ +150.
- **Held-out**: train `wind_power ∈ [10,15], turbulence ∈ [1.0,1.5]`;
  held-out **disjoint OOD** `[15,20] × [1.5,2.0]`. Naive PID tuned
  in calm wind collapses under held-out turbulence.

### 6. `bipedal_hardcore` — BipedalWalkerHardcore-v3

- **Baselines**: random ≈ −100; expert (PPO-trained, 50M+ steps)
  ≈ +300. Hand-crafted policies rarely clear; expected LLM ceiling
  is far below expert.
- **Held-out**: 256 procedural terrain layouts (stumps + ladders +
  pits sampled by env seed).
- **Bottleneck**: **failure-mode discrimination** is the key
  signal here — different LLMs fall at different obstacles, even
  if all score near zero.

### 7. `car_racing` — Downsampled CarRacing

- **Baselines**: random ≈ −100; classical color-seg + PD ≈ +200–400.
- **Held-out**: 256 procedural tracks.
- **Bottleneck**: 16×16 resolution is enough for color-based road
  following; iteration teaches the agent to handle sharp turns
  and off-track recovery.

### 8. `car_racing_pixel` — Full CarRacing

- **Baselines**: random ≈ −100; RL-trained expert ≈ +900.
- **Mechanism**: `obs_storage="external"` — per-step 96×96×3 frames
  written to `observations.npy` (SPEC §4.6).
- **Bottleneck**: full resolution enables horizon estimation (look
  20 cells ahead) that the lite variant can't support.

### 9–12. MuJoCo locomotion suite

| Env | Obs / Action | Random | Expert (~) | Distinct ask |
|---|---|---|---|---|
| `half_cheetah` | (17) / (6) | −300 | +7000 | open-loop CPG works partially |
| `hopper` | (11) / (3) | +5 | +3500 | single-leg balance, falls terminate |
| `walker2d` | (17) / (6) | +5 | +4500 | bipedal alternating gait, falls terminate |
| `ant` | (105) / (8) | −50 | +6000 | 4-legged trotting, highest obs dim |

Held-out across all four: 256 random initial-state perturbations.
Per-seed parameter randomization is **not** applied (deferred); held-out
generalization here is "robustness to init-state distribution," not
parameter OOD.

### 13–16. MiniGrid POMDP suite

All four MiniGrid envs share the obs wrapper: MiniGrid's native Dict
obs `{image: 7×7×3, direction: int, mission: str}` is flattened to
`Box(148,) uint8 = image.flatten() + [direction]`. Mission text is
static per env (in TASK.md). Action space is the standard 7-action
MiniGrid set.

| Env | Map size | Max steps | Distinct ask |
|---|---|---|---|
| `minigrid_doorkey` | 16×16 | 2560 | locate key, unlock door, reach goal |
| `minigrid_keycorridor` | 3-room corridor × 2 | 1080 | search rooms, find hidden key, retrieve target ball |
| `minigrid_lavacrossing` | 11×11 with 5 lava strips | 880 | thread the gaps without stepping on lava (terminates) |
| `minigrid_obstructedmaze` | maze + locked doors blocked by boxes | 2304 | **multi-step manipulation**: drop key, push box, pick up key, unlock — hardest MiniGrid variant |

**Reward**: sparse positive (+1 × discount factor based on episode
length) on success, 0 on timeout/hazard. Random policy expected ≈ 0;
expert ≈ 0.9.

Held-out across all four: 256 random procedural seeds (maze layout,
object positions). Generalization claim: "policy works on unseen
maze structures of the same procedural class."

---

## Acceptance / verification results

To validate the v1 scored suite end-to-end, we ran the full 16-env
matrix on **Claude Sonnet 4.6** under the standard configuration:

- `budget = 256` episodes/env
- `max-turns = 256` (uncapped — the agent decides when to stop)
- `turn-timeout = 1800s`
- `max-parallel = 4` (4 envs concurrently)
- exp-id `v1paper-sonnet-b256-20260530-0154`

Total wall: ~3h11m on a single MacBook Pro (M-series). Per-env
results below; the per-env `run.json` is preserved at
`runs/claude-code-auto/<env>/v1paper-sonnet-b256-20260530-0154/`.

| # | Category | Env | `final_score` | held-out mean return | n_submits |
|---|---|---|---:|---:|---:|
| 1 | Classic | `cartpole_balance` | **100.00** | 500.00 | 2 |
| 2 | Classic | `pendulum` | **101.07** | -138.77 | 14 |
| 3 | Classic | `acrobot` | **95.23** | -100.02 | 15 |
| 4 | Classic | `mountain_car_continuous` | **104.52** | 99.34 | 17 |
| 5 | Box2D | `lunar_hardcore` | **120.00** | 231.81 | 18 |
| 6 | Box2D | `bipedal_hardcore` | **0.00** | -101.94 | 18 |
| 7 | Box2D | `car_racing` | **85.04** | 410.24 | 22 |
| 8 | Box2D | `car_racing_pixel` | **58.11** | 481.10 | 15 |
| 9 | MuJoCo | `half_cheetah` | **14.59** | 764.87 | 23 |
| 10 | MuJoCo | `hopper` | **30.01** | 1053.97 | 103 |
| 11 | MuJoCo | `walker2d` | **20.16** | 911.27 | 34 |
| 12 | MuJoCo | `ant` | **17.01** | 979.12 | 31 |
| 13 | MiniGrid | `minigrid_doorkey` | **103.33** | 0.98 | 5 |
| 14 | MiniGrid | `minigrid_keycorridor` | **103.64** | 0.93 | 8 |
| 15 | MiniGrid | `minigrid_lavacrossing` | **106.48** | 0.96 | 3 |
| 16 | MiniGrid | `minigrid_obstructedmaze` | **90.71** | 0.82 | 21 |
| | | **mean (all 16)** | **71.87** | | |

### Per-category aggregate

| Category | Mean `final_score` (4 envs) | Spread (max − min) |
|---|---:|---:|
| Classic Control | **100.21** | 9.29 (95.23 → 104.52) |
| Box2D | **65.79** | 120.00 (0.00 → 120.00) |
| MuJoCo | **20.44** | 15.42 (14.59 → 30.01) |
| MiniGrid | **101.04** | 15.77 (90.71 → 106.48) |

### Capability cartography

The 16-env scores form a **clear three-tier distribution** that maps
to recognizable LLM capability axes:

| Tier | Range | Count | Envs | What it tests |
|---|---|---:|---|---|
| 🏆 **Ceiling** | ≥ 90 | 9 | cartpole 100 / pendulum 101 / acrobot 95 / mountain_car 105 / lunar_hardcore 120 / minigrid all 4 (91-106) | textbook control + symbolic POMDP — solved by code recall |
| 🌫️ **Mid** | 50–89 | 2 | car_racing 85 / car_racing_pixel 58 | visual reasoning at limited resolution |
| ⬇️ **Floor** | < 35 | 5 | bipedal_hardcore 0 / half_cheetah 15 / walker2d 20 / ant 17 / hopper 30 | fine motor control synthesis (MuJoCo locomotion + Box2D bipedal) |

**Headline finding (paper Section 6 / Section 7.1)**:
Sonnet at budget=256 achieves a **bimodal score distribution** — it
reaches the per-env ceiling on 9 of 16 envs via code recall of
textbook controllers (PD, energy shaping, BFS planning), but cannot
synthesize working code for fine motor control on any of the 4
MuJoCo locomotion envs (mean = 20.4) or for BipedalWalkerHardcore
(0.0). The visual envs (car_racing variants) sit in the middle,
with full-resolution pixel input penalizing the score by ~27 points
relative to the 16×16 downsampled lite variant.

### Notes on calibration

A few baselines need refinement before paper submission:

- **`lunar_hardcore` clipped at 120**: the score reached the
  `clip(0, 1.2) × 100` upper bound, suggesting `expert_baseline`
  is conservative. Suggest re-calibrating to a stronger expert
  (estimated ≈ +250 raw return) before paper publication.
- **`mountain_car_continuous` slightly above 100 (104.5)**: also
  hints at conservative `expert_baseline`. Less critical since not
  clipped.
- **`hopper` n_submits = 103**: agent retried unusually many times.
  Worth inspecting the trace for "stuck in local optimum" dynamics
  vs natural exploration.
- **`car_racing` regression (b64 = 106.7 → b256 = 85.0)**: more
  budget produced a *worse* score on the lite variant. Possible
  over-iteration / held-out variance. Worth investigating before
  publication.

These are calibration items, not protocol bugs — the suite as a
whole is performing as designed (bimodal distribution with clear
capability axes).

---

## Additional registered envs (not in v1 scored suite)

The benchmark registry also contains **7 additional envs** not part of
this v1 paper table. They are available via the same protocol but are
not scored in the headline `final_score` matrix:

| Env | Why excluded from v1 paper table | Use |
|---|---|---|
| `pendulum_hardcore` | superseded by `pendulum` as the Classic Control swing-up slot; DR is the v2 extension story | **DR demo** for v2 paper / robustness ablation |
| `bipedal_walker` (base) | superseded by `bipedal_hardcore` | regression test |
| `lunar_lander_continuous` (base) | superseded by `lunar_hardcore` | regression test |
| `cache_replacement` | not in 4-Gymnasium-category structure | "online algorithm" extension (v2 paper) |
| `k_server` | not in 4-Gymnasium-category structure | online algorithm extension |
| `online_bipartite_matching` | not in 4-Gymnasium-category structure | online algorithm extension |
| `pendulum_from_pixels` | overlaps `car_racing_pixel` for the "visual extraction" capability axis | calibration / `observations.npy` test |

These will be reported in v2 as an "extensions beyond Gymnasium"
ablation showing the protocol generalizes to non-RL task families
(online algorithms) and visual variants of state-based envs.

---

## Library version compatibility

| Dependency | Required version | Used by |
|---|---|---|
| `gymnasium[box2d]` | ≥ 0.29 (we tested 0.29–1.x) | all envs |
| `mujoco` | ≥ 3.1 (optional extra) | envs 9–12 |
| `minigrid` | ≥ 2.3 | envs 13–16 |
| `numpy` | ≥ 1.26 | all |
| Python | ≥ 3.12 | all |

**Latest-API check**: we use the v5 MuJoCo suite (vs. older v4/v2),
v3 Box2D (latest), and v1 MiniGrid where available. The one
env that historically used v0 — `MiniGrid-ObstructedMaze-2Dlhb-v0`
— is updated to v1 in row 16 to match `hlbench`'s table and avoid
the Gymnasium deprecation warning.

## Implementation status

All 16 envs are landed in the open-source v0.1 release
(`docs/envs.md` for the per-env architectural rationale,
`CHANGELOG.md > [Unreleased]` for the per-batch landing notes).
Tests: 238 passing under `pytest -q + ruff + mypy --strict`. Pixel
env end-to-end is verified via `tests/test_visual_envs.py::
test_pendulum_from_pixels_e2e_writes_observations_npy` (real
Sandbox-driven submit producing `observations.npy`).

Frontier-model evaluation matrix (5 models × 16 envs) is the v1.0
paper deliverable; v0.1 paper claims protocol + roster only.
