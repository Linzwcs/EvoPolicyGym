# Evidence ledger

| Candidate | Public Episodes | Wins | Mean Blinds | Validation | Notes |
|---|---:|---:|---:|---:|---|
| `sha256:60862b...` | 20 | 0 | 8.55 | 6.75 (8 eps) | Best public mean among historical digests |
| `sha256:8deda60...` | 80 | 0 | 6.23 | 6.75 (8 eps) | Validation-selected submission 10 |
| `sha256:8ac1c1b...` | 68 | 0 | 7.37 | 6.00 (8 eps) | Current copied baseline; voucher purchase removed |
| `sha256:c37a548...` | 32 paired | 0 | 6.84 | validation seeds 20260726–27 | H1 replacement candidate; paired baseline 6.66, delta +0.19 |
| `sha256:ceaeed...` | 32 paired | 0 | 7.41 | validation seeds 20260726–27 | H2 economy candidate; paired baseline 6.66, delta +0.75 |
| `sha256:d4e559...` | 32 paired | 1 | 8.88 | validation seeds 20260726–27 | Final candidate; 3.125% wins, no Policy failures |
| `sha256:598424...` | 32 paired | 1 | 8.88 | validation seeds 20260726–27 | H8 Boss constraints + safe Planet use; unchanged score, no Policy failures |
| `sha256:69b42d...` | 16 smoke | 1 | 9.75 | validation seeds 20260726–27 | H12 visible scoring calibration; retained win, no Policy failures |
| `sha256:93e2e3...` | 16 smoke | 1 | 9.81 | validation seeds 20260726–27 | H13 round-history scoring; retained win, one Episode +1 Blind |
| `sha256:9c8aca...` | 16 smoke | 1 | 9.81 | validation seeds 20260726–27 | H14 sell-value scoring calibration; behavior-neutral |
| `sha256:f0191b...` | 32 paired | 1 | 9.72 | validation seeds 20260726–27 | H16 consecutive-discard stop; +0.84 mean Blinds over H8 |
| `sha256:5095ed...` | 32 paired | 1 | 9.72 | validation seeds 20260726–27 | H17 mature direct-Planet purchase; behavior-neutral |
| `sha256:5095ed...` | 64 unseen validation | 1 | 9.34 | validation seeds 20260802–05 | frozen confirmation; median 10, early rate 26.6% |
| `sha256:5095ed...` | 64 final test | 1 | 10.84 | test seed 20260901 | frozen final result; median 11, early rate 12.5% |
| `sha256:981dd7...` | 64 unseen validation | 0 | 9.84 | validation seeds 20261101–04 | H19 gate; median 11, early rate 20.3%; final test not reused |
| `sha256:1f9468...` | 64 unseen validation | 1 | 10.33 | validation seeds 20261401–04 | H21 gate; ≥18 in 6 Episodes, final test not reused |
| `sha256:da7898...` | 64 unseen validation | 1 | 9.81 | validation seeds 20261601–04 | H22 Needle gate; ≥18 in 6 Episodes, final test not reused |
| `sha256:8d2526...` | 64 paired unseen validation | 2 | 10.70 | validation seeds 20261801–04 | H24 hidden-card discard gate; paired H22 mean 10.63, no seed regressed |
| `sha256:afe6da...` | 64 paired unseen validation | 0 | 10.45 | validation seeds 20262001–04 | H25 continuation gate; paired H24 mean 10.23, early deaths 15→14 |
| `sha256:c61ac8...` | 64 paired unseen validation | 2 | 9.89 | validation seeds 20262201–04 | H26 Joker order gate; paired H25 mean 9.78, ≥18 improved 3→5 |

Public means pool identical Program digests across submissions. Validation values
come from `validation/report.json`; Validation does not publish replay evidence.

## Experiment queue

| ID | Capability | Hypothesis | Status |
|---|---|---|---|
| H0 | Replay diagnosis | High-cash shop exits and weak full slots explain a material share of Ante 3–4 deaths | confirmed: 22/38 exits at $15+ with affordable inventory |
| H1 | Effect roles + replacement | Replacing low marginal-value Jokers restores mid-game growth | measured: +0.19 mean Blind over 32 paired Episodes; 5 better / 3 worse / 24 equal; 0 wins |
| H2 | Booster/voucher/reroll budget | Spending above the interest reserve raises Ante reach without destabilizing early survival | confirmed: +0.75 mean Blind over 32 paired Episodes |
| H3 | Consumable targeting | Legal targeted Tarot/Planet use improves build consistency | pending |
| H4 | Unified hand scoring | Conditional Joker activation prevents late weak High Card plays | confirmed: replay MAPE 0.890 → 0.156; 33.5% → 79.5% within ±25% |
| H5 | Draw safety margin | Accurate scores need an explicit reason to spend early discards before hands | confirmed: H4c reaches 8.72 mean Blinds |
| H6 | Copy build and ordering | Buying and dynamically positioning a visible copy Joker raises the late-game ceiling | confirmed: converted an Ante 7 loss into an Ante 8 win |
| H8 | Boss constraints + Planet | Public Boss rules prevent legal-but-zero plays; zero-target Planet use adds safe growth | measured: unchanged 1/32 wins and 8.875 mean Blinds; no Planet consumable appeared in this schedule |
| H9 | Skip Tags | Free Pack/edition/hand-size Tags should repay one skipped Blind | rejected: 3-episode smoke regressed seed26 12.00→7.67 and seed27 5.00→2.67; reverted |
| H10 | Joker role completion | Narrow XMult/retrigger/copy bonuses improve builds without disrupting stable purchase paths | rejected: retained win but pooled mean Blinds regressed 8.94→8.38; reverted |
| H11 | Straight-draw discard | Preserving four-card straight draws improves early hand conversion | rejected: seed26 8-Episode score fell to 7.125 and known win disappeared; reverted |
| H12 | Visible scoring calibration | Modeling public Stuntman, Bootstraps, and Raised Fist effects improves hand selection | smoke accepted: mean Blinds 8.94→9.75 over the same 16 Episodes; retained known win |
| H13 | Round-history scoring | Supernova and Card Sharp can be scored from public run/round history | smoke accepted: mean Blinds 9.75→9.81; retained win; Seltzer sub-experiment was behavior-neutral and reverted after MAPE worsened |
| H14 | Sell-value scoring | Swashbuckler can use other Jokers' public sell values instead of its placeholder mult | calibration accepted: removed its systematic underprediction; fixed-seed behavior unchanged |
| H15 | Early survival threshold | Empty Ante 1–2 builds should play near-sufficient hands instead of spending all discards | rejected: 0.90 moved one Episode 0→13 but another 8→0; 0.95 kept the regression and lost the gain; reverted |
| H16 | Consecutive-discard stop | Apply the early survival threshold only after a failed discard | confirmed: retained every seed26 smoke result, moved one seed27 Episode 0→13; 32-Episode mean Blinds 8.88→9.72 |
| H17 | Direct Planet purchase | Buy shop Planets for an established hand without disrupting economy | conservative variant accepted: 32-Episode behavior unchanged; broad variant caused two -1 Blind regressions |
| H18 | Generalization audit | Improvements should survive schedules never used for tuning | confirmed: +2.89 mean Blinds over 64 unseen validation Episodes and +4.20 over 64 final test Episodes |
| H19 | Moderate Joker replacement | Lowering the modeled upgrade threshold unlocks full-slot mid-game growth | accepted: same-seed train A/B 11.05→11.64 mean Blinds; all four train seeds improved; unseen validation gate mean 9.84 |
| H20 | Recent-hand replacement simulation | Use the last public scoring hand to choose which Joker to replace | rejected: train A/B 10.94→11.19 but only 2/4 seeds improved, one regressed; complexity not justified |
| H21 | Surplus paid reroll | Full-slot high-cash shops should search once more when no purchase qualifies | accepted: train A/B 3/4 seeds improved and one unchanged; wins 1→2; unseen validation mean 10.33 with 1 win |
| H22 | The Needle discard planning | A one-hand Boss should spend discards before committing its only hand | accepted: train A/B 3/4 seeds improved, one unchanged; mean 9.78→10.00 and early deaths 17→15 |
| H23 | The Flint score calibration | Halving only base Chips/Mult should improve Flint hand selection | rejected: exact model changed a myopic selector adversely; train mean 10.34→10.28, two seeds regressed; reverted |
| H24 | Hidden-card discard grouping | Face-down cards must not look like a same-rank or same-suit draw | accepted: train mean 10.34→10.63 with two targeted gains; paired unseen validation 10.63→10.70, no regressions |
| H25 | Safe play-throughput | Nonlethal hands should cycle safe kickers when future hands can use the extra draws | accepted narrow variant: Ante 2+, no discards; train 10.84→11.36, paired unseen validation 10.23→10.45 |
| H26 | Main Joker scoring order | Main-stage additive Mult should resolve before main-stage XMult | accepted: train wins 1→2; paired unseen validation mean 9.78→9.89 and ≥18 improved 3→5 |
| H27 | Brainstorm target arrangement | A single Brainstorm should copy the best visible-hand target at the left edge | rejected: train 10.48→10.52, but paired unseen validation regressed 10.64→10.61 and ≥12 fell 24→23 |

## H1 paired evaluation

Both candidates ran on identical public Benchmark schedules with
`split=validation`, no Policy failures, and no wins.

| Seed | Episodes | Baseline | H1 | Delta | H1 sell actions |
|---:|---:|---:|---:|---:|---:|
| 20260726 | 16 | 6.4375 | 6.6250 | +0.1875 | 12 |
| 20260727 | 16 | 6.8750 | 7.0625 | +0.1875 | 14 |
| pooled | 32 | 6.6563 | 6.8438 | +0.1875 | 26 |

Across the pooled paired Episodes, H1 improved 5, worsened 3, and left 24
unchanged. The effect is directionally reproducible but small, and the absence
of wins means it is not yet evidence of improved Ante 8 completion.

Artifacts:

- `runs/balatro-human-loop-p1-paired16-seed20260726/`
- `runs/balatro-human-loop-p1-paired16-seed20260727/`

## H2 paired evaluation

H2 kept baseline open-slot acquisition, added conservative high-value Vouchers,
one Celestial pack per shop, and free rerolls only.

| Seed | Episodes | Baseline | H2 | Delta |
|---:|---:|---:|---:|---:|
| 20260726 | 16 | 6.4375 | 7.4375 | +1.0000 |
| 20260727 | 16 | 6.8750 | 7.3750 | +0.5000 |
| pooled | 32 | 6.6563 | 7.4063 | +0.7500 |

Artifacts:

- `runs/balatro-human-loop-h2b-paired16-seed20260726/`
- `runs/balatro-human-loop-h2b-paired16-seed20260727/`

## Final paired evaluation

Both Programs used the same public deterministic schedules. The final candidate
had no Policy failures. Mean Blinds is reported separately because a win has
Run score 1024.

| Seed | Episodes | Baseline Run score | Final Run score | Final mean Blinds | Wins |
|---:|---:|---:|---:|---:|---:|
| 20260726 | 16 | 6.4375 | 72.3750 | 9.8750 | 1 |
| 20260727 | 16 | 6.8750 | 7.8750 | 7.8750 | 0 |
| pooled | 32 | 6.6563 | 40.1250 | 8.8750 | 1 |

The winning Episode cleared all 24 required Blinds, ended with
`progress.won=True` at Ante 9, and scored 139,104 against the Ante 8 Boss target
of 100,000. The last hand scored 50,400. Its final build was Half Joker,
Loyalty Card, Hack, Blueprint, and Blackboard; visible-hand simulation moved
Blueprint immediately left of Blackboard whenever the all-black held-card
condition made copying X3 stronger than copying Half Joker.

Artifacts:

- `runs/balatro-human-loop-final-paired16-seed20260726/`
- `runs/balatro-human-loop-final-paired16-seed20260727/`

## H8 paired evaluation

H8 added public-text Boss constraints for The Psychic, The Eye, The Mouth, and
face-down first hands. It also uses a Planet consumable only when the current
`use_consumable` descriptor explicitly admits a zero-target action matching the
Episode-local primary hand. Targeted consumables remain untouched.

| Seed | Episodes | Baseline | H8 | Delta | H8 wins |
|---:|---:|---:|---:|---:|---:|
| 20260726 | 16 | 6.4375 | 72.3750 | +65.9375 | 1 |
| 20260727 | 16 | 6.8750 | 7.8750 | +1.0000 | 0 |
| pooled | 32 | 6.6563 | 40.1250 | +33.4688 | 1 |

The large pooled Run score is caused by the 1024-point complete-run bonus. Mean
Blinds remains 8.875, and the change introduced no Policy failure.

Artifacts:

- `runs/balatro-human-loop-boss2-paired16-seed20260726/`
- `runs/balatro-human-loop-boss2-paired16-seed20260727/`

## H10 rejected smoke evaluation

Replacing open-slot acquisition entirely with modeled Joker value removed the
known seed26 win, so that broad change was rejected. The retained variant keeps
the prior acquisition ordering and adds narrow bonuses only for a missing
XMult, retrigger, or copy role. It retained the known win but still reduced
pooled mean Blinds from about 8.94 to 8.38, so it was reverted. The speculative
mandatory Joker/Boss cash reserve was also reverted after it removed the known
win.

| Seed | Episodes | Candidate Run score | Mean Blinds | Wins |
|---:|---:|---:|---:|---:|
| 20260726 | 8 | 134.750 | 9.750 | 1 |
| 20260727 | 8 | 7.000 | 7.000 | 0 |
| pooled | 16 | 70.875 | 8.375 | 1 |

Artifacts:

- `runs/balatro-human-loop-joker-blend-norestrict-smoke8-seed20260726/`
- `runs/balatro-human-loop-joker-blend-norestrict-smoke8-seed20260727/`

## H12 smoke evaluation

H12 uses only current public fields: Stuntman's nested `chip_mod`, Bootstraps'
visible money buckets, and Raised Fist's lowest visible held rank. On the
existing 32-Episode replay corpus, High Card MAPE fell from 20.2% to 12.2% and
Two Pair MAPE fell from 19.0% to 13.9%.

| Seed | Episodes | Candidate Run score | Candidate mean Blinds | Wins |
|---:|---:|---:|---:|---:|
| 20260726 | 8 | 137.750 | 12.750 | 1 |
| 20260727 | 8 | 6.750 | 6.750 | 0 |
| pooled | 16 | 72.250 | 9.750 | 1 |

Artifacts:

- `runs/balatro-human-loop-score-calibration-smoke8-seed20260726/`
- `runs/balatro-human-loop-score-calibration-smoke8-seed20260727/`

## H13 smoke evaluation

Supernova now uses the public per-hand `played` count including the pending
play. Card Sharp uses Episode-local same-round hand-type counts, reset at each
Blind. A separate Seltzer retrigger model changed no action and worsened its
replay MAPE, so that part was reverted.

| Seed | Episodes | Candidate Run score | Candidate mean Blinds | Wins |
|---:|---:|---:|---:|---:|
| 20260726 | 8 | 137.750 | 12.750 | 1 |
| 20260727 | 8 | 6.875 | 6.875 | 0 |
| pooled | 16 | 72.3125 | 9.8125 | 1 |

Artifacts:

- `runs/balatro-human-loop-card-sharp-smoke8-seed20260726/`
- `runs/balatro-human-loop-card-sharp-smoke8-seed20260727/`

## H14 smoke evaluation

Swashbuckler now sums the visible `sell_value` of every other Joker. Its old
placeholder `mult=1` is no longer double-counted. Replay calibration improved,
while both fixed-seed schedules remained identical to H13.

Artifacts:

- `runs/balatro-human-loop-swashbuckler-smoke8-seed20260726/`
- `runs/balatro-human-loop-swashbuckler-smoke8-seed20260727/`

## H16 paired evaluation

H16 tracks consecutive discards within the current Blind. Only after one
discard, at Ante 1–2 with no Joker, does it play a hand whose projected
remaining-hand total covers at least 90% of the target gap. Playing a hand or
starting a new Blind resets the streak.

| Seed | Episodes | Candidate Run score | Candidate mean Blinds | Wins |
|---:|---:|---:|---:|---:|
| 20260726 | 16 | 73.1875 | 10.6875 | 1 |
| 20260727 | 16 | 8.7500 | 8.7500 | 0 |
| pooled | 32 | 40.9688 | 9.7188 | 1 |

Compared with H8 on the same schedules, mean Blinds rose from 8.8750 to
9.7188. The count of Episodes ending at five or fewer Blinds fell from eight
to six. No Policy failure occurred.

Artifacts:

- `runs/balatro-human-loop-discard-stop-paired16-seed20260726/`
- `runs/balatro-human-loop-discard-stop-paired16-seed20260727/`

## H17 paired evaluation

The broad direct-Planet experiment caused two one-Blind regressions. The
retained variant buys at most one matching Planet per shop only at Ante 3+,
with a free consumable slot, five dollars above the normal reserve, and a
target hand played at least five times or already at Level 2. It produced the
same 32-Episode results as H16 and introduced no Policy failure.

Artifacts:

- `runs/balatro-human-loop-direct-planet-mature-paired16-seed20260726/`
- `runs/balatro-human-loop-direct-planet-mature-paired16-seed20260727/`

## H18 frozen generalization audit

Seeds 20260726–27 are development schedules and are no longer treated as
independent evidence. The Program was frozen at
`sha256:5095edea0571433beebe58a8bd3f3f24238fdf6e2a73754e99a3cf6e8abd74a8`
before the following evaluations. No unseen replay trajectory was used to tune
the Policy.

| Dataset | Episodes | Baseline mean Blind | Candidate mean Blind | Baseline / candidate median | Baseline / candidate early ≤5 | Candidate wins |
|---|---:|---:|---:|---:|---:|---:|
| unseen validation, seeds 20260802–05 | 64 | 6.4531 | 9.3438 | 7 / 10 | 45.3% / 26.6% | 1 |
| final test, seed 20260901 | 64 | 6.6406 | 10.8438 | 7 / 11 | 40.6% / 12.5% | 1 |

The test Run score is 26.4688 because one completed run contributes the
Benchmark's 1024 reward. Mean Blind replaces that reward with 24 cleared
Blinds, preventing the win bonus from distorting the main progress metric.

Artifacts:

- `runs/balatro-human-loop-unseen-validation16-seed20260802/`
- `runs/balatro-human-loop-unseen-validation16-seed20260803/`
- `runs/balatro-human-loop-unseen-validation16-seed20260804/`
- `runs/balatro-human-loop-unseen-validation16-seed20260805/`
- `runs/balatro-human-loop-final-test64-seed20260901/`

## H19 isolated development and gate

The final test above remains sealed and was not reused. Train seeds
20261001–04 diagnosed the problem: 53 of 57 observed mid-game terminal builds
had full Joker slots and median cash of 33. H19 changes only the modeled
replacement threshold from `max(6, 30%)` to `max(4, 20%)`.

On fresh train seeds 20261011–14, H19 and H17 were run on the same 64
Episodes:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 |
|---|---:|---:|---:|---:|---:|
| H17 control | 11.0469 | 11 | 7 | 29 | 2 |
| H19 | 11.6406 | 11 | 7 | 31 | 4 |

Every train seed improved. H19 then passed a one-way gate on unseen validation
seeds 20261101–04:

| Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Policy failures |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9.8438 | 11 | 13 (20.3%) | 20 | 3 | 0 |

Artifacts:

- `runs/balatro-human-loop-h19-train16-seed20261011/`
- `runs/balatro-human-loop-h19-train16-seed20261012/`
- `runs/balatro-human-loop-h19-train16-seed20261013/`
- `runs/balatro-human-loop-h19-train16-seed20261014/`
- `runs/balatro-human-loop-h17-control-train16-seed20261011/`
- `runs/balatro-human-loop-h17-control-train16-seed20261012/`
- `runs/balatro-human-loop-h17-control-train16-seed20261013/`
- `runs/balatro-human-loop-h17-control-train16-seed20261014/`
- `runs/balatro-human-loop-h19-gate16-seed20261101/`
- `runs/balatro-human-loop-h19-gate16-seed20261102/`
- `runs/balatro-human-loop-h19-gate16-seed20261103/`
- `runs/balatro-human-loop-h19-gate16-seed20261104/`

## H21 isolated development and gate

H21 permits one paid reroll per shop only at Ante 3+, with full Joker slots,
after all normal purchases fail, and only when paying the reroll still leaves
the normal reserve plus ten dollars.

On fresh train seeds 20261301–04, H21 and H19 were evaluated on identical
64-Episode schedules:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H19 control | 10.4531 | 11 | 11 | 26 | 2 | 1 |
| H21 | 10.5938 | 11 | 11 | 25 | 3 | 2 |

Three train seeds improved and one was unchanged. On unseen validation seeds
20261401–04, H21 achieved:

| Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 10.3281 | 10 | 14 (21.9%) | 24 | 6 | 1 | 0 |

The previously sealed final test was not rerun or inspected.

Artifacts:

- `runs/balatro-human-loop-h21-train16-seed20261301/`
- `runs/balatro-human-loop-h21-train16-seed20261302/`
- `runs/balatro-human-loop-h21-train16-seed20261303/`
- `runs/balatro-human-loop-h21-train16-seed20261304/`
- `runs/balatro-human-loop-h19-control3-train16-seed20261301/`
- `runs/balatro-human-loop-h19-control3-train16-seed20261302/`
- `runs/balatro-human-loop-h19-control3-train16-seed20261303/`
- `runs/balatro-human-loop-h19-control3-train16-seed20261304/`
- `runs/balatro-human-loop-h21-gate16-seed20261401/`
- `runs/balatro-human-loop-h21-gate16-seed20261402/`
- `runs/balatro-human-loop-h21-gate16-seed20261403/`
- `runs/balatro-human-loop-h21-gate16-seed20261404/`

## H22 isolated development and gate

Train-only Boss diagnosis found The Needle as the most frequent terminal Boss:
six observed failures averaged only 43.9% of its target. The cause was exact:
the generic discard guard required more than one remaining hand, while The
Needle exposes only one. H22 lets that visible rule spend discards before its
single hand; lethal predicted hands are still played immediately.

On fresh train seeds 20261501–04:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H21 control | 9.7812 | 10 | 17 | 21 | 4 | 2 |
| H22 | 10.0000 | 10 | 15 | 22 | 4 | 2 |

Three seeds improved and one was unchanged. On unseen validation seeds
20261601–04, H22 achieved:

| Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9.8125 | 10 | 18 (28.1%) | 17 | 6 | 1 | 0 |

The sealed final test remained untouched.

Artifacts:

- `runs/balatro-human-loop-h22-train16-seed20261501/`
- `runs/balatro-human-loop-h22-train16-seed20261502/`
- `runs/balatro-human-loop-h22-train16-seed20261503/`
- `runs/balatro-human-loop-h22-train16-seed20261504/`
- `runs/balatro-human-loop-h21-control4-train16-seed20261501/`
- `runs/balatro-human-loop-h21-control4-train16-seed20261502/`
- `runs/balatro-human-loop-h21-control4-train16-seed20261503/`
- `runs/balatro-human-loop-h21-control4-train16-seed20261504/`
- `runs/balatro-human-loop-h22-gate16-seed20261601/`
- `runs/balatro-human-loop-h22-gate16-seed20261602/`
- `runs/balatro-human-loop-h22-gate16-seed20261603/`
- `runs/balatro-human-loop-h22-gate16-seed20261604/`

## H23 rejected Flint calibration

H23 reproduced The Flint's public scoring order: halve and round only the
poker-hand base Chips and Mult before playing-card and Joker effects. Although
the formula matched the public breakdown, the current selector optimizes one
hand at a time. On train seeds 20261701–04, the exact estimate sometimes chose
a slightly stronger immediate High Card or spent more discards while reducing
future draw throughput.

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H22 control | 10.3438 | 11 | 6 | 18 | 3 | 0 |
| H23 | 10.2812 | 11 | 6 | 18 | 3 | 0 |

Two seeds were unchanged and two regressed. H23 was reverted without entering
validation. The result supports multi-hand continuation modeling before
reintroducing exact Flint calibration.

Artifacts:

- `runs/balatro-human-loop-h23-train16-seed20261701/`
- `runs/balatro-human-loop-h23-train16-seed20261702/`
- `runs/balatro-human-loop-h23-train16-seed20261703/`
- `runs/balatro-human-loop-h23-train16-seed20261704/`

## H24 hidden-card discard repair and paired gate

After The Fish plays a hand, newly drawn cards expose `rank=None` and
`suit=None`. The old discard grouping converted both values to the same string,
so multiple face-down cards were incorrectly preserved as both a pair and a
flush draw. H24 excludes unknown rank and suit values from grouping; they
remain eligible for discard unless selected for the current best hand.

On the same development schedules used to diagnose the bug:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H22 control | 10.3438 | 11 | 6 | 18 | 3 | 0 |
| H24 | 10.6250 | 11 | 6 | 19 | 4 | 0 |

Two seeds improved and two were unchanged. Only two Episode outcomes changed,
both at The Fish: one moved from 6 to 23 cleared Blinds and one from 8 to 9.

H24 was frozen before a paired gate on unseen validation seeds 20261801–04.
Only aggregate outcomes were inspected:

| Candidate | Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H22 control | 64 | 10.6250 | 11 | 17 | 25 | 10 | 2 | 0 |
| H24 | 64 | 10.7031 | 11 | 17 | 25 | 10 | 2 | 0 |

Three validation seeds were unchanged and one improved; none regressed. The
sealed final test remained untouched.

Artifacts:

- `runs/balatro-human-loop-h24-train16-seed20261701/`
- `runs/balatro-human-loop-h24-train16-seed20261702/`
- `runs/balatro-human-loop-h24-train16-seed20261703/`
- `runs/balatro-human-loop-h24-train16-seed20261704/`
- `runs/balatro-human-loop-h24-gate-paired16-seed20261801/`
- `runs/balatro-human-loop-h24-gate-paired16-seed20261802/`
- `runs/balatro-human-loop-h24-gate-paired16-seed20261803/`
- `runs/balatro-human-loop-h24-gate-paired16-seed20261804/`

## H25 bounded multi-hand continuation

H23 showed that a more accurate immediate-hand score can still lose when the
selector ignores what the play draws for the next hand. H25 adds a bounded
continuation proxy after the discard decision has already been made. For a
nonlethal play with future hands remaining, it may add low, unmodified cards
that are not part of the selected hand, a visible pair, or a four-card flush
draw. The augmented play must retain the same poker-hand type and estimated
score apart from the existing `0.05` non-scoring-card tie-break.

The broad development variant altered 32/64 outcomes. Although mean Blind rose
from 10.84 to 11.45 and wins from two to three, it caused several large early
regressions. Every inspected early regression first diverged while discards
were still available. Requiring zero discards reduced changed outcomes to 16,
but one Ante 1 Episode still moved from five Blinds to one. The retained H25
therefore activates only at Ante 2+ with zero discards.

On train seeds 20261901–04:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H24 control | 10.8438 | 11 | 11 | 23 | 4 | 2 |
| H25 retained | 11.3594 | 11 | 9 | 26 | 5 | 2 |

Three train seeds improved and one was unchanged; only 12/64 Episode outcomes
changed. One H24 win stopped at 23 Blinds while a separate 20-Blind Episode
became a win.

H25 was frozen before the paired validation gate on seeds 20262001–04. The
acceptance criteria were fixed in advance: no Policy failures, no reduction in
wins or mean Blind, and no increase in early deaths. Only aggregate outcomes
were inspected:

| Candidate | Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H24 control | 64 | 10.2344 | 11 | 15 | 27 | 5 | 0 | 0 |
| H25 | 64 | 10.4531 | 11 | 14 | 28 | 4 | 0 | 0 |

Two validation seeds had equal aggregate means, one improved by 1.0 Blind, and
one regressed by 0.125. H25 passed the predefined gate, but the `≥18` count
fell from five to four and neither side won. The evidence therefore supports
average survival and draw efficiency, not a demonstrated validation win-rate
increase. The sealed final test remained untouched.

Artifacts:

- `runs/balatro-human-loop-h25c-train16-seed20261901/`
- `runs/balatro-human-loop-h25c-train16-seed20261902/`
- `runs/balatro-human-loop-h25c-train16-seed20261903/`
- `runs/balatro-human-loop-h25c-train16-seed20261904/`
- `runs/balatro-human-loop-h25-gate-paired16-seed20262001/`
- `runs/balatro-human-loop-h25-gate-paired16-seed20262002/`
- `runs/balatro-human-loop-h25-gate-paired16-seed20262003/`
- `runs/balatro-human-loop-h25-gate-paired16-seed20262004/`

## H26 main-stage Joker ordering

H25 only rearranged Blueprint. A train-only counterfactual found 312 plays
where a main-stage XMult Joker appeared before a later additive-Mult Joker.
Moving only main-stage XMult to the right improved the visible estimate on 110
plays across 12 Episodes, with a median estimated ratio of 1.58. This was a
model-based diagnostic rather than environment evidence.

H26 resolves one visible inversion per observation. It excludes face-card,
scored-card, held-card, and per-card effects because those trigger in a
different scoring phase. It also declines generic ordering whenever a copy
Joker is present, leaving Blueprint behavior unchanged and avoiding unmodeled
Brainstorm chains.

On fresh train seeds 20262101–04:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H25 control | 10.6094 | 11 | 12 | 28 | 5 | 1 |
| H26 | 10.7813 | 11 | 12 | 28 | 5 | 2 |

Three train seeds improved and one was unchanged. Only five Episode outcomes
changed: four improved and one regressed. H26 was then frozen before a paired
validation gate on seeds 20262201–04. Only aggregate outcomes were inspected:

| Candidate | Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H25 control | 64 | 9.7813 | 11 | 14 | 23 | 3 | 2 | 0 |
| H26 | 64 | 9.8906 | 11 | 14 | 23 | 5 | 2 | 0 |

Two validation seeds improved and two were unchanged; none regressed. Only
three of 64 Episode outcomes changed. H26 passed the predefined gate: wins,
early deaths, and Policy failures did not worsen, while mean Blind and the
`≥18` tail improved. The sealed final test remained untouched.

Artifacts:

- `runs/balatro-human-loop-h26-train16-seed20262101/`
- `runs/balatro-human-loop-h26-train16-seed20262102/`
- `runs/balatro-human-loop-h26-train16-seed20262103/`
- `runs/balatro-human-loop-h26-train16-seed20262104/`
- `runs/balatro-human-loop-h26-gate-paired16-seed20262201/`
- `runs/balatro-human-loop-h26-gate-paired16-seed20262202/`
- `runs/balatro-human-loop-h26-gate-paired16-seed20262203/`
- `runs/balatro-human-loop-h26-gate-paired16-seed20262204/`

## H27 rejected Brainstorm arrangement

H27 modeled the public single-Brainstorm rule only when no other copy Joker was
present. Brainstorm copied the leftmost non-self Joker, including the engine's
special case where a leftmost Brainstorm skips itself. Before each hand, the
candidate compared visible-hand estimates and moved the selected target one
step toward the left edge. Blueprint–Brainstorm chains and multiple copy
Jokers were deliberately excluded.

On fresh train seeds 20262301–04:

| Candidate | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins |
|---|---:|---:|---:|---:|---:|---:|
| H26 control | 10.4844 | 11 | 11 | 24 | 2 | 0 |
| H27 | 10.5156 | 11 | 11 | 24 | 2 | 0 |

Only one of 64 train outcomes changed, improving from 14 to 16 Blinds. H27 was
frozen at `sha256:d095312...` before the paired validation gate on seeds
20262401–04. Only aggregate outcomes were inspected:

| Candidate | Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H26 control | 64 | 10.6406 | 11 | 12 | 24 | 5 | 1 | 0 |
| H27 | 64 | 10.6094 | 11 | 12 | 23 | 5 | 1 | 0 |

Three validation seeds were unchanged and one regressed by 0.125 mean reward.
Because both mean Blind and the `≥12` count declined, H27 failed the gate and
was fully reverted. No validation trajectory was inspected to construct a
follow-up patch. H26 remains selected, and the sealed final test remains
untouched.

Artifacts:

- `runs/balatro-human-loop-h27-train16-seed20262301/`
- `runs/balatro-human-loop-h27-train16-seed20262302/`
- `runs/balatro-human-loop-h27-train16-seed20262303/`
- `runs/balatro-human-loop-h27-train16-seed20262304/`
- `runs/balatro-human-loop-h27-gate-paired16-seed20262401/`
- `runs/balatro-human-loop-h27-gate-paired16-seed20262402/`
- `runs/balatro-human-loop-h27-gate-paired16-seed20262403/`
- `runs/balatro-human-loop-h27-gate-paired16-seed20262404/`

## H26 random validation audit

After H27 was rejected, four random evaluation-level master seeds were drawn
before evaluation: 36079239, 67411017, 43177812, and 61585883. None matched an
existing run directory. For each master seed, the Benchmark derived 16 distinct
environment seeds from the split, master seed, and Episode index using its
SHA-256 domain-separated schedule. All 64 environment seeds were distinct. The
selected H26 digest remained
`sha256:c61ac881f7e642f639335c6391698975fdaf8102c68b97dc6495878aee3febe6`.
Each seed evaluated 16 validation Episodes against the original Agent policy
on the identical schedule. The sealed test split was not reused, and no replay
trajectory was inspected.

| Candidate | Episodes | Mean Blind | Median | Early ≤5 | ≥12 | ≥18 | Wins | Policy failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Agent | 64 | 5.7188 | 5 | 34 | 5 | 0 | 0 | 0 |
| H26 | 64 | 10.4219 | 11 | 12 | 27 | 2 | 0 | 0 |

H26 improved 45 paired Episode outcomes, left 16 unchanged, and regressed
three. Its mean paired gain was 4.7031 Blinds. The best run cleared 23 Blinds,
one short of completing Ante 8. Although this batch produced no win, its mean,
median, and early-death rate are consistent with the preceding unseen H26
validation batches. It supports a stable mid-game improvement over the
original Agent but does not establish a stable completion rate.

Artifacts:

- `runs/balatro-human-loop-h26-random-test-paired16-seed36079239/`
- `runs/balatro-human-loop-h26-random-test-paired16-seed67411017/`
- `runs/balatro-human-loop-h26-random-test-paired16-seed43177812/`
- `runs/balatro-human-loop-h26-random-test-paired16-seed61585883/`
