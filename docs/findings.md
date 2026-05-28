# Calibration findings — Pendulum-v1, reference PD

Six budgets (8/16/32/64/128/256) × one run each. Reference policy is
the energy-shaping swing-up + PD stabilize in `agents/pd_pendulum/`,
which is **stateless** — it doesn't learn from feedback. So this run
isolates the budget's effect on the *measurement*, not on the policy.

## Raw numbers

| budget | n_submits | final_score | held_out_mean | in_loop_mean | auc_in_loop | to_50pct | to_80pct |
|-------:|----------:|------------:|--------------:|-------------:|------------:|---------:|---------:|
|      8 |         1 |        98.3 |        −168.0 |       −123.0 |        51.3 |        8 |        8 |
|     16 |         2 |        98.3 |        −168.0 |       −161.2 |        75.1 |        8 |        8 |
|     32 |         4 |        98.3 |        −168.0 |       −161.5 |        86.3 |        8 |        8 |
|     64 |         8 |        98.3 |        −168.0 |       −178.8 |        91.3 |        8 |        8 |
|    128 |        16 |        98.3 |        −168.0 |       −169.6 |        95.0 |        8 |        8 |
|    256 |        32 |        98.3 |        −168.0 |       −167.8 |        96.8 |        8 |        8 |

(`max_episodes_per_submit = 8` for all runs; PD on Pendulum runs every
episode for the full 200 steps so the unit of work is roughly constant.)

## What we expected (and saw)

1. **`held_out_mean` and `final_score` constant across budgets.** The
   policy never changes, the held-out pool never changes, the
   evaluation is deterministic → the score is a single fixed point.
   *This is the test of held-out determinism — passed.*

2. **`auc_in_loop` rises monotonically with budget.** The trapezoidal
   AUC is anchored at `(0, 0)`. As the budget grows, the policy spends
   proportionally more wall-time at its plateau, so the area-under
   averages closer to the plateau value. Reference PD plateaus at
   normalized ≈ 1.0 (`final_score / 100`), and AUC asymptotes toward
   that as `budget → ∞`.

3. **`episodes_to_50pct` and `episodes_to_80pct` = 8 always.** The PD
   clears both thresholds in its first submit (4 episodes/submit on
   small budgets, 8 on larger), so the metric latches early and never
   moves.

## What surprised us

4. **In-loop sample means look noisy even with a deterministic policy.**
   Each row's `in_loop_mean` is the mean of submit means, and individual
   submits can land anywhere in `[-264, -120]` depending on which
   `env_instance` IDs they hit (Pendulum's initial state distribution
   has high variance). This is a property of the env, not the
   benchmark — but it tells us that **per-submit `mean_return` is a
   noisy signal** at small `max_episodes_per_submit`. An adaptive
   agent that sizes submits to noise floor would see this in
   `summary.json:std_return`.

5. **`held_out_gap` is small and positive (+8 on a 256-episode budget).**
   The in-loop final and held-out means agree to within env noise. No
   overfitting because there's no fitting. This makes Pendulum + PD a
   good null-hypothesis fixture for the benchmark — agents that
   "overfit" must beat this baseline gap to count as having actually
   learned something.

## Implications for the spec

### `episodes_to_Npct` definition is broken for negative-reward envs

SPEC.md §5.3 says ">= 0.5 × expert". For Pendulum (`expert = −150`),
that's `>= −75`, which is *better than expert* — the threshold is
unreachable. The implementation in `scoring.episodes_to_threshold`
ignores the literal SPEC wording and uses the normalized form:
`normalized_score(mean) >= 0.5`. Action item in the SPEC pass below.

### AUC under-counts small-budget runs

A *learning* agent with a 16-budget gets a structurally lower AUC
than the same agent with a 256-budget, even if both end at the same
plateau. The (0, 0) anchor isn't wrong per se — it represents "before
any submit, you've learned nothing" — but it does mean AUC is not a
clean cross-budget comparison metric. Recommend documenting AUC as a
*within-budget* signal (compare agents at the same budget), not a
cross-budget one.

### Stateless baselines collapse the benchmark to "did you write a
working policy?"

Pendulum-v1 happens to be small enough that a hand-tuned PD is at
expert level. So *any agent* that writes a correct PD on first
attempt scores 98+, regardless of budget. To discriminate agents,
we need:
- harder envs where the optimal policy needs iteration (HalfCheetah,
  CarRacing), or
- a `min_episodes_per_submit` floor that forces multiple submits, so
  short-circuit agents can't take "first try wins"

These belong on the post-Pendulum roadmap, not in the MVP.

## Reproducing

```bash
.venv/bin/python scripts/calibration.py \
  --budgets 8 16 32 64 128 256 \
  --max-per-submit 8 --runs 1 \
  --out calibration.json
```

`--runs N` averages across N repeats per budget; useful if a future
env has non-determinism in `env_factory()` (Pendulum doesn't).
