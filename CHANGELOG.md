# Changelog

All notable changes to hlbench-pro are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.1.0a0] — 2026-05-29

### Added

This is the MVP — single env (Pendulum-v1), full submit-to-finalize
loop end-to-end. See [`docs/findings.md`](./docs/findings.md) for
the calibration sweep that closed out the 14-day plan in
[`docs/architecture.md`](./docs/architecture.md).

**Server library** (`src/hlbench/`):
- `core.Server` — per-run server class with `info()` / `submit()` /
  `finalize()` (SPEC §3 server interface, SPEC §1.1 `/info` schema).
- `core.SubmitHandler` — 7-phase submit lifecycle with the unified
  11-verdict enum (`docs/submit-protocol.md` §2-3).
- `core.Sandbox` — `multiprocessing(spawn)` child holding one
  `Policy` instance, `signal.setitimer(ITIMER_REAL)` for `act()`
  wall-time, parent-side `Pipe.poll(timeout)` for init wall-time.
- `core.env_runner` — single-episode runner with the SPEC §4.2
  trajectory schema; categorizes per-episode failures as
  `act_error` / `act_timeout` / `reset_error` / `on_episode_end_error`.
- `core.heldout` — held-out evaluation that runs all M held-out
  seeds against a snapshotted final policy (SPEC §6).
- `core.scoring` — normalized score, `clip(0, 1.2) * 100` final
  score (SPEC §5.2), AUC + episodes-to-Npct auxiliary metrics
  (SPEC §5.3, normalized interpretation per the Day 14 spec fix).
- `core.feedback` — atomic `summary.json` writer, per-episode
  `trajectory.jsonl` with NaN/Inf encoded as strings, submit/episode
  `errors.txt` writers (SPEC §4.1-4.4).
- `core.seed_manager` — env-instance ID → hidden real seed
  resolver; train and held-out pools loaded from static JSON
  alongside the env package.
- `envs.registry` — `EnvDefinition` dataclass; `register_env()` /
  `get_env()`; `public_env_meta()` strips server-internal fields
  (`expert_baseline`, `random_baseline`, seed paths) per CLAUDE.md
  invariant 2.
- `envs.pendulum` — Pendulum-v1 with 256 train + 256 held-out
  seeds (disjoint, generated from `master_seed=42`), `TASK.md`
  template, expert/random baselines.
- `http_server` — stdlib `http.server` HTTP wrapper exposing
  `GET /info`, `POST /submit`, `POST /finalize` (CLAUDE.md
  invariant 8).

**Consumers** (outside `src/hlbench/` per the lib/consumer
separation rule):
- `hlbench_cli/` — argparse CLI: `hlbench {init,serve,info,submit,
  finalize}`. Range / list / mixed env-instance specs (`0-7`,
  `0,2,5`, `0-3,7`).
- `agents/pd_pendulum/policy.py` — reference energy-shaping
  swing-up + PD stabilize controller. Drop into
  `workspace/system/policy.py` to submit it.
- `scripts/gen_seeds.py` — regenerates `train.json`/`heldout.json`
  from `master_seed`. Reproducible: re-running with `--master-seed
  42` yields the byte-identical files checked in.
- `scripts/calibration.py` — sweeps the budget and tabulates
  `final_score` + auxiliary metrics. Output drives
  `docs/findings.md`.

**Documentation**:
- `README.md` — pitch, quick start, document map.
- `AGENT.md` — agent rules, sandbox, anti-hack (kept as-shipped).
- `SPEC.md` — wire contract, scoring, held-out details.
- `docs/quickstart.md` — 60-second walkthrough plus agent-loop stub.
- `docs/architecture.md` — original implementation plan with a
  status banner and as-built layout.
- `docs/output.md` — `runs/<...>/` layout.
- `docs/submit-protocol.md` — 7 phases × 11 verdicts.
- `docs/findings.md` — calibration analysis (Day 14): `final_score`
  is constant for stateless policies (PD on Pendulum hits 98.3 at
  every budget); `auc_in_loop` rises with budget because of the
  `(0, 0)` start anchor; `held_out_gap = +8` validates determinism.

**Tests**: 87 across 8 files. Coverage 84% (gaps are sandbox
child-process code behind the multiprocessing barrier, `cmd_serve`
which blocks forever, and the deferred `oom` partial-execute path).

### Spec changes (Day 14)

- **SPEC §5.3** `episodes_to_50pct` / `episodes_to_80pct`: was
  `mean_return >= 0.5 × expert`, which inverts for negative-reward
  envs (Pendulum has expert ≈ -150, so 0.5 × expert ≈ -75 is
  unreachably *better* than expert). Now defined as
  `normalized_score(mean_return) >= 0.5`. Implementation in
  `scoring.episodes_to_threshold` already used the normalized form;
  the spec wording now matches.

### Known limitations

- `denied_imports` enforcement: not yet enforced (post-MVP bundle).
- Network blocking: not yet enforced (post-MVP bundle).
- RSS-poll OOM detection: not implemented; `RLIMIT_AS` is set
  best-effort but Apple's malloc often ignores it on macOS.
- `submit_wall_s` deadline: not enforced.
- `stdout.txt` / `stderr.txt` per-episode capture: not implemented.
- `observations.npy` external-obs storage: not implemented (Pendulum
  only needs inline).
- 64KB error-file truncation cap: not implemented.
- `system/.final_submit` agent-designated final policy: not
  honored (always uses the most recent successful submit).
- Per-submit code snapshots in `checkpoints/`: not produced.
- `agent.jsonl` / `harness.log` / `env.log`: not produced.
- Only Pendulum-v1 ships; HalfCheetah / CarRacing / Atari are on
  the post-MVP roadmap.

[Unreleased]: https://example.com/compare/v0.1.0a0...HEAD
[0.1.0a0]: https://example.com/releases/tag/v0.1.0a0
