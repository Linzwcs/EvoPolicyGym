# paper/

This directory holds the HLBench-Pro paper drafts.

## Current

- **`paper-v0.md`** — v0.1 preprint draft (May 2026). Markdown
  source. Pre-arXiv revision: covers the protocol, the v1
  environment roster (16 envs across 6 categories), the 6 envs
  currently landed, the design rationale, and reference
  baselines. Frontier-model evaluation matrix is deferred to
  v1.0.

## Workflow

This v0.1 draft is intentionally markdown to keep iteration
fast. Before arXiv submission:

1. Convert to LaTeX via Pandoc (`pandoc paper-v0.md -o paper-v0.tex`).
2. Pick a stylesheet (NeurIPS, ICLR, or arXiv-default).
3. Render figures (capability matrix, score histograms, etc.)
   from the data sources called out in the paper body.
4. Add proper citations (currently lightweight inline).
5. Final author list + affiliations.

## Versioning

Subsequent revisions land as `paper-vN.md` siblings. The repo's
git history is the authoritative version log; we do not
maintain a `CHANGELOG.md` per paper.

## Status

- v0.1 (this draft): **protocol-only**. Empirical claims limited
  to reference-PD baseline on Pendulum-v1.
- v1.0 (planned): adds full frontier-model matrix
  (5+ models × 16 envs), once `observations.npy` infrastructure
  lands and visual envs are implemented.
