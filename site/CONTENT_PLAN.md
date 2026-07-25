# EvoPolicyGym Site Content Plan

This document defines the website information architecture while the clean
runtime and public contracts are still changing. It is a planning source, not
published product documentation.

## Principles

- The site must distinguish implemented behavior, draft contracts, planned
  systems, and historical research evidence.
- Public documentation follows code and representative tests. It does not
  invent a stable API ahead of them.
- Core explanatory pages are bilingual Markdown. Data-heavy result views remain
  Astro pages backed by explicit datasets.
- Environment authoring and benchmark results are separate surfaces.
- Version controls appear only when they correspond to real snapshots.

## Top-level information architecture

| Area | Purpose | Initial blocks | Content mode |
| --- | --- | --- | --- |
| Home | Explain the research thesis and route readers. | Thesis, current status, lifecycle, quick start, project portals. | Short curated Astro page. |
| Docs | Explain the currently implemented system. | Getting started, lifecycle and architecture, Policy ABI, runtime and safety. | Version-aware bilingual Markdown. |
| Environments | Compare current Benchmark distributions and explain how an external Environment is authored. | Four Gymnasium suites, Balatro, Policy interfaces, parameters, scoring, conformance. | Implementation-backed catalog plus normative authoring guide. |
| Results | Preserve and inspect benchmark evidence. | Methodology, score matrix, qualitative reruns, per-environment records. | Immutable experiment data with custom views. |
| Research | Hold rationale and forward-looking work. | Paper, protocol rationale, architecture notes, roadmap. | Essays and design notes; clearly marked draft or historical. |

Project governance links such as README, CONTRIBUTING, CHANGELOG, license, and
repository source remain in the footer or GitHub until they justify a dedicated
Project section.

## Content maturity

Every documentation page belongs to one visible state:

| State | Meaning | Allowed content |
| --- | --- | --- |
| <code>planning</code> | The section and questions are agreed, but implementation is moving. | Outline, ownership, status, completion gates. |
| <code>draft</code> | Code and tests exist, but compatibility is not frozen. | Implementation-backed guidance with explicit limitations. |
| <code>stable</code> | A released compatibility surface exists. | Complete guide, runnable examples, migration guarantees for that release. |
| <code>historical</code> | The material records a past release or experiment. | Corrections and provenance only; no silent semantic updates. |

## Documentation versions

The initial website has one real documentation channel:

- <code>next</code>: tracks the default branch and may change.

When the first compatibility-bearing release is frozen:

- <code>/docs/</code> points to the latest stable documentation;
- <code>/docs/next/</code> follows active development;
- <code>/docs/vX.Y/</code> preserves a release snapshot;
- the Environment landing page routes readers to the appropriate authoring
  version;
- a version selector is added only after at least two real snapshots exist.

Historical Results are versioned by experiment identity and generation date,
not automatically by the Python package version.

## Release update workflow

For each future release:

1. Update <code>src/data/project.ts</code> for the global package, protocol, and
   docs labels.
2. Freeze compatibility-bearing Markdown into the release snapshot.
3. Set page maturity to <code>stable</code>, <code>draft</code>, or
   <code>historical</code> as appropriate.
4. Add migration and compatibility notes only for contracts that actually
   changed.
5. Validate all bilingual routes, internal links, and canonical URLs.
6. Never rewrite historical score or rerun meaning to match a newer runtime.

## Environment section status

The public boundary, structural entry points, lifecycle, conformance semantics,
and CartPole reference contract are implementation-backed. The live catalog now
covers all 23 current tasks in Gymnasium's four documented built-in suites plus
Balatro. Future dedicated Environment pages should use the CartPole contract
shape and must remain grounded in the corresponding distribution and tests.
