# EvoPolicyGym Site Content Plan

This document defines the website information architecture while the clean
runtime and public contracts are still changing. It is a planning source, not
published product documentation.

## Principles

- The site distinguishes implemented behavior, draft contracts, planned
  systems, and historical research evidence.
- Public documentation follows code and representative tests. It does not
  invent a stable API ahead of them.
- Core explanatory pages are bilingual Markdown managed by Docusaurus.
- Data-heavy research views remain purpose-built React pages backed by explicit
  datasets.
- Environment authoring and benchmark results are separate surfaces.
- Version controls appear only when they correspond to real snapshots.

## Top-level information architecture

| Area | Purpose | Initial blocks | Content mode |
| --- | --- | --- | --- |
| Home | Explain the research thesis and route readers. | Thesis, current status, lifecycle, quick start, project portals. | Curated React page in the Docusaurus shell. |
| Docs | Explain the currently implemented system. | Getting started, lifecycle and architecture, Policy ABI, runtime and safety. | Version-ready bilingual Markdown. |
| Environments | Index current Benchmark distributions and give each Environment room for its own research narrative. | Concise collection index; per-Environment task, interface, evaluation, interpretation, and media when available. | Typed catalog plus Markdown/MDX Environment pages and the normative authoring guide. |
| Results | Preserve and inspect benchmark evidence. | Methodology, score matrix, qualitative reruns, per-environment records. | Immutable experiment data with generated React views. |
| Blog | Record when and why the project changes. | Releases, engineering decisions, Benchmark integrations, experiment retrospectives. | Date-ordered bilingual Markdown posts. |

## Theme and content boundary

The site does not force every surface into Markdown. Choose the source by the
kind of change:

- write curated landing pages in `src/pages/` when structure and visual
  composition change together;
- write explanatory Docs prose in `docs/` and Blog prose in `blog/`;
- write Environment narratives in `environments/`, using MDX only when figures,
  video, or a focused interactive component requires it;
- keep Chinese Docs and Blog sources in the corresponding
  `i18n/zh-CN/docusaurus-plugin-content-*` directories;
- keep release identity and other shared facts in `src/data/`;
- keep large catalogs, score matrices, reruns, and generated evidence in typed
  datasets with purpose-built views;
- keep shared typography, spacing, navigation integration, article treatment,
  and responsive behavior in `src/css/` and `src/components/`.

Content files must not carry theme classes. Theme code must not contain
paragraph-length article copy. Front matter drives titles, descriptions,
ordering, tags, publication dates, and authorship. Docusaurus owns route
generation and localized content pairing.

Blog posts link to Docs, Results, and Environments as factual sources; they do
not redefine those surfaces. Project governance links such as README,
CONTRIBUTING, CHANGELOG, license, and repository source remain in the footer or
GitHub until they justify a dedicated Project section.

## Content maturity

Every documentation page belongs to one visible state:

| State | Meaning | Allowed content |
| --- | --- | --- |
| <code>planning</code> | The section and questions are agreed, but implementation is moving. | Outline, ownership, status, completion gates. |
| <code>draft</code> | Code and tests exist, but compatibility is not frozen. | Implementation-backed guidance with explicit limitations. |
| <code>stable</code> | A released compatibility surface exists. | Complete guide, runnable examples, migration guarantees for that release. |
| <code>historical</code> | The material records a past release or experiment. | Corrections and provenance only; no silent semantic updates. |

## Documentation versions

The initial website has one real documentation channel: the current default
branch. Docusaurus versioning should be enabled only when a compatibility-
bearing release is frozen. At that point:

- `/docs/` points to the latest stable documentation;
- a separate development snapshot follows active work;
- prior release snapshots remain immutable;
- Environment authoring links target the appropriate documentation version;
- a version selector appears only after at least two real snapshots exist.

Historical Results are versioned by experiment identity and generation date,
not automatically by the Python package version.

## Release update workflow

For each future release:

1. Update `src/data/project.ts` for package, protocol, paper, and docs labels.
2. Freeze compatibility-bearing Docs through Docusaurus versioning.
3. Mark content `stable`, `draft`, or `historical` as appropriate.
4. Add migration notes only for contracts that actually changed.
5. Validate type safety, both locale builds, internal links, and canonical URLs.
6. Never rewrite historical score or rerun meaning to match a newer runtime.

## Environment section status

The public boundary, lifecycle, conformance semantics, and Environment
reference contracts are implementation-backed. The catalog covers 57
independently installable Benchmark distributions and at least 208 named tasks
or profiles across 11 upstream ecosystems. Parameterized collections and size
variants extend the concrete configuration surface beyond that named count.
The typed catalog remains a compact inventory. Environment-specific
explanation and media live in Markdown/MDX pages and remain grounded in the
corresponding distribution and tests.
