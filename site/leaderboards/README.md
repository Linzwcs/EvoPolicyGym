# Leaderboard content packages

Each child directory is one immutable version of a leaderboard Distribution.
The site discovers these directories during the build; adding a Distribution
or adding Environments to one does not require a new page component or route
declaration. Each Environment becomes an independent leaderboard within its
Distribution.

```text
leaderboards/<distribution-version-slug>/
├── index.mdx                 # manifest plus English aggregate-page content
├── index.zh-CN.mdx           # Chinese aggregate-page content
├── environment.mdx           # English per-Environment page template
├── environment.zh-CN.mdx     # Chinese per-Environment page template
└── results.json              # environments, entries, scores, and provenance
```

`index.mdx` front matter owns Distribution-version identity, status, labels,
Profile, localized content paths, and the results filename. Its Markdown body
owns the Distribution page's headings, prose, captions, links, and section
order. The Environment templates do the same for generated per-Environment
leaderboard routes. Dynamic MDX components such as `AggregateTable`,
`EnvironmentDirectory`, and `EnvironmentChart` render structured values; they
do not contain editorial copy.

`results.json` uses `evopolicygym/leaderboard-results/v1`. Environment summaries
are localized alongside each Environment record so a Distribution version
remains self-contained. Agent entries must report every Environment in that
version; baselines may be partial. Raw scores are only compared within a single
Environment.

Active Distributions may declare machine-readable `test_configurations`. A
configured Environment lists its available configuration IDs and one default;
each entry then stores raw scores by Environment and configuration. The
Environment page selects a configuration before ranking, and its JSON export
contains that complete configuration together with the ordered raw scores.
Legacy archive results may retain the original flat Environment-to-score map.
Agent entries may declare `thinking_effort`; it belongs to the model invocation
and is rendered with the model, never as part of the test configuration.

The active site is publication-scoped. A result is added only when its complete
experiment and configuration are documented in a public EvoPolicyGym Blog
article or explicitly published as a leaderboard experiment. Diagnostic and
smoke runs remain local evidence and are not promoted to public rankings.
Numbers transcribed from an article must preserve the article's published
precision, even when a retained machine report contains additional decimal
places.

Exactly one active Distribution version must set `default: true`. Directory
names must match the manifest slug, content and result paths must remain inside
the directory, and all referenced files are validated during local development
and production builds. Historical paper results use `status: archived` and are
kept read-only. Public active routes live below `/leaderboard/distributions/`;
archives live below `/leaderboard/archive/`. Legacy `/leaderboard/suites/`
routes remain available for compatibility.
