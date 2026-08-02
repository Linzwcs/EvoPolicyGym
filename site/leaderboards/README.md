# Leaderboard content packages

Each child directory is one immutable or evolving leaderboard Suite. The site
discovers these directories during the build; adding a Suite does not require a
new page component or route declaration.

```text
leaderboards/<suite-slug>/
├── index.mdx                 # manifest plus English aggregate-page content
├── index.zh-CN.mdx           # Chinese aggregate-page content
├── environment.mdx           # English per-Environment page template
├── environment.zh-CN.mdx     # Chinese per-Environment page template
└── results.json              # environments, entries, scores, and provenance
```

`index.mdx` front matter owns Suite identity, status, labels, Profile, localized
content paths, and the results filename. Its Markdown body owns the aggregate
page's headings, prose, captions, links, and section order. The Environment
templates do the same for generated per-Environment routes. Dynamic MDX
components such as `AggregateTable`, `EnvironmentDirectory`, and
`EnvironmentChart` render structured values; they do not contain editorial
copy.

`results.json` uses `evopolicygym/leaderboard-results/v1`. Environment summaries
are localized alongside each Environment record so a Suite remains
self-contained. Agent entries must report every Environment in the Suite;
baselines may be partial. Raw scores are only compared within a single
Environment.

Exactly one Suite must set `default: true`. Directory names must match the Suite
slug, content and result paths must remain inside the Suite directory, and all
referenced files are validated during local development and production builds.
Historical Suites should be frozen instead of edited in place.
