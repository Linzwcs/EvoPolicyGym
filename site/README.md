# EvoPolicyGym website

The project website is a bilingual Docusaurus 3 application. Docusaurus owns
the documentation, research blog, navigation, localization, metadata, sitemap,
and static build. Purpose-built React pages retain the project's academic
identity and render structured benchmark evidence.

| Content | Source | Rendering |
| --- | --- | --- |
| Home and curated project pages | `src/pages/**/*.tsx` | React pages inside the Docusaurus shell |
| Documentation | `docs/*.md` | Docusaurus Docs, with Chinese sources under `i18n/zh-CN/` |
| Research blog | `blog/*.md` | Docusaurus Blog, with Chinese sources under `i18n/zh-CN/` |
| Environment narratives | `environments/**/*.mdx` | Markdown/MDX reference pages at `/environments/**` |
| Versions and paper identity | `src/data/project.ts` | Shared structured values |
| Structured catalogs and experiment results | Typed data plus `plugins/generated-pages/` | Generated routes with purpose-built React views |
| Core16 media and replay artifacts | `public/` | Static assets consumed by result and replay views |

The shared visual system lives in `src/css/custom.css` and reusable page
components live in `src/components/`. Editorial prose stays in Markdown.
Catalogs, score matrices, reruns, and interactive replay behavior stay in typed
data and React features rather than being forced into articles.

New Environment explanations should start from
`environments/_template.mdx`. The leading underscore keeps the template out of
the build. Front matter owns the route and metadata; Markdown owns the research
narrative, and MDX may embed figures, video, or a focused interactive
component. The catalog remains a concise index rather than duplicating those
details.

English is served at the site root and Chinese under `/zh-CN/`. The locale
switcher, translated navigation, Docs sidebar, Blog UI, and paired content are
managed through Docusaurus i18n.

## Local development

Use Node 22 LTS. The project currently caps Node below 25 because newer Node
runtimes are not yet reliable with Docusaurus's local static preview.

```bash
npm install
npm run dev
npm run typecheck
npm run build
npm run serve
```

The production output is written to `build/`. The GitHub Pages workflow uses
Node 22, installs the locked dependencies, builds both locales, and publishes
that directory when site changes reach `main`.

`public/media/` contains the Core16 paper-companion reruns. The current v0.3
runtime and historical research results remain labelled separately throughout
the site.
