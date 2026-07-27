# EvoPolicyGym website

The project website is a static Astro application. It intentionally uses no
client-side framework; JavaScript is limited to navigation, language preference,
and result filtering.

The site uses a deliberately mixed content model:

| Content | Source | Rendering |
| --- | --- | --- |
| Home and other tightly curated landing pages | Astro route | Direct composition from shared theme components |
| Documentation | `src/content/docs/{en,zh}/*.md` | Schema-validated bilingual Markdown |
| Blog posts | `src/content/blog/{en,zh}/*.md` | Date-ordered bilingual Markdown |
| Versions and paper identity | `src/data/project.ts` | Shared structured values |
| Environment catalogs and experiment results | Typed data or generated datasets | Purpose-built Astro views |

Theme code lives in `src/layouts/`, `src/components/`, and `src/styles/`.
Editorial prose does not belong in theme components. Conversely, Markdown
should not contain layout classes or depend on a particular visual treatment.
`BilingualArticleLayout.astro` provides the article shell for Docs, while the
route only loads paired content entries and derives the table of contents.

Blog entries declare `publishedAt`, `author`,
`tags`, and publication status; matching bilingual files generate
`/blog/<page>/` and appear in reverse chronological order.

The homepage intentionally remains a direct Astro page: it is a short,
carefully composed project entrance rather than a long-form article. The
Balatro page likewise renders bounded `replay.jsonl` artifacts entirely in the
browser. Keep data-heavy galleries and replay pages as Astro components rather
than forcing them into Markdown.

```bash
npm install
npm run dev
npm run build
```

`public/media/` contains the Core16 paper-companion reruns. The current v0.3
runtime and the historical research results are labelled separately throughout
the site.
