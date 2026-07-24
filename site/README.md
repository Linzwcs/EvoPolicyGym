# EvoPolicyGym website

The project website is a static Astro application. It intentionally uses no
client-side framework; JavaScript is limited to navigation, language preference,
and result filtering.

The visual shell lives in `src/layouts/`, `src/components/`, and
`src/styles/global.css`. Core guides are schema-validated bilingual Markdown
entries under `src/content/docs/{en,zh}/`; `src/pages/docs/[slug].astro` renders
them through the shared documentation theme. The Balatro page renders bounded
`replay.jsonl` artifacts entirely in the browser. Keep data-heavy galleries and
replay pages as Astro components rather than forcing them into Markdown.

```bash
npm install
npm run dev
npm run build
```

`public/media/` contains the Core16 paper-companion reruns. The current v0.3
runtime and the historical research results are labelled separately throughout
the site.
