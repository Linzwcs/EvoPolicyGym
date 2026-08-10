import {useDoc} from "@docusaurus/plugin-content-docs/client";
import {ThemeClassNames} from "@docusaurus/theme-common";
import Heading from "@theme/Heading";
import MDXContent from "@theme/MDXContent";
import clsx from "clsx";
import type {ReactNode} from "react";
import type {Props} from "@theme/DocItem/Content";
import {useSiteLanguage} from "../../../components/Localized";

type ResearchFrontMatter = {
  lead?: string;
  index?: string;
  docsVersion?: string;
  status?: string;
};

const statusLabels: Record<string, {en: string; zh: string}> = {
  current: {en: "Current documentation", zh: "当前文档"},
  stable: {en: "Stable documentation", zh: "稳定文档"},
  draft: {en: "Draft documentation", zh: "文档草案"},
  planning: {en: "Structure planning", zh: "结构规划"},
  historical: {en: "Historical record", zh: "历史记录"},
};

function useSyntheticTitle(): string | null {
  const {metadata, frontMatter, contentTitle} = useDoc();
  const shouldRender =
    !frontMatter.hide_title && typeof contentTitle === "undefined";
  return shouldRender ? metadata.title : null;
}

export default function DocItemContent({children}: Props): ReactNode {
  const syntheticTitle = useSyntheticTitle();
  const {frontMatter} = useDoc();
  const language = useSiteLanguage();
  const research = frontMatter as typeof frontMatter & ResearchFrontMatter;
  const status = research.status
    ? (statusLabels[research.status]?.[language] ?? research.status)
    : undefined;

  return (
    <div className={clsx(ThemeClassNames.docs.docMarkdown, "markdown")}>
      {syntheticTitle && (
        <div className="doc-article-column">
          <header className="doc-article-header">
            {(research.index || research.docsVersion || status) && (
              <div className="doc-article-meta">
                {research.index && <strong>{research.index}</strong>}
                {research.docsVersion && <span>{research.docsVersion}</span>}
                {status && (
                  <span className="doc-article-status">
                    <i aria-hidden="true" /> {status}
                  </span>
                )}
              </div>
            )}
            <Heading as="h1">{syntheticTitle}</Heading>
            {research.lead && <p className="doc-article-lead">{research.lead}</p>}
          </header>
        </div>
      )}
      <div className="doc-article-column doc-article-column--body">
        <MDXContent>{children}</MDXContent>
      </div>
    </div>
  );
}
