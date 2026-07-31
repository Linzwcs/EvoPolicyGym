import {useDoc} from "@docusaurus/plugin-content-docs/client";
import {ThemeClassNames} from "@docusaurus/theme-common";
import Heading from "@theme/Heading";
import MDXContent from "@theme/MDXContent";
import clsx from "clsx";
import type {ReactNode} from "react";
import type {Props} from "@theme/DocItem/Content";

type ResearchFrontMatter = {
  docsVersion?: string;
  index?: string;
  lead?: string;
  status?: string;
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
  const research = frontMatter as typeof frontMatter & ResearchFrontMatter;
  const metadata = [
    research.index,
    research.docsVersion,
    research.status,
  ].filter((value): value is string => Boolean(value));

  return (
    <div className={clsx(ThemeClassNames.docs.docMarkdown, "markdown")}>
      {syntheticTitle && (
        <header className="doc-article-header">
          {metadata.length > 0 && (
            <div className="doc-article-meta" aria-label="Document metadata">
              {metadata.map((value, index) => (
                <span key={value} className={index === 0 ? "is-index" : undefined}>
                  {value}
                </span>
              ))}
            </div>
          )}
          <Heading as="h1">{syntheticTitle}</Heading>
          {research.lead && <p className="doc-article-lead">{research.lead}</p>}
        </header>
      )}
      <MDXContent>{children}</MDXContent>
    </div>
  );
}
