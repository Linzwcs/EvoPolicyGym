import {useDoc} from "@docusaurus/plugin-content-docs/client";
import {ThemeClassNames} from "@docusaurus/theme-common";
import Heading from "@theme/Heading";
import MDXContent from "@theme/MDXContent";
import clsx from "clsx";
import type {ReactNode} from "react";
import type {Props} from "@theme/DocItem/Content";

type ResearchFrontMatter = {
  lead?: string;
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

  return (
    <div className={clsx(ThemeClassNames.docs.docMarkdown, "markdown")}>
      {syntheticTitle && (
        <div className="doc-article-column">
          <header className="doc-article-header">
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
