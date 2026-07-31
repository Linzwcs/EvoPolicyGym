import type {ReactNode} from "react";
import clsx from "clsx";
import {useBlogPost} from "@docusaurus/plugin-content-blog/client";
import BlogPostItemContainer from "@theme/BlogPostItem/Container";
import BlogPostItemContent from "@theme/BlogPostItem/Content";
import BlogPostItemFooter from "@theme/BlogPostItem/Footer";
import BlogPostItemHeader from "@theme/BlogPostItem/Header";
import BlogPostItemHeaderInfo from "@theme/BlogPostItem/Header/Info";
import BlogPostItemHeaderTitle from "@theme/BlogPostItem/Header/Title";
import type {Props} from "@theme/BlogPostItem";

export default function BlogPostItem({
  children,
  className,
}: Props): ReactNode {
  const {metadata, isBlogPostPage} = useBlogPost();

  if (isBlogPostPage) {
    return (
      <BlogPostItemContainer className={className}>
        <BlogPostItemHeader />
        <BlogPostItemContent>{children}</BlogPostItemContent>
        <BlogPostItemFooter />
      </BlogPostItemContainer>
    );
  }

  return (
    <BlogPostItemContainer className={clsx("blog-summary-card", className)}>
      <header className="blog-summary-card__header">
        <div className="blog-summary-card__meta">
          <BlogPostItemHeaderInfo />
        </div>
        <BlogPostItemHeaderTitle className="blog-summary-card__title" />
      </header>
      <p className="blog-summary-card__description">{metadata.description}</p>
      <BlogPostItemFooter />
    </BlogPostItemContainer>
  );
}
