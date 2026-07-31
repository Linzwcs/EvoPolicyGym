import Layout from "@theme/Layout";
import type {ReactNode} from "react";
import {Localized} from "./Localized";

interface AcademicPageProps {
  title: string;
  description: string;
  eyebrow?: ReactNode;
  heading: ReactNode;
  lead: ReactNode;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function AcademicPage({
  title,
  description,
  eyebrow,
  heading,
  lead,
  meta,
  children,
  className = "",
}: AcademicPageProps) {
  return (
    <Layout title={title} description={description}>
      <main className={`epg-page ${className}`.trim()}>
        <header className="epg-page-hero epg-wide">
          <div>
            {eyebrow && <p className="epg-eyebrow">{eyebrow}</p>}
            <h1>{heading}</h1>
            <div className="epg-lead">{lead}</div>
          </div>
          {meta && <aside className="epg-record">{meta}</aside>}
        </header>
        {children}
      </main>
    </Layout>
  );
}

export function SectionHeading({
  index,
  title,
  description,
}: {
  index: string;
  title: ReactNode;
  description?: ReactNode;
}) {
  return (
    <header className="epg-section-heading">
      <span>{index}</span>
      <div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </header>
  );
}

export function ResearchBoundary() {
  return (
    <div className="epg-note epg-note--historical">
      <strong>
        <Localized en="Historical research record" zh="历史研究记录" />
      </strong>
      <p>
        <Localized
          en="This evidence belongs to the paper-era experiment and is not a guarantee of current package availability."
          zh="这些证据属于论文时期实验，并不表示相关 package 当前仍然可用。"
        />
      </p>
    </div>
  );
}
