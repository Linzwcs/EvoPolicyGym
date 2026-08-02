import Link from "@docusaurus/Link";
import type {ReactNode} from "react";

export function LeaderboardIntro({
  eyebrow,
  children,
}: {
  eyebrow?: string;
  children: ReactNode;
}) {
  return (
    <section className="leaderboard-paper-intro">
      {eyebrow && <p className="leaderboard-paper-label">{eyebrow}</p>}
      {children}
    </section>
  );
}

export function LeaderboardLead({children}: {children: ReactNode}) {
  return <p className="leaderboard-paper-lead">{children}</p>;
}

export function LeaderboardNote({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <aside className="leaderboard-paper-note">
      <strong>{title}</strong>
      <div>{children}</div>
    </aside>
  );
}

export function LeaderboardSection({
  number,
  title,
  lead,
  variant,
  children,
}: {
  number: string;
  title: string;
  lead?: string;
  variant?: "method";
  children: ReactNode;
}) {
  const className = `leaderboard-paper-section${
    variant === "method" ? " leaderboard-paper-method" : ""
  }`;
  return (
    <section className={className}>
      <header>
        <span>{number}</span>
        <div>
          <h2>{title}</h2>
          {lead && <p>{lead}</p>}
        </div>
      </header>
      {variant === "method" ? (
        <div className="leaderboard-paper-section-body">{children}</div>
      ) : (
        children
      )}
    </section>
  );
}

export function LeaderboardCaption({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <p className="leaderboard-paper-caption">
      <span>{label}</span> {children}
    </p>
  );
}

export function LeaderboardLinks({children}: {children: ReactNode}) {
  return <nav className="leaderboard-paper-links">{children}</nav>;
}

export function LeaderboardLink({
  to,
  children,
}: {
  to: string;
  children: ReactNode;
}) {
  return <Link to={to}>{children}</Link>;
}
