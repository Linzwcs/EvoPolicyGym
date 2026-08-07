import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import type {
  LeaderboardRegistry,
  LeaderboardRegistryItem,
} from "../../../lib/leaderboard/types";
import {LeaderboardNavigator} from "./LeaderboardShell";
import {leaderboardPath} from "./model";

export default function LeaderboardIndexPage({
  registry,
}: {
  registry: LeaderboardRegistry;
}) {
  const language = useSiteLanguage();
  const active = registry.suites.filter(
    (item) => item.manifest.status !== "archived",
  );
  const archived = registry.suites.filter(
    (item) => item.manifest.status === "archived",
  );
  const copy =
    language === "zh"
      ? {
          title: "排行榜",
          description:
            "先选择 Distribution，再进入 Environment、选择测试配置，并按原始 Assessment 分数生成排名。",
          active: "Active Distributions",
          activeLead: "当前可继续加入 Environment 与评测结果的榜单集合。",
          archive: "Archive",
          archiveLead: "历史论文结果只读保存，不与当前 Distribution 混排。",
          environments: "Environments",
          profile: "测试 Profile",
          open: "打开 Distribution",
          archivedLabel: "历史论文档案",
          activeLabel: "当前 Distribution",
        }
      : {
          title: "Leaderboard",
          description:
            "Choose a Distribution, open an Environment, select a test configuration, and rank raw Assessment scores.",
          active: "Active Distributions",
          activeLead:
            "Current leaderboard collections that can grow with new Environments and evaluations.",
          archive: "Archive",
          archiveLead:
            "Historical paper results remain read-only and never mix with active Distribution rankings.",
          environments: "Environments",
          profile: "Test profile",
          open: "Open Distribution",
          archivedLabel: "Historical paper archive",
          activeLabel: "Active Distribution",
        };

  return (
    <Layout title={copy.title} description={copy.description}>
      <main className="leaderboard-paper">
        <div className="leaderboard-paper-shell">
          <LeaderboardNavigator registry={registry} />
          <article className="leaderboard-paper-article">
            <section className="leaderboard-paper-intro">
              <p className="leaderboard-paper-label">
                EvoPolicyGym · Leaderboard
              </p>
              <h1>{copy.title}</h1>
              <p className="leaderboard-paper-lead">{copy.description}</p>
            </section>

            <DistributionCollection
              number="1"
              id="distributions"
              title={copy.active}
              lead={copy.activeLead}
              items={active}
              language={language}
              labels={copy}
              initiallyOpen
            />

            {archived.length > 0 && (
              <DistributionCollection
                number="2"
                id="archive"
                title={copy.archive}
                lead={copy.archiveLead}
                items={archived}
                language={language}
                labels={copy}
              />
            )}
          </article>
        </div>
      </main>
    </Layout>
  );
}

function DistributionCollection({
  number,
  id,
  title,
  lead,
  items,
  language,
  labels,
  initiallyOpen = false,
}: {
  number: string;
  id: string;
  title: string;
  lead: string;
  items: LeaderboardRegistryItem[];
  language: "en" | "zh";
  labels: {
    environments: string;
    profile: string;
    open: string;
    archivedLabel: string;
    activeLabel: string;
  };
  initiallyOpen?: boolean;
}) {
  return (
    <section className="leaderboard-paper-section" id={id}>
      <header>
        <span>{number}</span>
        <div>
          <h2>{title}</h2>
          <p>{lead}</p>
        </div>
      </header>
      <div className="leaderboard-index-distributions">
        {items.map((item, index) => (
          <details
            className={`leaderboard-index-distribution${
              item.manifest.status === "archived" ? " is-archived" : ""
            }`}
            open={initiallyOpen && index === 0}
            key={item.manifest.id}
          >
            <summary>
              <span>
                {item.manifest.status === "archived"
                  ? labels.archivedLabel
                  : labels.activeLabel}
              </span>
              <strong>{pickLocalized(language, item.manifest.label)}</strong>
              <small>{pickLocalized(language, item.manifest.description)}</small>
              <i aria-hidden="true">+</i>
            </summary>
            <div className="leaderboard-index-distribution-body">
              <dl>
                <div>
                  <dt>{labels.environments}</dt>
                  <dd>{item.environments.length}</dd>
                </div>
                <div>
                  <dt>{labels.profile}</dt>
                  <dd>{pickLocalized(language, item.manifest.profile.label)}</dd>
                </div>
              </dl>
              <div className="leaderboard-index-environments">
                {item.environments.map((environment) => (
                  <Link
                    to={`${leaderboardPath(item.manifest)}environments/${environment.id}/`}
                    key={environment.id}
                  >
                    <span>{environment.display}</span>
                    <small>{environment.primary_metric}</small>
                    <b>→</b>
                  </Link>
                ))}
              </div>
              <Link
                className="leaderboard-index-open"
                to={leaderboardPath(item.manifest)}
              >
                {labels.open} →
              </Link>
            </div>
          </details>
        ))}
      </div>
    </section>
  );
}
