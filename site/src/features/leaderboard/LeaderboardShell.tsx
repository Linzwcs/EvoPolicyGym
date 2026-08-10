import Link from "@docusaurus/Link";
import type {ReactNode} from "react";
import {useState} from "react";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import type {
  LeaderboardRegistry,
  LeaderboardRegistryItem,
  LeaderboardSuiteData,
} from "../../../lib/leaderboard/types";
import {leaderboardPath} from "./model";
import {useLeaderboardMessages} from "./messages";

export function LeaderboardShell({
  suite,
  registry,
  currentEnvironmentId,
  children,
}: {
  suite: LeaderboardSuiteData;
  registry: LeaderboardRegistry;
  currentEnvironmentId?: string;
  children: ReactNode;
}) {
  return (
    <main className="leaderboard-paper">
      <div className="leaderboard-paper-shell">
        <LeaderboardNavigator
          registry={registry}
          currentSuiteId={suite.manifest.id}
          currentEnvironmentId={currentEnvironmentId}
        />
        <article className="leaderboard-paper-article">{children}</article>
        <LeaderboardPublicationRail />
      </div>
    </main>
  );
}

export function LeaderboardPublicationRail() {
  return (
    <aside className="leaderboard-publication-rail" aria-hidden="true">
      <span>EPG / 03</span>
      <i />
      <strong>Open evaluation records</strong>
      <small>Research index · 2026</small>
    </aside>
  );
}

export function LeaderboardNavigator({
  registry,
  currentSuiteId,
  currentEnvironmentId,
}: {
  registry: LeaderboardRegistry;
  currentSuiteId?: string;
  currentEnvironmentId?: string;
}) {
  const language = useSiteLanguage();
  const labels = useLeaderboardMessages();
  const [query, setQuery] = useState("");
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const [expandedDistributions, setExpandedDistributions] = useState<string[]>(
    [currentSuiteId ?? registry.defaultSuiteId],
  );
  const distributions = registry.suites.filter(
    (item) => item.manifest.status !== "archived",
  );
  const archives = registry.suites.filter(
    (item) => item.manifest.status === "archived",
  );

  function toggleDistribution(distributionId: string) {
    setExpandedDistributions((current) =>
      current.includes(distributionId)
        ? current.filter((item) => item !== distributionId)
        : [...current, distributionId],
    );
  }

  function matchingEnvironments(item: LeaderboardRegistryItem) {
    const distributionMatches = pickLocalized(language, item.manifest.label)
      .toLocaleLowerCase()
      .includes(normalizedQuery);
    return item.environments.filter(
      (environment) =>
        !normalizedQuery ||
        distributionMatches ||
        environment.display.toLocaleLowerCase().includes(normalizedQuery) ||
        environment.id.toLocaleLowerCase().includes(normalizedQuery),
    );
  }

  function distributionGroup(
    title: string,
    items: LeaderboardRegistryItem[],
  ) {
    const visibleItems = items.filter(
      (item) => matchingEnvironments(item).length > 0,
    );
    if (visibleItems.length === 0) return null;
    return (
      <nav className="leaderboard-sidebar-environments">
        <p>
          {title}
          <span>{visibleItems.length}</span>
        </p>
        {visibleItems.map((item) => {
          const environments = matchingEnvironments(item);
          const isCurrent = item.manifest.id === currentSuiteId;
          const isExpanded =
            normalizedQuery.length > 0 ||
            expandedDistributions.includes(item.manifest.id);
          const basePath = leaderboardPath(item.manifest);
          return (
            <div
              className={`leaderboard-sidebar-category leaderboard-sidebar-distribution${
                isExpanded ? " is-expanded" : ""
              }${isCurrent ? " is-current" : ""}`}
              key={item.manifest.id}
            >
              <button
                type="button"
                aria-expanded={isExpanded}
                onClick={() => toggleDistribution(item.manifest.id)}
              >
                <strong>{pickLocalized(language, item.manifest.label)}</strong>
                <span>{item.environments.length}</span>
                <i aria-hidden="true">+</i>
              </button>
              {isExpanded && (
                <div className="leaderboard-sidebar-category-links">
                  {environments.map((environment) => (
                    <Link
                      className={
                        isCurrent && currentEnvironmentId === environment.id
                          ? "is-active"
                          : ""
                      }
                      to={`${basePath}environments/${environment.id}/`}
                      key={`${item.manifest.id}:${environment.id}`}
                    >
                      {environment.display}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    );
  }

  return (
    <aside
      className="leaderboard-paper-toc leaderboard-sidebar"
      aria-label={labels.navigationAria}
    >
      <div className="leaderboard-sidebar-suite">
        <Link
          className={!currentSuiteId ? "is-active" : ""}
          to="/leaderboard/"
        >
          {labels.leaderboard}
        </Link>
        <span className="leaderboard-sidebar-status">
          <i aria-hidden="true" />
          {language === "zh" ? "公开研究索引" : "Public research index"}
        </span>
      </div>

      <div className="leaderboard-sidebar-search">
        <label htmlFor="leaderboard-environment-search">
          {labels.findEnvironment}
        </label>
        <input
          id="leaderboard-environment-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={labels.searchPlaceholder}
        />
      </div>

      {distributionGroup(labels.distributions, distributions)}
      {distributionGroup(labels.archive, archives)}
    </aside>
  );
}
