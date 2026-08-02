import Link from "@docusaurus/Link";
import type {ReactNode} from "react";
import {useState} from "react";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import type {
  LeaderboardRegistry,
  LeaderboardSuiteData,
} from "../../../lib/leaderboard/types";
import {suitePath} from "./model";
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
        <LeaderboardSidebar
          suite={suite}
          registry={registry}
          currentEnvironmentId={currentEnvironmentId}
        />
        <article className="leaderboard-paper-article">{children}</article>
      </div>
    </main>
  );
}

function LeaderboardSidebar({
  suite,
  registry,
  currentEnvironmentId,
}: {
  suite: LeaderboardSuiteData;
  registry: LeaderboardRegistry;
  currentEnvironmentId?: string;
}) {
  const language = useSiteLanguage();
  const labels = useLeaderboardMessages();
  const [query, setQuery] = useState("");
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const categories = Array.from(
    new Set(suite.results.environments.map((environment) => environment.category)),
  );
  const activeCategory = suite.results.environments.find(
    (environment) => environment.id === currentEnvironmentId,
  )?.category;
  const [expandedCategories, setExpandedCategories] = useState<string[]>(
    activeCategory ? [activeCategory] : [],
  );
  const basePath = suitePath(suite);

  function toggleCategory(category: string) {
    setExpandedCategories((current) =>
      current.includes(category)
        ? current.filter((item) => item !== category)
        : [...current, category],
    );
  }

  return (
    <aside
      className="leaderboard-paper-toc leaderboard-sidebar"
      aria-label={labels.navigationAria}
    >
      <div className="leaderboard-sidebar-suite">
        <strong>{labels.leaderboard}</strong>
      </div>

      <nav className="leaderboard-sidebar-current">
        <p>{labels.currentSuite}</p>
        <Link className={!currentEnvironmentId ? "is-active" : ""} to={basePath}>
          {pickLocalized(language, suite.manifest.label)}
        </Link>
      </nav>

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

      <nav className="leaderboard-sidebar-environments">
        <p>
          {labels.byEnvironment}
          <span>{suite.results.environments.length}</span>
        </p>
        {categories.map((category) => {
          const environments = suite.results.environments.filter(
            (environment) =>
              environment.category === category &&
              (!normalizedQuery ||
                environment.display.toLocaleLowerCase().includes(normalizedQuery) ||
                environment.id.toLocaleLowerCase().includes(normalizedQuery)),
          );
          if (environments.length === 0) return null;
          const isExpanded =
            normalizedQuery.length > 0 || expandedCategories.includes(category);
          return (
            <div
              className={`leaderboard-sidebar-category${isExpanded ? " is-expanded" : ""}`}
              key={category}
            >
              <button
                type="button"
                aria-expanded={isExpanded}
                onClick={() => toggleCategory(category)}
              >
                <strong>{category}</strong>
                <span>{environments.length}</span>
                <i aria-hidden="true">+</i>
              </button>
              {isExpanded && (
                <div className="leaderboard-sidebar-category-links">
                  {environments.map((environment) => (
                    <Link
                      className={
                        currentEnvironmentId === environment.id ? "is-active" : ""
                      }
                      to={`${basePath}environments/${environment.id}/`}
                      key={environment.id}
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

      {registry.suites.length > 1 && (
        <nav className="leaderboard-sidebar-archives">
          <p>{labels.suites}</p>
          {registry.suites.map((item) => (
            <Link
              className={item.manifest.id === suite.manifest.id ? "is-current" : ""}
              to={`/leaderboard/suites/${item.manifest.slug}/`}
              key={item.manifest.id}
            >
              {pickLocalized(language, item.manifest.label)}
            </Link>
          ))}
        </nav>
      )}
    </aside>
  );
}
