import Link from "@docusaurus/Link";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import {useLeaderboard, useLeaderboardEnvironment} from "./context";
import {formatLeaderboardScore, rankedEntries, suitePath} from "./model";
import {useLeaderboardMessages} from "./messages";

export function EnvironmentHeader({eyebrow}: {eyebrow: string}) {
  const {suite} = useLeaderboard();
  const environment = useLeaderboardEnvironment();
  const language = useSiteLanguage();
  return (
    <>
      <p className="leaderboard-paper-label">
        {eyebrow} · {suite.manifest.experiment_version}
      </p>
      <h1>{environment.display}</h1>
      <p className="leaderboard-paper-lead">
        {pickLocalized(language, environment.summary)}
      </p>
    </>
  );
}

export function EnvironmentProfile() {
  const {suite} = useLeaderboard();
  const language = useSiteLanguage();
  return (
    <p className="leaderboard-paper-section-profile">
      {pickLocalized(language, suite.manifest.profile.label)} · {suite.manifest.profile.budget} {suite.manifest.profile.budget_unit}
    </p>
  );
}

export function EnvironmentChart() {
  const {suite} = useLeaderboard();
  const environment = useLeaderboardEnvironment();
  const labels = useLeaderboardMessages();
  const scores = rankedEntries(suite, environment);
  const officialScores = scores.filter((entry) => entry.kind === "agent");
  const baseline = scores.find((entry) => entry.kind === "baseline");
  const values = scores.map((entry) => entry.score);
  const scoreMinimum = Math.min(...values);
  const scoreMaximum = Math.max(...values);
  const scoreSpan = Math.max(scoreMaximum - scoreMinimum, 1e-9);

  return (
    <div className="leaderboard-paper-figure leaderboard-environment-figure">
      <div className="leaderboard-paper-chart">
        {scores.map((entry) => {
          const isBaseline = entry.kind === "baseline";
          const rank = officialScores.findIndex((item) => item.id === entry.id);
          const progress =
            environment.score_direction === "maximize"
              ? ((entry.score - scoreMinimum) / scoreSpan) * 100
              : ((scoreMaximum - entry.score) / scoreSpan) * 100;
          const delta = baseline
            ? environment.score_direction === "maximize"
              ? entry.score - baseline.score
              : baseline.score - entry.score
            : 0;
          return (
            <div className={isBaseline ? "is-baseline" : ""} key={entry.id}>
              <span className="leaderboard-paper-chart-rank">{isBaseline ? "—" : rank + 1}</span>
              <span className="leaderboard-paper-chart-label"><strong>{entry.display}</strong><small>{entry.harness}</small></span>
              <span className="leaderboard-paper-chart-score">{formatLeaderboardScore(entry.score)}</span>
              <span className="leaderboard-paper-chart-bar" aria-hidden="true"><i style={{width: `${Math.max(progress, 1.5)}%`}} /></span>
              <span className="leaderboard-paper-chart-delta">
                {isBaseline
                  ? labels.randomReference
                  : `${delta >= 0 ? "+" : ""}${formatLeaderboardScore(delta)} ${labels.versusRandom}`}
              </span>
            </div>
          );
        })}
        <div className="leaderboard-paper-axis" aria-hidden="true">
          <span>{formatLeaderboardScore(scoreMinimum)}</span>
          <span>{formatLeaderboardScore(scoreMinimum + scoreSpan / 2)}</span>
          <span>{formatLeaderboardScore(scoreMaximum)}</span>
        </div>
      </div>
    </div>
  );
}

export function EnvironmentPager({allLabel}: {allLabel: string}) {
  const {suite} = useLeaderboard();
  const environment = useLeaderboardEnvironment();
  const position = suite.results.environments.findIndex(
    (item) => item.id === environment.id,
  );
  const previous =
    suite.results.environments[
      (position - 1 + suite.results.environments.length) %
        suite.results.environments.length
    ];
  const next =
    suite.results.environments[(position + 1) % suite.results.environments.length];
  return (
    <nav className="leaderboard-environment-pager">
      <Link to={`${suitePath(suite)}environments/${previous.id}/`}>← {previous.display}</Link>
      <Link to={suitePath(suite)}>{allLabel}</Link>
      <Link to={`${suitePath(suite)}environments/${next.id}/`}>{next.display} →</Link>
    </nav>
  );
}
