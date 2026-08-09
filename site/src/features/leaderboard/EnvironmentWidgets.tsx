import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import {type CSSProperties} from "react";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import {useLeaderboard, useLeaderboardEnvironment} from "./context";
import {LeaderboardSection} from "./DocumentComponents";
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
  const {suite, selectedConfiguration, selectConfiguration} = useLeaderboard();
  const environment = useLeaderboardEnvironment();
  const language = useSiteLanguage();
  const labels = useLeaderboardMessages();
  const configurationIds = environment.configuration_ids;
  if (configurationIds !== undefined && selectedConfiguration !== undefined) {
    const configurations = (suite.results.test_configurations ?? []).filter(
      (configuration) => configurationIds.includes(configuration.id),
    );
    const parameterLabels: Record<string, string> =
      language === "zh"
        ? {
            episode_budget: "训练 Episodes",
            max_submissions: "最大 Submissions",
            validation_episodes_per_candidate: "Validation Episodes",
            assessment_episodes: "Assessment Episodes",
            finish_budget_policy: "Finish Policy",
          }
        : {
            episode_budget: "Training Episodes",
            max_submissions: "Max Submissions",
            validation_episodes_per_candidate: "Validation Episodes",
            assessment_episodes: "Assessment Episodes",
            finish_budget_policy: "Finish Policy",
          };
    const summaryKeys = [
      "episode_budget",
      "max_submissions",
      "validation_episodes_per_candidate",
      "assessment_episodes",
    ];
    const summaryParameters = summaryKeys.flatMap((name) => {
      const value = selectedConfiguration.parameters[name];
      return value === undefined ? [] : [{name, value}];
    });
    const parameterName = (name: string) =>
      parameterLabels[name] ?? name.replaceAll("_", " ");
    const parameterValue = (value: string | number | boolean) =>
      typeof value === "string" ? value.replaceAll("_", " ") : String(value);
    return (
      <section className="leaderboard-experiment-setup">
        <div className="leaderboard-experiment-picker">
          <label htmlFor="leaderboard-test-configuration">
            <span>{labels.testConfiguration}</span>
            <small>{labels.selectConfiguration}</small>
          </label>
          <div className="leaderboard-experiment-select-control">
            <select
              id="leaderboard-test-configuration"
              value={selectedConfiguration.id}
              onChange={(event) => selectConfiguration?.(event.target.value)}
            >
              {configurations.map((configuration) => (
                <option value={configuration.id} key={configuration.id}>
                  {pickLocalized(language, configuration.label)}
                </option>
              ))}
            </select>
            <i aria-hidden="true">⌄</i>
          </div>
        </div>

        <article className="leaderboard-experiment-detail">
          <header>
            <div>
              <span>{labels.selectedConfiguration}</span>
              <h3>{pickLocalized(language, selectedConfiguration.label)}</h3>
            </div>
            <code>{selectedConfiguration.id}</code>
          </header>
          <p>{pickLocalized(language, selectedConfiguration.description)}</p>
          {summaryParameters.length > 0 && (
            <dl className="leaderboard-experiment-summary">
              {summaryParameters.map(({name, value}) => (
                <div key={name}>
                  <dt>{parameterName(name)}</dt>
                  <dd>{parameterValue(value)}</dd>
                </div>
              ))}
            </dl>
          )}
          <details>
            <summary>{labels.configurationDetails}</summary>
            <dl>
              {Object.entries(selectedConfiguration.parameters).map(
                ([name, value]) => (
                  <div key={name}>
                    <dt>{parameterName(name)}</dt>
                    <dd>{parameterValue(value)}</dd>
                  </div>
                ),
              )}
            </dl>
          </details>
        </article>
      </section>
    );
  }
  return (
    <p className="leaderboard-paper-section-profile">
      {pickLocalized(language, suite.manifest.profile.label)} · {suite.manifest.profile.budget} {suite.manifest.profile.budget_unit}
    </p>
  );
}

export function EnvironmentChart() {
  const {suite, selectedConfiguration} = useLeaderboard();
  const environment = useLeaderboardEnvironment();
  const scoreDecimalPlaces = environment.score_decimal_places ?? 3;
  const labels = useLeaderboardMessages();
  const scores = rankedEntries(suite, environment, selectedConfiguration?.id);
  const officialScores = scores.filter((entry) => entry.kind === "agent");
  const values = scores.map((entry) => entry.score);
  const scoreMinimum = Math.min(...values);
  const scoreMaximum = Math.max(...values);
  const scaleMinimum = Math.min(0, scoreMinimum);
  const scaleMaximum = Math.max(0, scoreMaximum);
  const scoreSpan = Math.max(scaleMaximum - scaleMinimum, 1e-9);
  const entryColors = ["#36a99b", "#f0765e", "#8a4cff", "#365a68", "#d29a2e"];

  function exportRanking() {
    const rankedAgents = scores.filter((entry) => entry.kind === "agent");
    const baselines = scores.filter((entry) => entry.kind === "baseline");
    const payload = {
      schema: "evopolicygym/environment-ranking/v1",
      generated_at: suite.results.generated_at,
      distribution_id: suite.manifest.id,
      environment_id: environment.id,
      test_configuration: selectedConfiguration ?? {
        id: suite.manifest.profile.id,
        label: suite.manifest.profile.label,
        parameters: {
          budget: suite.manifest.profile.budget,
          budget_unit: suite.manifest.profile.budget_unit,
        },
      },
      primary_metric: environment.primary_metric,
      score_direction: environment.score_direction,
      ranking: rankedAgents.map((entry, index) => ({
        rank: index + 1,
        entry_id: entry.id,
        display: entry.display,
        harness: entry.harness,
        thinking_effort: entry.thinking_effort,
        score: entry.score,
      })),
      baselines: baselines.map((entry) => ({
        entry_id: entry.id,
        display: entry.display,
        harness: entry.harness,
        score: entry.score,
      })),
    };
    const content = `${JSON.stringify(payload, null, 2)}\n`;
    const objectUrl = URL.createObjectURL(
      new Blob([content], {type: "application/json"}),
    );
    const link = document.createElement("a");
    const configurationId =
      selectedConfiguration?.id ?? suite.manifest.profile.id;
    link.href = objectUrl;
    link.download = `${suite.manifest.slug}-${environment.id}-${configurationId}-ranking.json`;
    document.body.append(link);
    try {
      link.click();
    } finally {
      link.remove();
      URL.revokeObjectURL(objectUrl);
    }
  }

  return (
    <div className="leaderboard-paper-figure leaderboard-score-figure">
      <div className="leaderboard-ranking-actions">
        <button type="button" onClick={exportRanking}>
          {labels.exportRanking}
        </button>
      </div>
      <div className="leaderboard-score-columns" aria-hidden="true">
        <span>{labels.rank}</span>
        <span>{labels.entry}</span>
        <span>{labels.scoreSignal}</span>
        <span>{labels.rawScore}</span>
      </div>
      <div className="leaderboard-paper-chart">
        {scores.map((entry) => {
          const isBaseline = entry.kind === "baseline";
          const rank = officialScores.findIndex((item) => item.id === entry.id);
          const progress = ((entry.score - scaleMinimum) / scoreSpan) * 100;
          const entryColor = isBaseline
            ? "#929a95"
            : entryColors[Math.max(rank, 0) % entryColors.length];
          const className = [
            isBaseline ? "is-baseline" : "",
            rank === 0 ? "is-first" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <div
              className={className}
              key={entry.id}
              style={{"--leaderboard-entry-color": entryColor} as CSSProperties}
            >
              <span className="leaderboard-paper-chart-rank">{isBaseline ? "—" : rank + 1}</span>
              <span className="leaderboard-paper-chart-label">
                <i aria-hidden="true" />
                <span>
                  <strong>
                    {entry.display}
                    {entry.thinking_effort !== undefined && (
                      <em>{entry.thinking_effort}</em>
                    )}
                  </strong>
                  <small>{entry.harness}</small>
                </span>
              </span>
              <span className="leaderboard-paper-chart-bar" aria-hidden="true"><i style={{width: `${Math.max(progress, 1.5)}%`}} /></span>
              <span className="leaderboard-paper-chart-score">{formatLeaderboardScore(entry.score, scoreDecimalPlaces)}</span>
            </div>
          );
        })}
        <div className="leaderboard-paper-axis" aria-hidden="true">
          <span>{formatLeaderboardScore(scaleMinimum, scoreDecimalPlaces)}</span>
          <span>{formatLeaderboardScore(scaleMinimum + scoreSpan / 2, scoreDecimalPlaces)}</span>
          <span>{formatLeaderboardScore(scaleMaximum, scoreDecimalPlaces)}</span>
        </div>
      </div>
    </div>
  );
}

export function EnvironmentTopPolicies() {
  const {suite, selectedConfiguration} = useLeaderboard();
  const environment = useLeaderboardEnvironment();
  const labels = useLeaderboardMessages();
  const baseUrl = useBaseUrl("/");
  const scoreDecimalPlaces = environment.score_decimal_places ?? 3;
  const entryColors = ["#36a99b", "#f0765e", "#8a4cff"];
  const topEntries = rankedEntries(
    suite,
    environment,
    selectedConfiguration?.id,
  )
    .filter((entry) => entry.kind === "agent")
    .slice(0, 3);
  const cards = topEntries.flatMap((entry, rank) => {
    const rollout = suite.results.policy_rollouts?.find(
      (item) =>
        item.entry_id === entry.id &&
        item.environment_id === environment.id &&
        item.configuration_id === selectedConfiguration?.id,
    );
    return rollout === undefined ? [] : [{entry, rank, rollout}];
  });
  if (cards.length === 0 || cards.length !== topEntries.length) return null;

  return (
    <LeaderboardSection
      number="3"
      title={labels.topPolicies}
      lead={labels.topPoliciesLead}
    >
      <div className="leaderboard-policy-rollouts">
        {cards.map(({entry, rank, rollout}) => {
          const artifact = `${baseUrl}${rollout.artifact.replace(/^\/+/, "")}`;
          const color = entryColors[rank % entryColors.length];
          return (
            <article
              className="leaderboard-policy-rollout"
              key={entry.id}
              style={{"--leaderboard-entry-color": color} as CSSProperties}
            >
              <header>
                <span>#{rank + 1}</span>
                <div>
                  <strong>{entry.display}</strong>
                  <small>
                    {entry.harness}
                    {entry.thinking_effort !== undefined &&
                      ` · ${entry.thinking_effort}`}
                  </small>
                </div>
                <b>{formatLeaderboardScore(entry.score, scoreDecimalPlaces)}</b>
              </header>
              <a
                className="leaderboard-policy-rollout-media"
                href={artifact}
                target="_blank"
                rel="noreferrer"
                aria-label={`${labels.openRollout}: ${entry.display}`}
              >
                <img
                  src={artifact}
                  alt={`${entry.display}, ${labels.assessmentEpisode} ${rollout.episode_index}`}
                  loading="lazy"
                  decoding="async"
                />
              </a>
              <footer>
                <span>
                  {labels.assessmentEpisode} {rollout.episode_index}
                  {rollout.camera !== undefined && ` · ${rollout.camera}`}
                </span>
                <span>
                  {labels.episodeScore} <strong>{formatLeaderboardScore(rollout.score, scoreDecimalPlaces)}</strong>
                </span>
              </footer>
            </article>
          );
        })}
      </div>
    </LeaderboardSection>
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
