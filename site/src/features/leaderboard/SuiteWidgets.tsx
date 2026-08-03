import Link from "@docusaurus/Link";
import {aggregateEntries, suitePath} from "./model";
import {useLeaderboard} from "./context";
import {useLeaderboardMessages} from "./messages";

export function SuiteFacts() {
  const {suite} = useLeaderboard();
  const labels = useLeaderboardMessages();
  const aggregateRows = aggregateEntries(suite);
  return (
    <dl className="leaderboard-paper-facts">
      <div><dt>{labels.tracks}</dt><dd>{suite.results.environments.length}</dd></div>
      <div><dt>{labels.agentEntries}</dt><dd>{aggregateRows.length}</dd></div>
      <div><dt>{labels.finalReruns}</dt><dd>{suite.results.final_reruns}</dd></div>
      <div><dt>{labels.profile}</dt><dd>{suite.manifest.profile.budget}<small>{suite.manifest.profile.budget_unit}</small></dd></div>
    </dl>
  );
}

export function AggregateTable() {
  const {suite} = useLeaderboard();
  const labels = useLeaderboardMessages();
  const aggregateRows = aggregateEntries(suite);
  return (
    <div className="leaderboard-paper-table" role="table">
      <div className="leaderboard-paper-table-head" role="row">
        <span role="columnheader">#</span>
        <span role="columnheader">{labels.agentModel}</span>
        <span role="columnheader">{labels.trackWins}</span>
        <span role="columnheader">{labels.podiums}</span>
        <span role="columnheader">{labels.meanRank}</span>
      </div>
      {aggregateRows.map((row, index) => (
        <div className="leaderboard-paper-table-row" role="row" key={row.id}>
          <span className="leaderboard-paper-rank" role="cell">{index + 1}</span>
          <span className="leaderboard-paper-agent" role="cell"><strong>{row.display}</strong><small>{row.harness}</small></span>
          <span className="leaderboard-paper-win-cell" role="cell"><b>{row.wins}</b><i aria-hidden="true"><em style={{width: `${(row.wins / suite.results.environments.length) * 100}%`}} /></i></span>
          <span role="cell">{row.podiums}</span>
          <span role="cell">{row.averageRank.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export function EnvironmentDirectory() {
  const {suite} = useLeaderboard();
  const categories = Array.from(
    new Set(suite.results.environments.map((environment) => environment.category)),
  );
  return (
    <div className="leaderboard-environment-directory">
      {categories.map((category) => {
        const environments = suite.results.environments.filter(
          (environment) => environment.category === category,
        );
        return (
          <section key={category}>
            <header><h3>{category}</h3><span>{environments.length}</span></header>
            {environments.map((environment) => (
              <Link
                to={`${suitePath(suite)}environments/${environment.id}/`}
                key={environment.id}
              >
                <span>{String(environment.order + 1).padStart(2, "0")}</span>
                <strong>{environment.display}</strong>
                <small>{environment.primary_metric}</small>
                <b>→</b>
              </Link>
            ))}
          </section>
        );
      })}
    </div>
  );
}
