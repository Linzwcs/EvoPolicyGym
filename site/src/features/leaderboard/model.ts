import type {
  LeaderboardEntry,
  LeaderboardEnvironment,
  LeaderboardSuiteManifest,
  LeaderboardSuiteData,
} from "../../../lib/leaderboard/types";

export interface ScoredEntry extends LeaderboardEntry {
  score: number;
}

export interface AggregateEntry extends LeaderboardEntry {
  wins: number;
  podiums: number;
  averageRank: number;
}

export function leaderboardPath(manifest: LeaderboardSuiteManifest): string {
  const collection =
    manifest.status === "archived" ? "archive" : "distributions";
  return `/leaderboard/${collection}/${manifest.slug}/`;
}

export function suitePath(suite: LeaderboardSuiteData): string {
  return leaderboardPath(suite.manifest);
}

export function rankedEntries(
  suite: LeaderboardSuiteData,
  environment: LeaderboardEnvironment,
  configurationId: string | undefined = environment.default_configuration_id,
): ScoredEntry[] {
  const multiplier = environment.score_direction === "maximize" ? -1 : 1;
  return suite.results.entries
    .map((entry) => {
      const score = scoreForEntry(entry, environment, configurationId);
      return score === undefined ? null : {...entry, score};
    })
    .filter((entry): entry is ScoredEntry => entry !== null)
    .sort((left, right) => multiplier * (left.score - right.score));
}

export function scoreForEntry(
  entry: LeaderboardEntry,
  environment: LeaderboardEnvironment,
  configurationId: string | undefined = environment.default_configuration_id,
): number | undefined {
  const score = entry.scores[environment.id];
  if (typeof score === "number") return score;
  if (score === undefined || configurationId === undefined) return undefined;
  const configuredScore = score[configurationId];
  return typeof configuredScore === "number" ? configuredScore : undefined;
}

export function aggregateEntries(
  suite: LeaderboardSuiteData,
): AggregateEntry[] {
  const agents = suite.results.entries.filter((entry) => entry.kind === "agent");
  return agents
    .map((entry) => {
      const placements = suite.results.environments.map((environment) => {
        const score = scoreForEntry(entry, environment);
        if (score === undefined) return suite.results.entries.length + 1;
        return (
          1 +
          agents.filter((candidate) =>
            environment.score_direction === "maximize"
              ? (scoreForEntry(candidate, environment) ?? -Infinity) > score
              : (scoreForEntry(candidate, environment) ?? Infinity) < score,
          ).length
        );
      });
      return {
        ...entry,
        wins: placements.filter((rank) => rank === 1).length,
        podiums: placements.filter((rank) => rank <= 3).length,
        averageRank:
          placements.reduce((total, rank) => total + rank, 0) /
          placements.length,
      };
    })
    .sort((left, right) =>
      right.wins - left.wins || left.averageRank - right.averageRank,
    );
}

export function formatLeaderboardScore(
  value: number,
  decimalPlaces = 3,
): string {
  return value.toFixed(decimalPlaces);
}
