import type {
  LeaderboardEntry,
  LeaderboardEnvironment,
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

export function suitePath(suite: LeaderboardSuiteData): string {
  return `/leaderboard/suites/${suite.manifest.slug}/`;
}

export function rankedEntries(
  suite: LeaderboardSuiteData,
  environment: LeaderboardEnvironment,
): ScoredEntry[] {
  const multiplier = environment.score_direction === "maximize" ? -1 : 1;
  return suite.results.entries
    .filter((entry) => Number.isFinite(entry.scores[environment.id]))
    .map((entry) => ({...entry, score: entry.scores[environment.id]}))
    .sort((left, right) => multiplier * (left.score - right.score));
}

export function aggregateEntries(
  suite: LeaderboardSuiteData,
): AggregateEntry[] {
  const agents = suite.results.entries.filter((entry) => entry.kind === "agent");
  return agents
    .map((entry) => {
      const placements = suite.results.environments.map((environment) => {
        const score = entry.scores[environment.id];
        return (
          1 +
          agents.filter((candidate) =>
            environment.score_direction === "maximize"
              ? candidate.scores[environment.id] > score
              : candidate.scores[environment.id] < score,
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

export function formatLeaderboardScore(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}
