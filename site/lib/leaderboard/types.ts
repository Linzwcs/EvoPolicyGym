export interface LocalizedValue<T = string> {
  en: T;
  zh: T;
}

export interface LeaderboardSuiteManifest {
  schema: "evopolicygym/leaderboard-suite/v2";
  id: string;
  slug: string;
  status: "draft" | "active" | "frozen" | "archived";
  default: boolean;
  version: string;
  experiment_version: string;
  title: LocalizedValue;
  label: LocalizedValue;
  description: LocalizedValue;
  profile: {
    id: string;
    label: LocalizedValue;
    budget: number;
    budget_unit: string;
  };
  content: {
    suite: LocalizedValue;
    environment: LocalizedValue;
  };
  results: string;
}

export interface LeaderboardEnvironment {
  id: string;
  display: string;
  category: string;
  order: number;
  primary_metric: string;
  score_direction: "maximize" | "minimize";
  summary: LocalizedValue;
  configuration_ids?: string[];
  default_configuration_id?: string;
  evidence_path?: string;
}

export type LeaderboardConfigurationParameter = string | number | boolean;

export interface LeaderboardTestConfiguration {
  id: string;
  label: LocalizedValue;
  description: LocalizedValue;
  parameters: Record<string, LeaderboardConfigurationParameter>;
}

export type LeaderboardScore = number | Record<string, number>;

export interface LeaderboardEntry {
  id: string;
  display: string;
  harness: string;
  kind: "agent" | "baseline";
  thinking_effort?: string;
  scores: Record<string, LeaderboardScore>;
}

export interface LeaderboardResults {
  schema: "evopolicygym/leaderboard-results/v1";
  generated_at: string;
  final_reruns: number;
  test_configurations?: LeaderboardTestConfiguration[];
  environments: LeaderboardEnvironment[];
  entries: LeaderboardEntry[];
}

export interface LeaderboardSuiteData {
  manifest: LeaderboardSuiteManifest;
  results: LeaderboardResults;
}

export interface LeaderboardRegistryItem {
  manifest: LeaderboardSuiteManifest;
  environments: LeaderboardEnvironment[];
}

export interface LeaderboardRegistry {
  defaultSuiteId: string;
  suites: LeaderboardRegistryItem[];
}
