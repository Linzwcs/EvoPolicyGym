import rawShowcase from "../data/showcase.json";

export interface Model {
  slug: string;
  display: string;
  harness: string;
  order: number;
}

export interface Environment {
  id: string;
  display: string;
  category: string;
  order: number;
  strategy_focus: boolean;
}

export interface Clip {
  id: string;
  mode: string;
  model_slug: string;
  model_display: string;
  model_order: number;
  harness: string;
  env_id: string;
  env_display: string;
  env_order: number;
  category: string;
  submit_index: number;
  score: number;
  best_so_far: number;
  score_improved: boolean;
  event_notes: string;
  media: string;
  media_source: string;
  capture_source: string;
  run_id: string;
  run_path: string;
  episode_id: number;
  episode_return: number;
  rerun_case_index: number;
  rerun_steps: number;
  trajectory_path: string;
  observations_path: string;
}

export interface FinalResult {
  model_slug: string;
  env_id: string;
  score: number;
  best_submit_index: number;
  run_id: string;
  run_path: string;
}

export interface LeaderboardRow {
  model_slug: string;
  model_display: string;
  harness: string;
  scores: Record<string, number>;
}

export interface Showcase {
  generated_at: string;
  mode: string;
  clip_policy: string;
  models: Model[];
  envs: Environment[];
  strategy_envs: string[];
  clips: Clip[];
  final_results: FinalResult[];
  leaderboard: LeaderboardRow[];
  warnings: string[];
}

export const showcase = rawShowcase as Showcase;

export const environments = [...showcase.envs].sort((a, b) => a.order - b.order);
export const models = [...showcase.models].sort((a, b) => a.order - b.order);

export function clipsForEnvironment(envId: string): Clip[] {
  return showcase.clips
    .filter((clip) => clip.env_id === envId)
    .sort((a, b) => a.model_order - b.model_order);
}

export function scoresForEnvironment(envId: string): LeaderboardRow[] {
  return showcase.leaderboard
    .map((row) => ({ ...row, score: row.scores[envId] }))
    .filter((row) => Number.isFinite(row.score))
    .sort((a, b) => b.score - a.score);
}

export function formatScore(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}
