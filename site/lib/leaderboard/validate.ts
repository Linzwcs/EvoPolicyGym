import path from "node:path";
import type {
  LeaderboardResults,
  LeaderboardSuiteManifest,
  LocalizedValue,
} from "./types";

export function validateManifest(
  value: Record<string, unknown>,
  filePath: string,
): LeaderboardSuiteManifest {
  if (value.schema !== "evopolicygym/leaderboard-suite/v2") {
    throw new Error(`${filePath}: unsupported leaderboard suite schema`);
  }
  for (const name of ["id", "slug", "version", "experiment_version", "results"]) {
    requireText(value[name], `${filePath}: ${name}`);
  }
  if (!isSuiteStatus(value.status)) {
    throw new Error(`${filePath}: invalid suite status`);
  }
  if (typeof value.default !== "boolean") {
    throw new Error(`${filePath}: default must be a boolean`);
  }
  for (const name of ["title", "label", "description"]) {
    validateLocalized(value[name], `${filePath}: ${name}`);
  }
  if (!isRecord(value.profile)) {
    throw new Error(`${filePath}: profile must be an object`);
  }
  requireText(value.profile.id, `${filePath}: profile.id`);
  validateLocalized(value.profile.label, `${filePath}: profile.label`);
  if (
    typeof value.profile.budget !== "number" ||
    !Number.isFinite(value.profile.budget) ||
    value.profile.budget <= 0
  ) {
    throw new Error(`${filePath}: profile.budget must be positive`);
  }
  requireText(value.profile.budget_unit, `${filePath}: profile.budget_unit`);

  if (!isRecord(value.content)) {
    throw new Error(`${filePath}: content must be an object`);
  }
  for (const name of ["suite", "environment"]) {
    validateLocalized(value.content[name], `${filePath}: content.${name}`);
  }
  return value as unknown as LeaderboardSuiteManifest;
}

export function validateResults(value: unknown, filePath: string): LeaderboardResults {
  if (!isRecord(value)) throw new Error(`${filePath}: results must be an object`);
  if (value.schema !== "evopolicygym/leaderboard-results/v1") {
    throw new Error(`${filePath}: unsupported leaderboard results schema`);
  }
  requireText(value.generated_at, `${filePath}: generated_at`);
  if (!Number.isInteger(value.final_reruns) || Number(value.final_reruns) < 0) {
    throw new Error(`${filePath}: final_reruns must be a non-negative integer`);
  }
  if (!Array.isArray(value.environments) || value.environments.length === 0) {
    throw new Error(`${filePath}: environments must not be empty`);
  }
  if (!Array.isArray(value.entries) || value.entries.length === 0) {
    throw new Error(`${filePath}: entries must not be empty`);
  }

  const environmentIds = new Set<string>();
  for (const item of value.environments) {
    if (!isRecord(item)) throw new Error(`${filePath}: invalid environment`);
    for (const name of ["id", "display", "category", "primary_metric"]) {
      requireText(item[name], `${filePath}: environment.${name}`);
    }
    validateLocalized(item.summary, `${filePath}: environment.summary`);
    if (environmentIds.has(String(item.id))) {
      throw new Error(`${filePath}: duplicate environment ${String(item.id)}`);
    }
    environmentIds.add(String(item.id));
    if (!Number.isInteger(item.order) || Number(item.order) < 0) {
      throw new Error(`${filePath}: environment.order must be non-negative`);
    }
    if (!isScoreDirection(item.score_direction)) {
      throw new Error(`${filePath}: invalid score direction`);
    }
  }

  const entryIds = new Set<string>();
  for (const item of value.entries) {
    if (!isRecord(item)) throw new Error(`${filePath}: invalid entry`);
    for (const name of ["id", "display", "harness"]) {
      requireText(item[name], `${filePath}: entry.${name}`);
    }
    if (entryIds.has(String(item.id))) {
      throw new Error(`${filePath}: duplicate entry ${String(item.id)}`);
    }
    entryIds.add(String(item.id));
    if (!isEntryKind(item.kind)) {
      throw new Error(`${filePath}: invalid entry kind`);
    }
    if (!isRecord(item.scores)) {
      throw new Error(`${filePath}: entry.scores must be an object`);
    }
    for (const [environmentId, score] of Object.entries(item.scores)) {
      if (!environmentIds.has(environmentId)) {
        throw new Error(`${filePath}: score references unknown ${environmentId}`);
      }
      if (typeof score !== "number" || !Number.isFinite(score)) {
        throw new Error(`${filePath}: score for ${environmentId} must be finite`);
      }
    }
  }
  return value as unknown as LeaderboardResults;
}

export function validateSuiteCoverage(
  manifest: LeaderboardSuiteManifest,
  results: LeaderboardResults,
  filePath: string,
): void {
  const required = new Set(results.environments.map((environment) => environment.id));
  for (const entry of results.entries) {
    const missing = [...required].filter(
      (environmentId) => !Object.hasOwn(entry.scores, environmentId),
    );
    if (entry.kind === "agent" && missing.length > 0) {
      throw new Error(
        `${filePath}: ${manifest.id}/${entry.id} is missing ${missing.join(", ")}`,
      );
    }
  }
}

export function resolveSuiteFile(
  suiteDirectory: string,
  relativePath: string,
  label: string,
): string {
  const resolved = path.resolve(suiteDirectory, relativePath);
  const directoryPrefix = `${path.resolve(suiteDirectory)}${path.sep}`;
  if (!resolved.startsWith(directoryPrefix)) {
    throw new Error(`${label}: path must remain inside its suite directory`);
  }
  return resolved;
}

export function validateContentExtension(filePath: string): void {
  if (!/\.mdx?$/i.test(filePath)) {
    throw new Error(`${filePath}: leaderboard content must be Markdown or MDX`);
  }
}

function validateLocalized(value: unknown, label: string): LocalizedValue {
  if (!isRecord(value)) throw new Error(`${label} must be an object`);
  requireText(value.en, `${label}.en`);
  requireText(value.zh, `${label}.zh`);
  return value as unknown as LocalizedValue;
}

function requireText(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} must be non-empty text`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSuiteStatus(
  value: unknown,
): value is LeaderboardSuiteManifest["status"] {
  return ["draft", "active", "frozen", "archived"].includes(String(value));
}

function isScoreDirection(value: unknown): value is "maximize" | "minimize" {
  return ["maximize", "minimize"].includes(String(value));
}

function isEntryKind(value: unknown): value is "agent" | "baseline" {
  return ["agent", "baseline"].includes(String(value));
}
