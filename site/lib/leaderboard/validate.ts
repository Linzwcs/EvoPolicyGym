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

  const configurationIds = new Set<string>();
  if (value.test_configurations !== undefined) {
    if (
      !Array.isArray(value.test_configurations) ||
      value.test_configurations.length === 0
    ) {
      throw new Error(`${filePath}: test_configurations must not be empty`);
    }
    for (const item of value.test_configurations) {
      if (!isRecord(item)) {
        throw new Error(`${filePath}: invalid test configuration`);
      }
      requireText(item.id, `${filePath}: test_configuration.id`);
      if (configurationIds.has(item.id)) {
        throw new Error(`${filePath}: duplicate test configuration ${item.id}`);
      }
      configurationIds.add(item.id);
      validateLocalized(item.label, `${filePath}: test_configuration.label`);
      validateLocalized(
        item.description,
        `${filePath}: test_configuration.description`,
      );
      if (!isRecord(item.parameters)) {
        throw new Error(
          `${filePath}: test_configuration.parameters must be an object`,
        );
      }
      for (const [name, parameter] of Object.entries(item.parameters)) {
        requireText(name, `${filePath}: test_configuration parameter name`);
        if (
          typeof parameter !== "string" &&
          typeof parameter !== "boolean" &&
          (typeof parameter !== "number" || !Number.isFinite(parameter))
        ) {
          throw new Error(
            `${filePath}: test_configuration parameter ${name} is invalid`,
          );
        }
      }
    }
  }

  const environmentIds = new Set<string>();
  const environmentConfigurationIds = new Map<string, Set<string>>();
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
    if (
      item.score_decimal_places !== undefined &&
      (!Number.isInteger(item.score_decimal_places) ||
        Number(item.score_decimal_places) < 0 ||
        Number(item.score_decimal_places) > 12)
    ) {
      throw new Error(
        `${filePath}: environment.score_decimal_places must be an integer from 0 to 12`,
      );
    }
    const configured = item.configuration_ids !== undefined;
    if (configured !== (item.default_configuration_id !== undefined)) {
      throw new Error(
        `${filePath}: environment configuration fields must be declared together`,
      );
    }
    if (configured) {
      if (
        !Array.isArray(item.configuration_ids) ||
        item.configuration_ids.length === 0
      ) {
        throw new Error(
          `${filePath}: environment.configuration_ids must not be empty`,
        );
      }
      requireText(
        item.default_configuration_id,
        `${filePath}: environment.default_configuration_id`,
      );
      const selectedIds = new Set<string>();
      for (const configurationId of item.configuration_ids) {
        requireText(
          configurationId,
          `${filePath}: environment.configuration_id`,
        );
        if (!configurationIds.has(configurationId)) {
          throw new Error(
            `${filePath}: environment references unknown configuration ${configurationId}`,
          );
        }
        if (selectedIds.has(configurationId)) {
          throw new Error(
            `${filePath}: environment repeats configuration ${configurationId}`,
          );
        }
        selectedIds.add(configurationId);
      }
      if (!selectedIds.has(item.default_configuration_id)) {
        throw new Error(
          `${filePath}: environment default configuration is not selected`,
        );
      }
      environmentConfigurationIds.set(String(item.id), selectedIds);
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
    if (item.thinking_effort !== undefined) {
      requireText(item.thinking_effort, `${filePath}: entry.thinking_effort`);
    }
    if (!isRecord(item.scores)) {
      throw new Error(`${filePath}: entry.scores must be an object`);
    }
    for (const [environmentId, score] of Object.entries(item.scores)) {
      if (!environmentIds.has(environmentId)) {
        throw new Error(`${filePath}: score references unknown ${environmentId}`);
      }
      const selectedConfigurationIds = environmentConfigurationIds.get(environmentId);
      if (selectedConfigurationIds === undefined) {
        if (typeof score !== "number" || !Number.isFinite(score)) {
          throw new Error(`${filePath}: score for ${environmentId} must be finite`);
        }
        continue;
      }
      if (!isRecord(score)) {
        throw new Error(
          `${filePath}: configured score for ${environmentId} must be an object`,
        );
      }
      for (const [configurationId, configuredScore] of Object.entries(score)) {
        if (!selectedConfigurationIds.has(configurationId)) {
          throw new Error(
            `${filePath}: score references unavailable configuration ${configurationId}`,
          );
        }
        if (
          typeof configuredScore !== "number" ||
          !Number.isFinite(configuredScore)
        ) {
          throw new Error(
            `${filePath}: score for ${environmentId}/${configurationId} must be finite`,
          );
        }
      }
    }
  }
  if (value.policy_rollouts !== undefined) {
    if (!Array.isArray(value.policy_rollouts)) {
      throw new Error(`${filePath}: policy_rollouts must be an array`);
    }
    const rolloutKeys = new Set<string>();
    for (const item of value.policy_rollouts) {
      if (!isRecord(item)) {
        throw new Error(`${filePath}: invalid policy rollout`);
      }
      for (const name of ["entry_id", "environment_id", "artifact", "split"]) {
        requireText(item[name], `${filePath}: policy_rollout.${name}`);
      }
      if (!entryIds.has(String(item.entry_id))) {
        throw new Error(
          `${filePath}: policy rollout references unknown entry ${String(item.entry_id)}`,
        );
      }
      const environmentId = String(item.environment_id);
      if (!environmentIds.has(environmentId)) {
        throw new Error(
          `${filePath}: policy rollout references unknown environment ${environmentId}`,
        );
      }
      const selectedConfigurationIds = environmentConfigurationIds.get(environmentId);
      if (selectedConfigurationIds === undefined) {
        if (item.configuration_id !== undefined) {
          throw new Error(
            `${filePath}: unconfigured policy rollout cannot select a configuration`,
          );
        }
      } else {
        requireText(
          item.configuration_id,
          `${filePath}: policy_rollout.configuration_id`,
        );
        if (!selectedConfigurationIds.has(item.configuration_id)) {
          throw new Error(
            `${filePath}: policy rollout references unavailable configuration ${item.configuration_id}`,
          );
        }
      }
      if (
        !String(item.artifact).startsWith("/") ||
        String(item.artifact).includes("..")
      ) {
        throw new Error(
          `${filePath}: policy rollout artifact must be a safe site-absolute path`,
        );
      }
      if (item.media_type !== "image/gif") {
        throw new Error(`${filePath}: policy rollout media_type must be image/gif`);
      }
      if (item.camera !== undefined) {
        requireText(item.camera, `${filePath}: policy_rollout.camera`);
      }
      if (!Number.isInteger(item.episode_index) || Number(item.episode_index) < 0) {
        throw new Error(
          `${filePath}: policy rollout episode_index must be non-negative`,
        );
      }
      if (typeof item.score !== "number" || !Number.isFinite(item.score)) {
        throw new Error(`${filePath}: policy rollout score must be finite`);
      }
      const rolloutKey = [
        item.entry_id,
        environmentId,
        item.configuration_id ?? "",
      ].join("\0");
      if (rolloutKeys.has(rolloutKey)) {
        throw new Error(`${filePath}: duplicate policy rollout ${rolloutKey}`);
      }
      rolloutKeys.add(rolloutKey);
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
    if (entry.kind !== "agent") continue;
    for (const environment of results.environments) {
      const configurationIds = environment.configuration_ids;
      if (configurationIds === undefined) continue;
      const configuredScores = entry.scores[environment.id];
      if (
        typeof configuredScores === "number" ||
        configuredScores === undefined
      ) {
        throw new Error(
          `${filePath}: ${manifest.id}/${entry.id} has no configured scores for ${environment.id}`,
        );
      }
      const missingConfigurations = configurationIds.filter(
        (configurationId) => !Object.hasOwn(configuredScores, configurationId),
      );
      if (missingConfigurations.length > 0) {
        throw new Error(
          `${filePath}: ${manifest.id}/${entry.id}/${environment.id} is missing ${missingConfigurations.join(", ")}`,
        );
      }
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
