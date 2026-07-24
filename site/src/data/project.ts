export const projectMeta = {
  name: "EvoPolicyGym",
  packageVersion: "0.3.0",
  versionLabel: "v0.3",
  versionSeries: "0.3",
  docsChannel: "v0.3",
  protocolVersion: "policy/v1",
  protocolLabelEn: "Policy ABI v1",
  protocolLabelZh: "Policy ABI v1",
  developmentStageEn: "Active alpha",
  developmentStageZh: "活跃 Alpha",
} as const;

export type DocsStatus = "planning" | "draft" | "stable" | "historical";

export const docsStatusLabels: Record<
  DocsStatus,
  { en: string; zh: string }
> = {
  planning: { en: "Structure planning", zh: "结构规划" },
  draft: { en: "Draft documentation", zh: "文档草案" },
  stable: { en: "Stable documentation", zh: "稳定文档" },
  historical: { en: "Historical record", zh: "历史记录" },
};
