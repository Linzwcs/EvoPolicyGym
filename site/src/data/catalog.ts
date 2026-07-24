export type CatalogStatus = "active" | "incubating" | "historical";
export type CatalogTone = "active" | "planned" | "research";

export interface CatalogEnvironment {
  id: string;
  index: string;
  title: string;
  eyebrowEn: string;
  eyebrowZh: string;
  descriptionEn: string;
  descriptionZh: string;
  status: CatalogStatus;
  statusEn: string;
  statusZh: string;
  tone: CatalogTone;
  visual: "cartpole" | "balatro" | "core16";
  facts: Array<{
    labelEn: string;
    labelZh: string;
    valueEn: string;
    valueZh: string;
  }>;
  actionEn: string;
  actionZh: string;
  actionPath: string;
}

export const catalogEnvironments: CatalogEnvironment[] = [
  {
    id: "cartpole",
    index: "ENV / 001",
    title: "CartPole",
    eyebrowEn: "Reference benchmark",
    eyebrowZh: "参考 Benchmark",
    descriptionEn:
      "A small, reproducible control task for exercising the complete Policy-evolution lifecycle.",
    descriptionZh:
      "一个轻量、可复现的控制任务，用来验证完整的 Policy 演化生命周期。",
    status: "active",
    statusEn: "Implemented",
    statusZh: "已实现",
    tone: "active",
    visual: "cartpole",
    facts: [
      {
        labelEn: "Observation",
        labelZh: "Observation",
        valueEn: "4 finite floats",
        valueZh: "4 个有限浮点数",
      },
      {
        labelEn: "Action",
        labelZh: "Action",
        valueEn: "Left / right",
        valueZh: "左 / 右",
      },
      {
        labelEn: "Objective",
        labelZh: "目标",
        valueEn: "Mean return · max 500",
        valueZh: "平均回报 · 最高 500",
      },
    ],
    actionEn: "Open CartPole",
    actionZh: "查看 CartPole",
    actionPath: "environments/#cartpole",
  },
  {
    id: "balatro",
    index: "ENV / 002",
    title: "Balatro",
    eyebrowEn: "Game benchmark",
    eyebrowZh: "游戏 Benchmark",
    descriptionEn:
      "A long-horizon deckbuilding environment where an Agent must author strategy across hands, shops, Jokers, and antes.",
    descriptionZh:
      "一个长时程牌组构筑环境，Agent 需要为出牌、商店、Joker 与 Ante 编写完整策略。",
    status: "incubating",
    statusEn: "In development",
    statusZh: "开发中",
    tone: "planned",
    visual: "balatro",
    facts: [
      {
        labelEn: "Observation",
        labelZh: "Observation",
        valueEn: "Structured game state",
        valueZh: "结构化游戏状态",
      },
      {
        labelEn: "Action",
        labelZh: "Action",
        valueEn: "Play / discard / shop",
        valueZh: "出牌 / 弃牌 / 商店",
      },
      {
        labelEn: "Runtime",
        labelZh: "运行要求",
        valueEn: "Licensed game + bridge",
        valueZh: "正版游戏 + Bridge",
      },
    ],
    actionEn: "See integration plan",
    actionZh: "查看接入计划",
    actionPath: "environments/#balatro",
  },
  {
    id: "core16",
    index: "ENV / H16",
    title: "Core16",
    eyebrowEn: "Research archive",
    eyebrowZh: "研究档案",
    descriptionEn:
      "Sixteen control, navigation, driving, and robotics tasks preserved with the paper's scores and final-policy reruns.",
    descriptionZh:
      "论文中的 16 个控制、导航、驾驶与机器人任务，保留完整分数和最终 Policy 重跑。",
    status: "historical",
    statusEn: "Historical",
    statusZh: "历史结果",
    tone: "research",
    visual: "core16",
    facts: [
      {
        labelEn: "Environments",
        labelZh: "环境数量",
        valueEn: "16 tasks",
        valueZh: "16 个任务",
      },
      {
        labelEn: "Agents",
        labelZh: "Agents",
        valueEn: "4 coding agents",
        valueZh: "4 个 Coding Agent",
      },
      {
        labelEn: "Evidence",
        labelZh: "证据",
        valueEn: "Scores + 64 reruns",
        valueZh: "分数 + 64 段重跑",
      },
    ],
    actionEn: "Explore the archive",
    actionZh: "浏览研究档案",
    actionPath: "results/",
  },
];
