import type { LocalizedText } from "./environmentCatalog";

export interface EnvironmentReferenceField {
  name: string;
  description: LocalizedText;
}

export interface EnvironmentReferenceAction {
  value: string;
  description: LocalizedText;
}

export interface EnvironmentReferenceMeasure {
  label: LocalizedText;
  value: LocalizedText;
}

export interface EnvironmentReference {
  slug: string;
  collectionId: string;
  title: string;
  ecosystem: string;
  suite: string;
  lead: LocalizedText;
  packageName: string;
  importName: string;
  benchmarkClass: string;
  benchmarkId: string;
  provider: string;
  horizon: LocalizedText;
  direction: LocalizedText;
  sourcePath: string;
  upstreamUrl: string;
  task: LocalizedText[];
  observation: LocalizedText;
  observationFields: EnvironmentReferenceField[];
  action: LocalizedText;
  actions: EnvironmentReferenceAction[];
  measures: EnvironmentReferenceMeasure[];
  feedback: LocalizedText;
  feedbackFields: EnvironmentReferenceField[];
  artifact: EnvironmentReferenceField;
}

export const environmentReferences: EnvironmentReference[] = [
  {
    slug: "atcoder/ahc054/treants-forest",
    collectionId: "atcoder-ahc054",
    title: "Treant’s Forest",
    ecosystem: "AtCoder",
    suite: "AHC054",
    lead: {
      en: "Place permanent Treants on unseen cells to delay a deterministic adventurer while preserving a route to the flower.",
      zh: "在未揭示的格子上永久放置 Treant，在保持通往花朵路径的同时延缓确定性冒险者。",
    },
    packageName: "evopolicygym-benchmark-treants-forest==0.1.0",
    importName: "treants_forest",
    benchmarkClass: "TreantsForestBenchmark",
    benchmarkId: "atcoder/AHC054/TreantsForest/capped-mean-turns-v1",
    provider: "AtCoder Heuristic Contest 054 · AHC054 A",
    horizon: {
      en: "Up to 2,048 interactive turns",
      zh: "最多 2,048 个交互回合",
    },
    direction: {
      en: "Maximize capped mean adventurer turns",
      zh: "最大化有上限的冒险者平均回合数",
    },
    sourcePath: "environments/atcoder/ahc054/treants_forest",
    upstreamUrl: "https://atcoder.jp/contests/ahc054/tasks/ahc054_a",
    task: [
      {
        en: "Each Case is a 20–40 cell square forest with an entrance, a flower, fixed trees, and an adventurer. Before every adventurer move, the Policy may place permanent Treants on cells that have not yet been revealed.",
        zh: "每个 Case 是边长 20–40 的方形森林，包含入口、花朵、固定树木和一名冒险者。每次冒险者移动前，Policy 可以在尚未揭示的格子上永久放置 Treant。",
      },
      {
        en: "A placement set is applied atomically. It must leave both the entrance and the current adventurer position connected to the flower. The adventurer then reveals sight lines and follows a deterministic shortest path toward a private target.",
        zh: "一组放置会被原子执行，并且必须保证入口和冒险者当前位置都仍与花朵连通。随后冒险者揭示视线范围，并沿确定性最短路径前往私有目标。",
      },
    ],
    observation: {
      en: "The first observation publishes the static forest. Later observations publish only the evolving public state; same-Episode Policy memory can retain the map and prior placements.",
      zh: "第一次 Observation 发布静态森林；后续 Observation 只发布变化中的公开状态，同一 Episode 内的 Policy 记忆可以保留地图和历史放置。",
    },
    observationFields: [
      {
        name: "turn",
        description: { en: "Current turn index", zh: "当前回合索引" },
      },
      {
        name: "adventurer",
        description: {
          en: "Current adventurer coordinate",
          zh: "冒险者当前坐标",
        },
      },
      {
        name: "newly_revealed",
        description: {
          en: "Cells revealed by the preceding move",
          zh: "上一步移动新揭示的格子",
        },
      },
      {
        name: "revealed_cells / placed_treants",
        description: {
          en: "Public progress counters",
          zh: "公开的进度计数",
        },
      },
      {
        name: "initial",
        description: {
          en: "Size, entrance, flower, and fixed trees; first observation only",
          zh: "尺寸、入口、花朵和固定树木；仅第一次 Observation",
        },
      },
    ],
    action: {
      en: "Return one atomic placement set. An empty set advances the adventurer without placing a Treant.",
      zh: "返回一组原子化放置。空集合表示不放置 Treant，直接让冒险者继续移动。",
    },
    actions: [
      {
        value: '{"placements": [[row, column], ...]}',
        description: {
          en: "Place distinct Treants on valid unseen empty cells",
          zh: "在合法、未揭示且为空的格子上放置互异的 Treant",
        },
      },
      {
        value: '{"placements": []}',
        description: { en: "Place nothing this turn", zh: "本回合不放置" },
      },
    ],
    measures: [
      {
        label: { en: "Episode contribution", zh: "Episode 计分" },
        value: {
          en: "Valid adventurer movement count, capped at 2,048",
          zh: "合法冒险者移动次数，上限为 2,048",
        },
      },
      {
        label: { en: "Benchmark score", zh: "Benchmark 得分" },
        value: {
          en: "Arithmetic mean of Episode contributions",
          zh: "所有 Episode 计分的算术平均值",
        },
      },
      {
        label: { en: "Policy failure", zh: "Policy failure" },
        value: { en: "Contributes 0", zh: "计为 0" },
      },
    ],
    feedback: {
      en: "Feedback summarizes delay, placements, terminal outcomes, failures, and trace coverage across the evaluated Cases.",
      zh: "Feedback 汇总所有评估 Case 的延迟、放置数量、终止结果、失败与 trace 覆盖。",
    },
    feedbackFields: [
      {
        name: "capped_mean_turns",
        description: { en: "Primary Benchmark score", zh: "主要 Benchmark 得分" },
      },
      {
        name: "mean_placed_treants",
        description: {
          en: "Mean number of permanent placements",
          zh: "永久放置数量的平均值",
        },
      },
      {
        name: "flower_reached / turn_cap_reached",
        description: {
          en: "Terminal outcome counts",
          zh: "终止结果计数",
        },
      },
      {
        name: "policy_failures",
        description: { en: "Failed Episode count", zh: "失败 Episode 数量" },
      },
    ],
    artifact: {
      name: "trace.jsonl",
      description: {
        en: "A bounded semantic trace of public observations, placement sets, and terminal outcomes.",
        zh: "有界的语义 trace，包含公开 Observation、放置集合与终止结果。",
      },
    },
  },
  {
    slug: "atcoder/ahc057/molecules",
    collectionId: "atcoder-ahc057",
    title: "Molecules",
    ecosystem: "AtCoder",
    suite: "AHC057",
    lead: {
      en: "Schedule bonds among moving points on a toroidal plane and finish with ten equal-size components at minimum distance cost.",
      zh: "在环面上为运动点安排连接，并以尽可能低的距离成本形成十个等规模分量。",
    },
    packageName: "evopolicygym-benchmark-molecules==0.1.0",
    importName: "molecules",
    benchmarkClass: "MoleculesBenchmark",
    benchmarkId: "atcoder/AHC057/Molecules/mean-log-cost-score-v1",
    provider: "AtCoder Heuristic Contest 057 · AHC057 A",
    horizon: { en: "1,000 interactive turns", zh: "1,000 个交互回合" },
    direction: {
      en: "Maximize mean logarithmic bond-cost score",
      zh: "最大化平均对数连接成本得分",
    },
    sourcePath: "environments/atcoder/ahc057/molecules",
    upstreamUrl: "https://atcoder.jp/contests/ahc057/tasks/ahc057_a",
    task: [
      {
        en: "Each Case begins with 300 independently moving points on a 100,000 × 100,000 torus. The Policy may add bonds before each simultaneous movement phase.",
        zh: "每个 Case 从 100,000 × 100,000 环面上的 300 个独立运动点开始。Policy 可以在每次同步移动阶段前添加连接。",
      },
      {
        en: "A bond joins two different connected components, incurs rounded toroidal distance cost, and combines component velocity through momentum conservation. At turn 1,000 the graph must contain exactly ten components of 30 points.",
        zh: "连接必须合并两个不同的连通分量，其成本为取整后的环面距离，并通过动量守恒合并分量速度。第 1,000 回合结束时，图必须恰好包含十个各有 30 个点的分量。",
      },
    ],
    observation: {
      en: "Every observation contains the complete public moving system. The first observation also includes the fixed task constants.",
      zh: "每次 Observation 都包含完整的公开运动系统；第一次 Observation 还包含固定任务常量。",
    },
    observationFields: [
      {
        name: "turn / turns_remaining",
        description: { en: "Current temporal state", zh: "当前时间状态" },
      },
      {
        name: "positions",
        description: {
          en: "300 × 2 point positions",
          zh: "300 × 2 的点位置",
        },
      },
      {
        name: "velocities",
        description: {
          en: "300 × 2 component velocities",
          zh: "300 × 2 的分量速度",
        },
      },
      {
        name: "components / component_count",
        description: {
          en: "Canonical component labels and current count",
          zh: "规范化分量标签与当前数量",
        },
      },
      {
        name: "total_cost / initial",
        description: {
          en: "Accumulated cost and first-observation task constants",
          zh: "累计成本与仅首次出现的任务常量",
        },
      },
    ],
    action: {
      en: "Return all bonds for the current turn as one atomic set. An empty set advances the point system without bonding.",
      zh: "将当前回合的全部连接作为一个原子集合返回。空集合表示不连接并推进点系统。",
    },
    actions: [
      {
        value: '{"bonds": [[point_i, point_j], ...]}',
        description: {
          en: "Join pairs from different current components",
          zh: "连接当前不同分量中的点对",
        },
      },
      {
        value: '{"bonds": []}',
        description: { en: "Advance without bonding", zh: "不连接并推进" },
      },
    ],
    measures: [
      {
        label: { en: "Completion", zh: "完成条件" },
        value: {
          en: "Exactly 10 components of 30 points after 1,000 turns",
          zh: "1,000 回合后恰好形成 10 个各含 30 点的分量",
        },
      },
      {
        label: { en: "Benchmark score", zh: "Benchmark 得分" },
        value: {
          en: "Mean official logarithmic distance-cost score",
          zh: "官方对数距离成本得分的平均值",
        },
      },
      {
        label: { en: "Policy failure", zh: "Policy failure" },
        value: { en: "Contributes 0", zh: "计为 0" },
      },
    ],
    feedback: {
      en: "Feedback reports score, total bond cost, completion, failures, and bounded bond-event coverage.",
      zh: "Feedback 报告得分、总连接成本、完成情况、失败与有界连接事件覆盖。",
    },
    feedbackFields: [
      {
        name: "mean_log_cost_score",
        description: { en: "Primary Benchmark score", zh: "主要 Benchmark 得分" },
      },
      {
        name: "mean_total_cost",
        description: {
          en: "Mean completed-solution bond cost",
          zh: "已完成解的平均连接成本",
        },
      },
      {
        name: "completed / policy_failures",
        description: {
          en: "Episode outcome counts",
          zh: "Episode 结果计数",
        },
      },
      {
        name: "bond_events / bond_events_omitted",
        description: {
          en: "Published and omitted trace event counts",
          zh: "已发布与省略的 trace 事件数",
        },
      },
    ],
    artifact: {
      name: "trace.jsonl",
      description: {
        en: "Initial public point state plus a bounded sequence of bond events and resulting component state.",
        zh: "初始公开点状态，以及有界的连接事件与后续分量状态序列。",
      },
    },
  },
  {
    slug: "atcoder/ahc058/apple-incremental-game",
    collectionId: "atcoder-ahc058",
    title: "Apple Incremental Game",
    ecosystem: "AtCoder",
    suite: "AHC058",
    lead: {
      en: "Allocate apples across a four-level production hierarchy and compound machine output over 500 turns.",
      zh: "在四层生产体系中分配苹果，并在 500 回合内复合增长机器产出。",
    },
    packageName: "evopolicygym-benchmark-apple-incremental-game==0.1.0",
    importName: "apple_incremental_game",
    benchmarkClass: "AppleIncrementalGameBenchmark",
    benchmarkId: "atcoder/AHC058/AppleIncrementalGame/mean-log2-score-v1",
    provider: "AtCoder Heuristic Contest 058 · AHC058 A",
    horizon: { en: "500 interactive turns", zh: "500 个交互回合" },
    direction: {
      en: "Maximize mean final log2 apple score",
      zh: "最大化最终苹果数的平均 log2 得分",
    },
    sourcePath: "environments/atcoder/ahc058/apple_incremental_game",
    upstreamUrl: "https://atcoder.jp/contests/ahc058/tasks/ahc058_a",
    task: [
      {
        en: "Each Case contains ten machine IDs at each of four production levels. The system starts with one apple and runs for 500 turns.",
        zh: "每个 Case 在四个生产层级上各包含十个机器 ID。系统从一个苹果开始，持续运行 500 回合。",
      },
      {
        en: "On each turn the Policy strengthens at most one affordable machine or waits. Production then runs in level order, so investment timing and cross-level compounding determine the final apple count.",
        zh: "每回合 Policy 至多强化一台负担得起的机器，或选择等待。随后生产按层级顺序执行，因此投资时机和跨层复合增长决定最终苹果数。",
      },
    ],
    observation: {
      en: "The evolving state is fully public. Capacities and initial costs appear once and can be retained in same-Episode Policy memory.",
      zh: "变化中的状态完全公开。容量和初始成本只出现一次，可由同一 Episode 内的 Policy 记忆保留。",
    },
    observationFields: [
      {
        name: "turn / turns_remaining",
        description: { en: "Current temporal state", zh: "当前时间状态" },
      },
      {
        name: "apples",
        description: { en: "Current spendable apples", zh: "当前可用苹果数" },
      },
      {
        name: "machines",
        description: {
          en: "4 × 10 machine counts",
          zh: "4 × 10 的机器数量",
        },
      },
      {
        name: "powers",
        description: {
          en: "4 × 10 current production powers",
          zh: "4 × 10 的当前生产能力",
        },
      },
      {
        name: "initial",
        description: {
          en: "Capacities and initial costs; first observation only",
          zh: "容量与初始成本；仅第一次 Observation",
        },
      },
    ],
    action: {
      en: "Choose one machine upgrade or wait. The selected level and machine ID must be in range and affordable.",
      zh: "选择一次机器强化或等待。所选层级和机器 ID 必须在范围内且当前可支付。",
    },
    actions: [
      {
        value: '{"upgrade": [level, machine_id]}',
        description: {
          en: "Strengthen one machine before production",
          zh: "在生产前强化一台机器",
        },
      },
      {
        value: "None",
        description: { en: "Wait for one turn", zh: "等待一个回合" },
      },
    ],
    measures: [
      {
        label: { en: "Episode score", zh: "Episode 得分" },
        value: {
          en: "round(100,000 × log2(final apples))",
          zh: "round(100,000 × log2(最终苹果数))",
        },
      },
      {
        label: { en: "Benchmark score", zh: "Benchmark 得分" },
        value: {
          en: "Arithmetic mean of Episode scores",
          zh: "所有 Episode 得分的算术平均值",
        },
      },
      {
        label: { en: "Policy failure", zh: "Policy failure" },
        value: { en: "Contributes 0", zh: "计为 0" },
      },
    ],
    feedback: {
      en: "Feedback connects the primary score to final production scale, upgrade count, completion, and bounded transition coverage.",
      zh: "Feedback 将主要得分与最终生产规模、强化次数、完成情况和有界 transition 覆盖关联起来。",
    },
    feedbackFields: [
      {
        name: "mean_log2_score",
        description: { en: "Primary Benchmark score", zh: "主要 Benchmark 得分" },
      },
      {
        name: "mean_final_apples",
        description: { en: "Mean final apple count", zh: "最终苹果数平均值" },
      },
      {
        name: "mean_total_upgrades",
        description: { en: "Mean upgrade count", zh: "强化次数平均值" },
      },
      {
        name: "completed / policy_failures",
        description: {
          en: "Episode outcome counts",
          zh: "Episode 结果计数",
        },
      },
    ],
    artifact: {
      name: "trace.jsonl",
      description: {
        en: "A bounded turn sequence containing public production state and selected upgrades.",
        zh: "包含公开生产状态与所选强化操作的有界回合序列。",
      },
    },
  },
  {
    slug: "codechef/june18/warehouseman",
    collectionId: "codechef-warehous",
    title: "Warehouseman",
    ecosystem: "CodeChef",
    suite: "June Challenge 2018 · WAREHOUS",
    lead: {
      en: "Compile a complete forklift instruction program that stores arriving shipments and retrieves them in numeric order.",
      zh: "编写完整的叉车指令程序，存放依次到达的货物并按编号顺序取回。",
    },
    packageName: "evopolicygym-benchmark-warehouseman==0.1.0",
    importName: "warehouseman",
    benchmarkClass: "WarehousemanBenchmark",
    benchmarkId: "codechef/WAREHOUS/Warehouseman/mean-normalized-cost-v1",
    provider: "CodeChef June Challenge 2018 · WAREHOUS",
    horizon: {
      en: "One atomic constructive submission",
      zh: "一次原子化构造提交",
    },
    direction: {
      en: "Minimize mean normalized instruction cost",
      zh: "最小化平均归一化指令成本",
    },
    sourcePath: "environments/codechef/june18/warehouseman",
    upstreamUrl: "https://www.codechef.com/problems/WAREHOUS",
    task: [
      {
        en: "Each Case provides a 6–20 by 6–20 warehouse and the complete public shipment arrival permutation. The forklift must store every arrival, retrieve shipments in numeric order, and return to the entrance.",
        zh: "每个 Case 提供一个 6–20 行、6–20 列的仓库和完整公开的货物到达排列。叉车必须存放全部到货，按编号顺序取回，并返回入口。",
      },
      {
        en: "The Policy emits the whole instruction program at once. The Environment parses, validates, and executes it atomically under the 500,000-character limit.",
        zh: "Policy 一次性输出完整指令程序。Environment 在 500,000 字符限制下原子化地解析、校验并执行。",
      },
    ],
    observation: {
      en: "Warehouseman is a constructive one-step task. The initial observation contains every public input needed to synthesize the complete solution.",
      zh: "Warehouseman 是一步构造任务。初始 Observation 包含生成完整解所需的全部公开输入。",
    },
    observationFields: [
      {
        name: "rows / columns",
        description: {
          en: "Warehouse dimensions in the inclusive range 6–20",
          zh: "仓库尺寸，范围均为 6–20",
        },
      },
      {
        name: "arrivals",
        description: {
          en: "Complete shipment arrival permutation",
          zh: "完整货物到达排列",
        },
      },
      {
        name: "instruction_limit",
        description: {
          en: "Maximum output length: 500,000 characters",
          zh: "最大输出长度：500,000 字符",
        },
      },
    ],
    action: {
      en: "Return one ASCII instruction string using movement, pickup, drop-off, load, and unload operations.",
      zh: "返回一个 ASCII 指令字符串，使用移动、拾取、放下、装载和卸载操作。",
    },
    actions: [
      {
        value: "N W S E",
        description: { en: "Move the forklift", zh: "移动叉车" },
      },
      {
        value: "P D",
        description: {
          en: "Pick up an arrival or deliver a shipment",
          zh: "拾取到货或交付货物",
        },
      },
      {
        value: "LN LW LS LE",
        description: {
          en: "Load a neighboring stored shipment",
          zh: "装载相邻已存货物",
        },
      },
      {
        value: "UN UW US UE",
        description: {
          en: "Unload into a neighboring cell",
          zh: "卸载到相邻格子",
        },
      },
    ],
    measures: [
      {
        label: { en: "Completion", zh: "完成条件" },
        value: {
          en: "Store all arrivals, retrieve in numeric order, and return to the entrance",
          zh: "存放全部到货、按编号取回并返回入口",
        },
      },
      {
        label: { en: "Benchmark score", zh: "Benchmark 得分" },
        value: {
          en: "Mean official normalized instruction cost",
          zh: "官方归一化指令成本的平均值",
        },
      },
      {
        label: { en: "Policy failure", zh: "Policy failure" },
        value: {
          en: "Contributes bounded cost 1,000,000",
          zh: "计为有界成本 1,000,000",
        },
      },
    ],
    feedback: {
      en: "Feedback reports normalized cost, instruction length, completed solutions, failures, and diagnostic coverage.",
      zh: "Feedback 报告归一化成本、指令长度、完成解、失败与 diagnostics 覆盖。",
    },
    feedbackFields: [
      {
        name: "mean_normalized_cost",
        description: { en: "Primary Benchmark score", zh: "主要 Benchmark 得分" },
      },
      {
        name: "mean_instruction_characters",
        description: {
          en: "Mean output length for completed solutions",
          zh: "已完成解的平均输出长度",
        },
      },
      {
        name: "completed / policy_failures",
        description: {
          en: "Episode outcome counts",
          zh: "Episode 结果计数",
        },
      },
      {
        name: "diagnostic_episodes",
        description: {
          en: "Number of Episodes represented in diagnostics",
          zh: "diagnostics 覆盖的 Episode 数量",
        },
      },
    ],
    artifact: {
      name: "diagnostics.jsonl",
      description: {
        en: "Bounded per-Episode completion and cost diagnostics without raw arrival permutations or instruction strings.",
        zh: "有界的逐 Episode 完成与成本 diagnostics，不包含原始到货排列或指令字符串。",
      },
    },
  },
];
