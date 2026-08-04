export interface LocalizedText {
  en: string;
  zh: string;
}

export interface EnvironmentItem {
  name: string;
  path?: string;
}

export interface EnvironmentCollection {
  id: string;
  domain: string;
  ecosystem: string;
  suite: string;
  distributions: number;
  taskProfiles: number;
  coverage: LocalizedText;
  summary: LocalizedText;
  policyInterface: LocalizedText;
  score: LocalizedText;
  sourcePath: string;
  referencePath?: string;
  items: EnvironmentItem[];
}

export interface EnvironmentDomainGroup {
  id: string;
  title: LocalizedText;
}

const environmentDomainTitles: Readonly<Record<string, LocalizedText>> = {
  control: { en: "Control and robotics", zh: "控制与机器人" },
  planning: { en: "Navigation and planning", zh: "导航与规划" },
  games: { en: "Interactive simulation and games", zh: "交互模拟与游戏" },
};

function humanizeDomainId(domainId: string): string {
  return domainId
    .split(/[-_]/u)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export const environmentCollections: EnvironmentCollection[] = [
  {
    id: "atcoder-ahc054",
    domain: "planning",
    ecosystem: "AtCoder",
    suite: "AHC054 · Treant's Forest",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "1 interactive task", zh: "1 个交互式任务" },
    summary: {
      en: "Long-horizon constraint placement against a deterministic explorer with private targets.",
      zh: "面对具有私有目标的确定性探索者，进行长时程约束放置。",
    },
    policyInterface: {
      en: "Revealed cells and atomic Treant placements",
      zh: "已确认格与原子化 Treant 放置",
    },
    score: {
      en: "Capped mean adventurer turns",
      zh: "有上限的冒险者平均回合数",
    },
    sourcePath: "environments/atcoder/ahc054/treants_forest",
    referencePath: "environments/atcoder/ahc054/treants-forest/",
    items: [
      {
        name: "Treant's Forest",
        path: "environments/atcoder/ahc054/treants-forest/",
      },
    ],
  },
  {
    id: "atcoder-ahc057",
    domain: "planning",
    ecosystem: "AtCoder",
    suite: "AHC057 · Molecules",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "1 interactive task", zh: "1 个交互式任务" },
    summary: {
      en: "Schedule low-cost bonds among moving components on a toroidal plane.",
      zh: "在环面上为运动分量安排低成本连接。",
    },
    policyInterface: {
      en: "Moving point state and atomic bond sets",
      zh: "运动点状态与原子化连接集合",
    },
    score: {
      en: "Mean logarithmic bond-cost score",
      zh: "平均对数连接成本得分",
    },
    sourcePath: "environments/atcoder/ahc057/molecules",
    referencePath: "environments/atcoder/ahc057/molecules/",
    items: [
      {
        name: "Molecules",
        path: "environments/atcoder/ahc057/molecules/",
      },
    ],
  },
  {
    id: "atcoder-ahc058",
    domain: "planning",
    ecosystem: "AtCoder",
    suite: "AHC058 · Apple Incremental Game",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "1 interactive task", zh: "1 个交互式任务" },
    summary: {
      en: "Five hundred turns of hierarchical machine investment and compounding production.",
      zh: "围绕分层机器投资与复合生产的 500 回合规划。",
    },
    policyInterface: {
      en: "Production state and one machine upgrade per turn",
      zh: "生产状态与每回合一次机器强化",
    },
    score: {
      en: "Mean final log2 apple score",
      zh: "最终苹果数的平均 log2 得分",
    },
    sourcePath: "environments/atcoder/ahc058/apple_incremental_game",
    referencePath: "environments/atcoder/ahc058/apple-incremental-game/",
    items: [
      {
        name: "Apple Incremental Game",
        path: "environments/atcoder/ahc058/apple-incremental-game/",
      },
    ],
  },
  {
    id: "codechef-warehous",
    domain: "planning",
    ecosystem: "CodeChef",
    suite: "WAREHOUS · Warehouseman",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "1 constructive challenge", zh: "1 个构造式挑战" },
    summary: {
      en: "Full-range warehouse storage and retrieval through a compact forklift instruction language.",
      zh: "在完整尺寸范围内，通过紧凑叉车指令语言完成仓储与取货。",
    },
    policyInterface: {
      en: "Public arrivals and one atomic instruction Program",
      zh: "公开到货顺序与单次原子化指令 Program",
    },
    score: {
      en: "Mean normalized instruction cost",
      zh: "平均归一化指令成本",
    },
    sourcePath: "environments/codechef/june18/warehouseman",
    referencePath: "environments/codechef/june18/warehouseman/",
    items: [
      {
        name: "Warehouseman",
        path: "environments/codechef/june18/warehouseman/",
      },
    ],
  },
  {
    id: "gymnasium-classic-control",
    domain: "control",
    ecosystem: "Gymnasium",
    suite: "Classic Control",
    distributions: 5,
    taskProfiles: 5,
    coverage: { en: "5 task distributions", zh: "5 个任务 distributions" },
    summary: {
      en: "Small state-space control tasks with discrete or scalar continuous Actions.",
      zh: "小状态空间控制任务，使用离散或标量连续 Action。",
    },
    policyInterface: {
      en: "Named finite state values",
      zh: "具名有限状态值",
    },
    score: { en: "Mean Episode return", zh: "Episode 平均回报" },
    sourcePath: "environments/gymnasium/classic_control",
    items: [
      {
        name: "CartPole",
        path: "environments/gymnasium/classic-control/cartpole/",
      },
      { name: "Acrobot" },
      { name: "Mountain Car" },
      { name: "Continuous Mountain Car" },
      { name: "Pendulum" },
    ],
  },
  {
    id: "gymnasium-toy-text",
    domain: "planning",
    ecosystem: "Gymnasium",
    suite: "Toy Text",
    distributions: 4,
    taskProfiles: 4,
    coverage: { en: "4 task distributions", zh: "4 个任务 distributions" },
    summary: {
      en: "Rule-based discrete tasks for planning, stochastic transitions, and delayed reward.",
      zh: "面向规划、随机转移与延迟奖励的规则型离散任务。",
    },
    policyInterface: {
      en: "Semantic discrete state",
      zh: "语义化离散状态",
    },
    score: {
      en: "Mean return or success rate",
      zh: "平均回报或成功率",
    },
    sourcePath: "environments/gymnasium/toy_text",
    items: [
      { name: "Blackjack" },
      { name: "CliffWalking" },
      { name: "FrozenLake" },
      { name: "Taxi" },
    ],
  },
  {
    id: "gymnasium-box2d",
    domain: "control",
    ecosystem: "Gymnasium",
    suite: "Box2D",
    distributions: 3,
    taskProfiles: 3,
    coverage: { en: "3 task distributions", zh: "3 个任务 distributions" },
    summary: {
      en: "Landing, locomotion, and visual driving in medium-weight physics environments.",
      zh: "中等规模物理环境中的着陆、运动控制与视觉驾驶。",
    },
    policyInterface: {
      en: "Physical state or RGB tensor",
      zh: "物理状态或 RGB Tensor",
    },
    score: { en: "Mean Episode return", zh: "Episode 平均回报" },
    sourcePath: "environments/gymnasium/box2d",
    items: [
      { name: "LunarLander" },
      { name: "BipedalWalker" },
      { name: "CarRacing" },
    ],
  },
  {
    id: "gymnasium-mujoco",
    domain: "control",
    ecosystem: "Gymnasium",
    suite: "MuJoCo",
    distributions: 11,
    taskProfiles: 11,
    coverage: { en: "11 v5 task distributions", zh: "11 个 v5 任务 distributions" },
    summary: {
      en: "Continuous control from small arms and pendulums to contact-rich locomotion.",
      zh: "从小型机械臂、倒立摆到接触丰富运动任务的连续控制。",
    },
    policyInterface: {
      en: "Named body state and continuous torque",
      zh: "具名身体状态与连续力矩",
    },
    score: { en: "Mean Episode return", zh: "Episode 平均回报" },
    sourcePath: "environments/gymnasium/mujoco",
    items: [
      { name: "Reacher" },
      { name: "Pusher" },
      { name: "InvertedPendulum" },
      { name: "InvertedDoublePendulum" },
      { name: "Swimmer" },
      { name: "Hopper" },
      { name: "Walker2d" },
      { name: "HalfCheetah" },
      { name: "Ant" },
      { name: "Humanoid" },
      { name: "HumanoidStandup" },
    ],
  },
  {
    id: "minigrid-standard",
    domain: "planning",
    ecosystem: "MiniGrid",
    suite: "MiniGrid",
    distributions: 21,
    taskProfiles: 21,
    coverage: {
      en: "21 task-family distributions",
      zh: "21 个任务族 distributions",
    },
    summary: {
      en: "Partially observable navigation, memory, object interaction, and procedural layouts.",
      zh: "部分可观测导航、记忆、物体交互与程序化地图。",
    },
    policyInterface: {
      en: "Egocentric grid and mission state",
      zh: "第一视角网格与任务状态",
    },
    score: {
      en: "Success rate or room coverage",
      zh: "成功率或房间覆盖率",
    },
    sourcePath: "environments/minigrid/minigrid",
    items: [
      { name: "DoorKey" },
      { name: "KeyCorridor" },
      { name: "Memory" },
      { name: "Fetch" },
      { name: "MultiRoom" },
      { name: "DynamicObstacles" },
      { name: "ObstructedMaze" },
      { name: "PutNear" },
      { name: "RedBlueDoors" },
      { name: "BlockedUnlockPickup" },
      { name: "UnlockPickup" },
      { name: "Unlock" },
      { name: "LockedRoom" },
      { name: "Crossing" },
      { name: "LavaGap" },
      { name: "DistShift" },
      { name: "GoToObject" },
      { name: "GoToDoor" },
      { name: "FourRooms" },
      { name: "Playground" },
      { name: "Empty" },
    ],
  },
  {
    id: "minigrid-wfc",
    domain: "planning",
    ecosystem: "MiniGrid",
    suite: "WFC",
    distributions: 1,
    taskProfiles: 22,
    coverage: { en: "22 procedural presets", zh: "22 个程序化 presets" },
    summary: {
      en: "Wave Function Collapse layouts spanning mazes, rooms, dungeons, obstacles, and caves.",
      zh: "使用 Wave Function Collapse 生成迷宫、房间、地下城、障碍与洞穴地图。",
    },
    policyInterface: {
      en: "Egocentric procedural grid",
      zh: "第一视角程序化网格",
    },
    score: { en: "Goal success rate", zh: "目标成功率" },
    sourcePath: "environments/minigrid/minigrid/wfc",
    items: [
      { name: "Maze" },
      { name: "Rooms" },
      { name: "Dungeon" },
      { name: "Obstacles" },
      { name: "Skew" },
      { name: "22 presets total" },
    ],
  },
  {
    id: "minigrid-babyai",
    domain: "planning",
    ecosystem: "MiniGrid",
    suite: "BabyAI",
    distributions: 1,
    taskProfiles: 40,
    coverage: { en: "40 tasks · 5 families", zh: "40 个任务 · 5 个任务族" },
    summary: {
      en: "Language-conditioned instruction following from atomic navigation to composite missions.",
      zh: "从原子导航到复合任务的语言条件指令跟随。",
    },
    policyInterface: {
      en: "Grid observation and natural-language mission",
      zh: "网格 Observation 与自然语言任务",
    },
    score: { en: "Mission success rate", zh: "任务成功率" },
    sourcePath: "environments/minigrid/babyai",
    items: [
      { name: "GoTo" },
      { name: "Open" },
      { name: "PickupPut" },
      { name: "Unlock" },
      { name: "Composite" },
      { name: "40 tasks total" },
    ],
  },
  {
    id: "gymnasium-robotics",
    domain: "control",
    ecosystem: "Gymnasium-Robotics",
    suite: "Robotics",
    distributions: 1,
    taskProfiles: 21,
    coverage: { en: "21 task profiles", zh: "21 个任务 profiles" },
    summary: {
      en: "Goal-conditioned reaching, manipulation, maze, dexterous hand, and kitchen tasks.",
      zh: "目标条件的到达、操作、迷宫、灵巧手与厨房任务。",
    },
    policyInterface: {
      en: "Goal-conditioned state and continuous control",
      zh: "目标条件状态与连续控制",
    },
    score: { en: "Success rate", zh: "成功率" },
    sourcePath: "environments/gymnasium_robotics/robotics",
    items: [
      { name: "Fetch" },
      { name: "Point / Ant Maze" },
      { name: "Adroit Hand" },
      { name: "Shadow Hand" },
      { name: "FrankaKitchen" },
    ],
  },
  {
    id: "metaworld",
    domain: "control",
    ecosystem: "MetaWorld",
    suite: "MT",
    distributions: 1,
    taskProfiles: 52,
    coverage: {
      en: "50 MT1 tasks · MT10 · MT50 · custom",
      zh: "50 个 MT1 任务 · MT10 · MT50 · custom",
    },
    summary: {
      en: "Single-task and multi-task robotic manipulation collections.",
      zh: "单任务与多任务机器人操作集合。",
    },
    policyInterface: {
      en: "39-value state with optional task identity",
      zh: "39 维状态与可选任务标识",
    },
    score: { en: "Success rate", zh: "成功率" },
    sourcePath: "environments/metaworld/metaworld",
    items: [
      { name: "MT1" },
      { name: "MT10" },
      { name: "MT50" },
      { name: "Custom collections" },
    ],
  },
  {
    id: "highway-env",
    domain: "games",
    ecosystem: "HighwayEnv",
    suite: "Driving",
    distributions: 1,
    taskProfiles: 10,
    coverage: { en: "10 driving profiles", zh: "10 个驾驶 profiles" },
    summary: {
      en: "Traffic negotiation, parking, racing, and lane keeping.",
      zh: "交通博弈、停车、赛道驾驶与车道保持。",
    },
    policyInterface: {
      en: "Kinematic state with discrete or continuous control",
      zh: "运动学状态与离散或连续控制",
    },
    score: { en: "Mean Episode return", zh: "Episode 平均回报" },
    sourcePath: "environments/highway_env/highway_env",
    items: [
      { name: "Highway" },
      { name: "Merge" },
      { name: "Roundabout" },
      { name: "Intersection" },
      { name: "TwoWay" },
      { name: "Exit" },
      { name: "UTurn" },
      { name: "Parking" },
      { name: "Racetrack" },
      { name: "LaneKeeping" },
    ],
  },
  {
    id: "crafter",
    domain: "games",
    ecosystem: "Crafter",
    suite: "Crafter",
    distributions: 1,
    taskProfiles: 3,
    coverage: { en: "3 scoring profiles", zh: "3 个计分 profiles" },
    summary: {
      en: "Open-world survival, resource collection, crafting, and long-horizon development from RGB observations.",
      zh: "基于 RGB observation 的开放世界生存、资源收集、制造与长时程发展。",
    },
    policyInterface: {
      en: "64 × 64 RGB tensor and 17 discrete Actions",
      zh: "64 × 64 RGB tensor 与 17 个离散 Actions",
    },
    score: {
      en: "Achievement, long-horizon development, or survival-development score",
      zh: "成就、长时程发展或生存发展得分",
    },
    sourcePath: "environments/crafter/crafter",
    items: [
      { name: "Canonical achievement score" },
      { name: "Long-horizon development" },
      { name: "Survival development" },
    ],
  },
  {
    id: "nle-nethack",
    domain: "games",
    ecosystem: "NLE",
    suite: "NetHack",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "NetHackScore-v0", zh: "NetHackScore-v0" },
    summary: {
      en: "Long-horizon procedural dungeon survival with complete semantic training trajectories.",
      zh: "提供完整语义训练轨迹的长时程程序生成地下城生存任务。",
    },
    policyInterface: {
      en: "Terminal map, status, message, inventory, and 23 Actions",
      zh: "终端地图、状态、消息、背包与 23 个 Actions",
    },
    score: { en: "Mean shaped NLE return", zh: "NLE shaped return 平均值" },
    sourcePath: "environments/nle/nethack",
    referencePath: "blog/nethack-policy-evolution/",
    items: [
      { name: "NetHackScore-v0", path: "blog/nethack-policy-evolution/" },
    ],
  },
  {
    id: "ale-atari",
    domain: "games",
    ecosystem: "ALE",
    suite: "Atari",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "Tetris", zh: "Tetris" },
    summary: {
      en: "Arcade control using the redistributable Tetris ROM.",
      zh: "使用可再分发 Tetris ROM 的街机控制。",
    },
    policyInterface: {
      en: "RGB frames and 5 discrete Actions",
      zh: "RGB 画面与 5 个离散 Actions",
    },
    score: { en: "Mean Episode return", zh: "Episode 平均回报" },
    sourcePath: "environments/ale/atari",
    items: [{ name: "Tetris-v5" }],
  },
  {
    id: "vizdoom",
    domain: "games",
    ecosystem: "ViZDoom",
    suite: "Standard scenarios",
    distributions: 1,
    taskProfiles: 12,
    coverage: { en: "12 scenarios", zh: "12 个 scenarios" },
    summary: {
      en: "First-person navigation and combat across the standard packaged scenarios.",
      zh: "标准内置场景中的第一人称导航与战斗。",
    },
    policyInterface: {
      en: "RGB, game variables, and mixed control",
      zh: "RGB、游戏变量与混合控制",
    },
    score: { en: "Mean Episode return", zh: "Episode 平均回报" },
    sourcePath: "environments/vizdoom/vizdoom",
    items: [
      { name: "Basic" },
      { name: "DeadlyCorridor" },
      { name: "Deathmatch" },
      { name: "HealthGathering" },
      { name: "MyWayHome" },
      { name: "TakeCover" },
      { name: "12 scenarios total" },
    ],
  },
  {
    id: "stable-retro",
    domain: "games",
    ecosystem: "Stable-Retro",
    suite: "Airstriker",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "Airstriker Level 1", zh: "Airstriker Level 1" },
    summary: {
      en: "Genesis arcade control from the packaged Level 1 state.",
      zh: "从内置 Level 1 状态开始的 Genesis 街机控制。",
    },
    policyInterface: {
      en: "RGB frames and restricted controller Actions",
      zh: "RGB 画面与受限控制器 Actions",
    },
    score: { en: "Mean score delta", zh: "平均分数增量" },
    sourcePath: "environments/stable_retro/airstriker",
    items: [{ name: "Airstriker-Genesis-v0" }],
  },
  {
    id: "jackdaw-balatro",
    domain: "games",
    ecosystem: "Jackdaw",
    suite: "Balatro",
    distributions: 1,
    taskProfiles: 1,
    coverage: { en: "Red Deck · White Stake", zh: "红色牌组 · 白注" },
    summary: {
      en: "Long-horizon deckbuilding with hands, shops, Jokers, consumables, packs, and antes.",
      zh: "包含出牌、商店、Joker、消耗牌、补充包与 Ante 的长时程牌组构筑。",
    },
    policyInterface: {
      en: "Semantic game state and 21 Action kinds",
      zh: "语义游戏状态与 21 种 Action kind",
    },
    score: { en: "Mean run score", zh: "平均 Run 分数" },
    sourcePath: "environments/jackdaw/balatro",
    items: [{ name: "Red Deck · White Stake" }],
  },
];

export const environmentDomains: EnvironmentDomainGroup[] = Array.from(
  new Set(environmentCollections.map((collection) => collection.domain)),
  (domainId) => {
    const fallbackTitle = humanizeDomainId(domainId);
    return {
      id: domainId,
      title: environmentDomainTitles[domainId] ?? {
        en: fallbackTitle,
        zh: fallbackTitle,
      },
    };
  },
);

export const environmentDistributionCount = environmentCollections.reduce(
  (total, collection) => total + collection.distributions,
  0,
);

export const environmentTaskProfileCount = environmentCollections.reduce(
  (total, collection) => total + collection.taskProfiles,
  0,
);

export const environmentEcosystemCount = new Set(
  environmentCollections.map((collection) => collection.ecosystem),
).size;
