import {useSiteLanguage} from "../../components/Localized";

const messages = {
  en: {
    navigationAria: "Leaderboard navigation",
    leaderboard: "Leaderboard",
    currentSuite: "Current suite",
    findEnvironment: "Find environment",
    searchPlaceholder: "Search",
    byEnvironment: "By environment",
    suites: "Suites",
    tracks: "Tracks",
    agentEntries: "Agent entries",
    finalReruns: "Final reruns",
    profile: "Profile",
    agentModel: "Agent / model",
    trackWins: "Track wins",
    podiums: "Podiums",
    meanRank: "Mean rank",
    randomReference: "random reference",
    versusRandom: "vs random",
  },
  zh: {
    navigationAria: "排行榜导航",
    leaderboard: "排行榜",
    currentSuite: "当前榜单",
    findEnvironment: "查询环境",
    searchPlaceholder: "输入名称",
    byEnvironment: "环境分榜",
    suites: "版本",
    tracks: "Tracks",
    agentEntries: "Agent 条目",
    finalReruns: "最终重跑",
    profile: "Profile",
    agentModel: "Agent / 模型",
    trackWins: "Track 胜场",
    podiums: "前三",
    meanRank: "平均名次",
    randomReference: "随机参考",
    versusRandom: "相对随机",
  },
} as const;

export function useLeaderboardMessages() {
  return messages[useSiteLanguage()];
}
