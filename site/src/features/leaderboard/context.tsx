import {createContext, type ReactNode, useContext} from "react";
import type {
  LeaderboardEnvironment,
  LeaderboardRegistry,
  LeaderboardSuiteData,
} from "../../../lib/leaderboard/types";

interface LeaderboardContextValue {
  suite: LeaderboardSuiteData;
  registry: LeaderboardRegistry;
  environment?: LeaderboardEnvironment;
}

const LeaderboardContext = createContext<LeaderboardContextValue | null>(null);

export function LeaderboardProvider({
  suite,
  registry,
  environment,
  children,
}: LeaderboardContextValue & {children: ReactNode}) {
  return (
    <LeaderboardContext.Provider value={{suite, registry, environment}}>
      {children}
    </LeaderboardContext.Provider>
  );
}

export function useLeaderboard(): LeaderboardContextValue {
  const value = useContext(LeaderboardContext);
  if (!value) {
    throw new Error("Leaderboard components must render inside LeaderboardProvider");
  }
  return value;
}

export function useLeaderboardEnvironment(): LeaderboardEnvironment {
  const {environment} = useLeaderboard();
  if (!environment) {
    throw new Error("This component requires an Environment leaderboard route");
  }
  return environment;
}
