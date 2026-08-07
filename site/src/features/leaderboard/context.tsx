import {createContext, type ReactNode, useContext, useState} from "react";
import type {
  LeaderboardEnvironment,
  LeaderboardRegistry,
  LeaderboardSuiteData,
  LeaderboardTestConfiguration,
} from "../../../lib/leaderboard/types";

interface LeaderboardContextValue {
  suite: LeaderboardSuiteData;
  registry: LeaderboardRegistry;
  environment?: LeaderboardEnvironment;
  selectedConfiguration?: LeaderboardTestConfiguration;
  selectConfiguration?: (configurationId: string) => void;
}

const LeaderboardContext = createContext<LeaderboardContextValue | null>(null);

export function LeaderboardProvider({
  suite,
  registry,
  environment,
  children,
}: Pick<LeaderboardContextValue, "suite" | "registry" | "environment"> & {
  children: ReactNode;
}) {
  const [selectedConfigurationId, setSelectedConfigurationId] = useState(
    environment?.default_configuration_id,
  );
  const selectedConfiguration = suite.results.test_configurations?.find(
    (configuration) => configuration.id === selectedConfigurationId,
  );
  const selectConfiguration = environment?.configuration_ids
    ? (configurationId: string) => {
        if (!environment.configuration_ids?.includes(configurationId)) {
          throw new Error(
            `Configuration ${configurationId} is unavailable for ${environment.id}`,
          );
        }
        setSelectedConfigurationId(configurationId);
      }
    : undefined;
  return (
    <LeaderboardContext.Provider
      value={{
        suite,
        registry,
        environment,
        selectedConfiguration,
        selectConfiguration,
      }}
    >
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
