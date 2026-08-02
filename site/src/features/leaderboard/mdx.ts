import type {ComponentType} from "react";
import {
  LeaderboardCaption,
  LeaderboardIntro,
  LeaderboardLead,
  LeaderboardLink,
  LeaderboardLinks,
  LeaderboardNote,
  LeaderboardSection,
} from "./DocumentComponents";
import {
  EnvironmentChart,
  EnvironmentHeader,
  EnvironmentPager,
  EnvironmentProfile,
} from "./EnvironmentWidgets";
import {AggregateTable, EnvironmentDirectory, SuiteFacts} from "./SuiteWidgets";

export type LeaderboardDocument = ComponentType<{
  components?: Record<string, unknown>;
}>;

const sharedDocumentComponents = {
  LeaderboardCaption,
  LeaderboardIntro,
  LeaderboardLead,
  LeaderboardLink,
  LeaderboardLinks,
  LeaderboardNote,
  LeaderboardSection,
};

export const suiteDocumentComponents = {
  ...sharedDocumentComponents,
  AggregateTable,
  EnvironmentDirectory,
  SuiteFacts,
};

export const environmentDocumentComponents = {
  ...sharedDocumentComponents,
  EnvironmentChart,
  EnvironmentHeader,
  EnvironmentPager,
  EnvironmentProfile,
};
