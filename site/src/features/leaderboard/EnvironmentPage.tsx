import Layout from "@theme/Layout";
import MDXContent from "@theme/MDXContent";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import type {
  LeaderboardRegistry,
  LeaderboardSuiteData,
} from "../../../lib/leaderboard/types";
import {LeaderboardProvider} from "./context";
import {LeaderboardShell} from "./LeaderboardShell";
import {environmentDocumentComponents, type LeaderboardDocument} from "./mdx";

export default function EnvironmentPage({
  suite,
  registry,
  pageData,
  content: Content,
}: {
  suite: LeaderboardSuiteData;
  registry: LeaderboardRegistry;
  pageData: {environmentId: string};
  content: LeaderboardDocument;
}) {
  const language = useSiteLanguage();
  const environment = suite.results.environments.find(
    (item) => item.id === pageData.environmentId,
  );
  if (!environment) {
    throw new Error(`Unknown leaderboard environment: ${pageData.environmentId}`);
  }
  return (
    <Layout
      title={`${environment.display} · ${pickLocalized(language, suite.manifest.label)}`}
      description={pickLocalized(language, environment.summary)}
    >
      <LeaderboardProvider
        suite={suite}
        registry={registry}
        environment={environment}
      >
        <LeaderboardShell
          suite={suite}
          registry={registry}
          currentEnvironmentId={environment.id}
        >
          <MDXContent>
            <Content components={environmentDocumentComponents} />
          </MDXContent>
        </LeaderboardShell>
      </LeaderboardProvider>
    </Layout>
  );
}
