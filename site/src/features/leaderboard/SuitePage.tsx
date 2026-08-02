import Layout from "@theme/Layout";
import MDXContent from "@theme/MDXContent";
import {pickLocalized, useSiteLanguage} from "../../components/Localized";
import type {
  LeaderboardRegistry,
  LeaderboardSuiteData,
} from "../../../lib/leaderboard/types";
import {LeaderboardProvider} from "./context";
import {LeaderboardShell} from "./LeaderboardShell";
import {suiteDocumentComponents, type LeaderboardDocument} from "./mdx";

export default function SuitePage({
  suite,
  registry,
  content: Content,
}: {
  suite: LeaderboardSuiteData;
  registry: LeaderboardRegistry;
  content: LeaderboardDocument;
}) {
  const language = useSiteLanguage();
  return (
    <Layout
      title={pickLocalized(language, suite.manifest.title)}
      description={pickLocalized(language, suite.manifest.description)}
    >
      <LeaderboardProvider suite={suite} registry={registry}>
        <LeaderboardShell suite={suite} registry={registry}>
          <MDXContent>
            <Content components={suiteDocumentComponents} />
          </MDXContent>
        </LeaderboardShell>
      </LeaderboardProvider>
    </Layout>
  );
}
