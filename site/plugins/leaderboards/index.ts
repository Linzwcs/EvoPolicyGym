import {readdir, readFile} from "node:fs/promises";
import path from "node:path";
import type {LoadContext, Plugin, RouteMetadata} from "@docusaurus/types";
import {aliasedSitePath, parseMarkdownFile} from "@docusaurus/utils";
import type {
  LeaderboardRegistry,
  LeaderboardSuiteData,
  LocalizedValue,
} from "../../lib/leaderboard/types";
import {
  resolveSuiteFile,
  validateContentExtension,
  validateManifest,
  validateResults,
  validateSuiteCoverage,
} from "../../lib/leaderboard/validate";

interface LeaderboardDocuments {
  suite: LocalizedValue;
  environment: LocalizedValue;
}

interface LoadedLeaderboardSuite {
  data: LeaderboardSuiteData;
  documents: LeaderboardDocuments;
  manifestPath: string;
}

interface LeaderboardsContent {
  suites: LoadedLeaderboardSuite[];
}

export default function leaderboardsPlugin(
  context: LoadContext,
): Plugin<LeaderboardsContent> {
  const contentRoot = path.join(context.siteDir, "leaderboards");
  const routePath = (route: string) =>
    `${context.baseUrl}${route.replace(/^\/+/, "")}`;
  const locale = context.i18n.currentLocale.toLowerCase().startsWith("zh")
    ? "zh"
    : "en";

  return {
    name: "evopolicygym-leaderboards",

    getPathsToWatch() {
      return [`${contentRoot}/**/*.{md,mdx,json}`];
    },

    async loadContent() {
      return {suites: await loadSuites(context, contentRoot)};
    },

    async contentLoaded({content, actions}) {
      const defaultSuite =
        content.suites.find((suite) => suite.data.manifest.default) ??
        content.suites[0];
      if (!defaultSuite) {
        throw new Error("At least one leaderboard suite is required");
      }

      const registry: LeaderboardRegistry = {
        defaultSuiteId: defaultSuite.data.manifest.id,
        suites: content.suites.map(({data}) => ({
          manifest: data.manifest,
          environments: data.results.environments,
        })),
      };
      const registryData = await actions.createData(
        "leaderboard-suites.json",
        JSON.stringify(registry),
      );

      for (const loaded of content.suites) {
        const {data: suite} = loaded;
        const suiteData = await actions.createData(
          `leaderboard-suite-${suite.manifest.id}.json`,
          JSON.stringify(suite),
        );
        const sharedModules = {suite: suiteData, registry: registryData};
        const metadata = routeMetadata(context, loaded.manifestPath);
        const canonicalBase = leaderboardBase(suite);
        const routeBases = Array.from(
          new Set([
            canonicalBase,
            `/leaderboard/suites/${suite.manifest.slug}/`,
          ]),
        );

        for (const routeBase of routeBases) {
          actions.addRoute({
            path: routePath(routeBase),
            component: "@site/src/features/leaderboard/SuitePage.tsx",
            exact: true,
            modules: {
              ...sharedModules,
              content: loaded.documents.suite[locale],
            },
            metadata,
          });
        }

        for (const environment of suite.results.environments) {
          const pageData = await actions.createData(
            `leaderboard-${suite.manifest.id}-${environment.id}.json`,
            JSON.stringify({environmentId: environment.id}),
          );
          for (const routeBase of routeBases) {
            actions.addRoute({
              path: routePath(`${routeBase}environments/${environment.id}/`),
              component: "@site/src/features/leaderboard/EnvironmentPage.tsx",
              exact: true,
              modules: {
                ...sharedModules,
                pageData,
                content: loaded.documents.environment[locale],
              },
              metadata,
            });
          }
        }
      }

      actions.addRoute({
        path: routePath("/leaderboard/"),
        component: "@site/src/features/leaderboard/LeaderboardIndexPage.tsx",
        exact: true,
        modules: {
          registry: registryData,
        },
        metadata: routeMetadata(context, defaultSuite.manifestPath),
      });
    },
  };
}

function leaderboardBase(suite: LeaderboardSuiteData): string {
  const collection =
    suite.manifest.status === "archived" ? "archive" : "distributions";
  return `/leaderboard/${collection}/${suite.manifest.slug}/`;
}

async function loadSuites(
  context: LoadContext,
  contentRoot: string,
): Promise<LoadedLeaderboardSuite[]> {
  const directories = (await readdir(contentRoot, {withFileTypes: true}))
    .filter((entry) => entry.isDirectory())
    .sort((left, right) => left.name.localeCompare(right.name));
  const suites = await Promise.all(
    directories.map((directory) =>
      loadSuite(context, path.join(contentRoot, directory.name), directory.name),
    ),
  );

  if (suites.filter((suite) => suite.data.manifest.default).length !== 1) {
    throw new Error("Exactly one leaderboard suite must be marked as default");
  }
  const defaultSuite = suites.find((suite) => suite.data.manifest.default);
  if (defaultSuite?.data.manifest.status === "archived") {
    throw new Error("The default leaderboard Distribution cannot be archived");
  }
  const ids = new Set<string>();
  for (const suite of suites) {
    if (ids.has(suite.data.manifest.id)) {
      throw new Error(`Duplicate leaderboard suite id: ${suite.data.manifest.id}`);
    }
    ids.add(suite.data.manifest.id);
  }
  return suites;
}

async function loadSuite(
  context: LoadContext,
  suiteDirectory: string,
  directoryName: string,
): Promise<LoadedLeaderboardSuite> {
  const manifestPath = path.join(suiteDirectory, "index.mdx");
  const fileContent = await readFile(manifestPath, "utf8");
  const parsed = await parseMarkdownFile({
    filePath: manifestPath,
    fileContent,
    parseFrontMatter: context.siteConfig.markdown.parseFrontMatter,
  });
  const manifest = validateManifest(parsed.frontMatter, manifestPath);
  if (manifest.slug !== directoryName) {
    throw new Error(`${manifestPath}: slug must match its containing directory`);
  }

  const resultsPath = resolveSuiteFile(
    suiteDirectory,
    manifest.results,
    `${manifestPath}: results`,
  );
  const results = validateResults(
    JSON.parse(await readFile(resultsPath, "utf8")) as unknown,
    resultsPath,
  );
  validateSuiteCoverage(manifest, results, resultsPath);

  const documents = {
    suite: await resolveDocuments(context, suiteDirectory, manifest.content.suite),
    environment: await resolveDocuments(
      context,
      suiteDirectory,
      manifest.content.environment,
    ),
  };
  return {data: {manifest, results}, documents, manifestPath};
}

async function resolveDocuments(
  context: LoadContext,
  suiteDirectory: string,
  files: LocalizedValue,
): Promise<LocalizedValue> {
  async function resolveDocument(relativePath: string): Promise<string> {
    const filePath = resolveSuiteFile(
      suiteDirectory,
      relativePath,
      `${suiteDirectory}: content`,
    );
    validateContentExtension(filePath);
    await readFile(filePath, "utf8");
    return aliasedSitePath(filePath, context.siteDir);
  }
  const [en, zh] = await Promise.all([
    resolveDocument(files.en),
    resolveDocument(files.zh),
  ]);
  return {en, zh};
}

function routeMetadata(context: LoadContext, source: string): RouteMetadata {
  return {sourceFilePath: path.relative(context.siteDir, source)};
}
