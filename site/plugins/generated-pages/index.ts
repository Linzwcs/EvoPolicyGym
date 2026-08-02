import type {LoadContext, Plugin} from "@docusaurus/types";
import {environmentReferences} from "../../src/data/environmentReferences";
import {environments} from "../../src/lib/showcase";

export default function generatedPagesPlugin(context: LoadContext): Plugin {
  const routePath = (path: string) =>
    `${context.baseUrl}${path.replace(/^\/+/, "")}`;

  return {
    name: "evopolicygym-generated-pages",
    async contentLoaded({actions}) {
      for (const reference of environmentReferences) {
        const referenceData = await actions.createData(
          `environment-${reference.collectionId}.json`,
          JSON.stringify(reference),
        );
        actions.addRoute({
          path: routePath(`/environments/${reference.slug}/`),
          component:
            "@site/src/features/environments/EnvironmentReferencePage.tsx",
          exact: true,
          modules: {reference: referenceData},
        });
      }

      for (const environment of environments) {
        const pageData = await actions.createData(
          `result-${environment.id}.json`,
          JSON.stringify({id: environment.id}),
        );
        actions.addRoute({
          path: routePath(`/results/environments/${environment.id}/`),
          component:
            "@site/src/features/results/EnvironmentResultPage.tsx",
          exact: true,
          modules: {pageData},
        });
      }
    },
  };
}
