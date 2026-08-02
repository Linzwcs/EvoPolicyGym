import type {LoadContext, Plugin} from "@docusaurus/types";
import {environmentReferences} from "../../src/data/environmentReferences";
import {environments} from "../../src/lib/showcase";

export default function catalogPagesPlugin(context: LoadContext): Plugin<void> {
  const routePath = (route: string) =>
    `${context.baseUrl}${route.replace(/^\/+/, "")}`;

  return {
    name: "evopolicygym-catalog-pages",
    async contentLoaded({actions}) {
      for (const reference of environmentReferences) {
        const referenceData = await actions.createData(
          `environment-${reference.collectionId}.json`,
          JSON.stringify(reference),
        );
        actions.addRoute({
          path: routePath(`/environments/${reference.slug}/`),
          component: "@site/src/features/environments/EnvironmentReferencePage.tsx",
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
          component: "@site/src/features/results/EnvironmentResultPage.tsx",
          exact: true,
          modules: {pageData},
        });
      }
    },
  };
}
