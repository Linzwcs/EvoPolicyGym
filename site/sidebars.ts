import type {SidebarsConfig} from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docsSidebar: [
    "index",
    {
      type: "category",
      label: "Introduction",
      collapsed: false,
      items: ["getting-started", "concepts"],
    },
    {
      type: "category",
      label: "API",
      collapsed: false,
      items: ["programs", "policy", "evaluation", "runs", "runtime"],
    },
    {
      type: "category",
      label: "Extend",
      collapsed: false,
      items: ["authoring"],
    },
  ],
};

export default sidebars;
