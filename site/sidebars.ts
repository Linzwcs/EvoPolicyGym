import type {SidebarsConfig} from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docsSidebar: [
    "index",
    {
      type: "category",
      label: "Start",
      collapsed: false,
      items: ["getting-started", "concepts"],
    },
    {
      type: "category",
      label: "Core reference",
      collapsed: false,
      items: ["policy", "evaluation", "runtime"],
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
