import type {SidebarsConfig} from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docsSidebar: [
    "index",
    {
      type: "category",
      label: "Start",
      items: ["getting-started", "concepts"],
    },
    {
      type: "category",
      label: "Core reference",
      items: ["policy", "evaluation", "runtime"],
    },
    {
      type: "category",
      label: "Extend",
      items: ["authoring"],
    },
  ],
};

export default sidebars;
