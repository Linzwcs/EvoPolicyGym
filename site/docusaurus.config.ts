import type {Config} from "@docusaurus/types";
import {themes as prismThemes} from "prism-react-renderer";

const config: Config = {
  title: "EvoPolicyGym",
  tagline: "Autonomous policy evolution in interactive environments",
  favicon: "favicon.svg",

  url: "https://linzwcs.github.io",
  baseUrl: "/EvoPolicyGym/",
  organizationName: "Linzwcs",
  projectName: "EvoPolicyGym",
  deploymentBranch: "gh-pages",
  trailingSlash: true,

  onBrokenLinks: "throw",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "throw",
      onBrokenMarkdownImages: "throw",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en", "zh-CN"],
    localeConfigs: {
      en: {
        label: "English",
        htmlLang: "en",
      },
      "zh-CN": {
        label: "中文",
        htmlLang: "zh-CN",
      },
    },
  },

  staticDirectories: ["public"],

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "docs",
          sidebarPath: "./sidebars.ts",
          showLastUpdateAuthor: false,
          showLastUpdateTime: true,
        },
        blog: {
          routeBasePath: "blog",
          showReadingTime: true,
          blogSidebarTitle: "Research notes",
          blogSidebarCount: "ALL",
          postsPerPage: 8,
          feedOptions: {
            type: ["rss", "atom"],
            copyright: `Copyright © ${new Date().getFullYear()} EvoPolicyGym contributors`,
          },
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
        sitemap: {
          changefreq: "weekly",
          priority: 0.5,
        },
      },
    ],
  ],

  plugins: [
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "environments",
        path: "environments",
        routeBasePath: "environments",
        sidebarPath: false,
        breadcrumbs: false,
        showLastUpdateAuthor: false,
        showLastUpdateTime: true,
      },
    ],
    "./plugins/generated-pages/index.ts",
  ],

  themeConfig: {
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    image: "og.png",
    colorMode: {
      defaultMode: "light",
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: "EvoPolicyGym",
      logo: {
        alt: "EvoPolicyGym",
        src: "favicon.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Docs",
        },
        {
          to: "/environments/",
          label: "Environments",
          position: "left",
        },
        {
          to: "/results/",
          label: "Results",
          position: "left",
        },
        {
          to: "/blog/",
          label: "Blog",
          position: "left",
        },
        {
          href: "https://arxiv.org/abs/2607.02440",
          label: "Paper ↗",
          position: "right",
        },
        {
          href: "https://github.com/Linzwcs/EvoPolicyGym",
          label: "GitHub ↗",
          position: "right",
        },
        {
          type: "localeDropdown",
          position: "right",
        },
      ],
    },
    footer: {
      style: "light",
      links: [
        {
          label: "Docs",
          to: "/docs/",
        },
        {
          label: "Environments",
          to: "/environments/",
        },
        {
          label: "Results",
          to: "/results/",
        },
        {
          label: "Blog",
          to: "/blog/",
        },
        {
          label: "Paper ↗",
          href: "https://arxiv.org/abs/2607.02440",
        },
        {
          label: "GitHub ↗",
          href: "https://github.com/Linzwcs/EvoPolicyGym",
        },
      ],
      copyright: `EvoPolicyGym · open-source research software · ${new Date().getFullYear()}`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["bash", "json"],
    },
    metadata: [
      {
        name: "keywords",
        content:
          "coding agents, policy evolution, reinforcement learning, benchmarks, interactive environments",
      },
    ],
  },
};

export default config;
