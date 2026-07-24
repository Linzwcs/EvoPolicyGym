import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const docs = defineCollection({
  loader: glob({
    base: "./src/content/docs",
    pattern: "**/*.md",
  }),
  schema: z.object({
    locale: z.enum(["en", "zh"]),
    page: z.enum([
      "getting-started",
      "concepts",
      "policy",
      "evaluation",
      "runtime",
      "authoring",
    ]),
    section: z.enum(["start", "core", "extend"]),
    title: z.string(),
    navTitle: z.string(),
    description: z.string(),
    lead: z.string(),
    index: z.string(),
    order: z.number().int().nonnegative(),
    docsVersion: z.string().default("next"),
    status: z
      .enum(["planning", "draft", "stable", "historical"])
      .default("draft"),
  }),
});

export const collections = { docs };
