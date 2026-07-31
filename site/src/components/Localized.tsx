import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import type {ReactNode} from "react";

export type SiteLanguage = "en" | "zh";

export function useSiteLanguage(): SiteLanguage {
  const {i18n} = useDocusaurusContext();
  return i18n.currentLocale.toLowerCase().startsWith("zh") ? "zh" : "en";
}

export interface LocalizedValue {
  en: string;
  zh: string;
}

interface LocalizedProps extends LocalizedValue {
  children?: never;
}

export function Localized({en, zh}: LocalizedProps): ReactNode {
  return useSiteLanguage() === "zh" ? zh : en;
}

export function pickLocalized<T>(language: SiteLanguage, value: {en: T; zh: T}): T {
  return value[language];
}
