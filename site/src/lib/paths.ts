export function withBase(path = ""): string {
  const base = import.meta.env.BASE_URL;
  const clean = path.replace(/^\/+/, "");
  return `${base}${clean}`;
}

export function sectionFromPath(pathname: string): string {
  const sections = ["docs", "environments", "results", "blog"];
  return sections.find((section) => pathname.includes(`/${section}`)) ?? "home";
}
