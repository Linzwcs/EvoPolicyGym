export function withBase(path = ""): string {
  const base = import.meta.env.BASE_URL;
  const clean = path.replace(/^\/+/, "");
  return `${base}${clean}`;
}

export function sectionFromPath(pathname: string): string {
  if (pathname.includes("/results")) return "research";
  const sections = ["docs", "environments", "runs", "research"];
  return sections.find((section) => pathname.includes(`/${section}`)) ?? "home";
}
