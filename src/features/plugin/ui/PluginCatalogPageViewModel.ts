import type {
  AppServerPluginCatalogDetail,
  AppServerPluginCatalogSummary,
} from "@/lib/api/appServerTypes";

export type PluginCatalogView = "all" | "installed";

export interface PluginCatalogFilterState {
  query: string;
  source: string;
  view: PluginCatalogView;
}

export function filterPluginCatalog(
  plugins: AppServerPluginCatalogSummary[],
  filters: PluginCatalogFilterState,
): AppServerPluginCatalogSummary[] {
  const query = filters.query.trim().toLocaleLowerCase();
  return plugins.filter((plugin) => {
    if (filters.view === "installed" && !plugin.installed) {
      return false;
    }
    if (filters.source !== "all" && plugin.source !== filters.source) {
      return false;
    }
    if (!query) {
      return true;
    }
    return [plugin.id, plugin.name, plugin.description, plugin.source]
      .join(" ")
      .toLocaleLowerCase()
      .includes(query);
  });
}

export function listPluginCatalogSources(
  plugins: AppServerPluginCatalogSummary[],
): string[] {
  return Array.from(
    new Set(plugins.map((plugin) => plugin.source.trim()).filter(Boolean)),
  ).sort((left, right) => left.localeCompare(right));
}

export function mergePluginCatalogSummary(
  plugins: AppServerPluginCatalogSummary[],
  summary: AppServerPluginCatalogSummary,
): AppServerPluginCatalogSummary[] {
  const next = plugins.filter((plugin) => plugin.id !== summary.id);
  next.push(summary);
  return next.sort((left, right) => left.name.localeCompare(right.name));
}

export function detailCapabilityCount(
  detail: AppServerPluginCatalogDetail,
): number {
  return (
    detail.skills.length +
    detail.mcpServers.length +
    detail.apps.length +
    detail.hooks.length
  );
}
