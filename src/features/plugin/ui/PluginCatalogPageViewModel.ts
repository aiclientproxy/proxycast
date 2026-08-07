import type {
  AppServerAppInfo,
  AppServerInstalledApp,
  AppServerPluginCatalogCapability,
  AppServerPluginCatalogDetail,
  AppServerPluginCatalogSummary,
} from "@/lib/api/appServerTypes";

export type PluginCatalogView = "all" | "installed";

export interface PluginCatalogFilterState {
  query: string;
  source: string;
  view: PluginCatalogView;
}

export type PluginCatalogAppReadinessStatus = "disabled" | "pending" | "ready";

export interface PluginCatalogAppReadinessItem {
  callable: boolean;
  enabled: boolean;
  id: string;
  name: string;
  status: PluginCatalogAppReadinessStatus;
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

export function projectPluginCatalogApps(
  capabilities: AppServerPluginCatalogCapability[],
  catalog: AppServerAppInfo[],
  installed: AppServerInstalledApp[],
): PluginCatalogAppReadinessItem[] {
  const catalogById = new Map(catalog.map((app) => [app.id, app]));
  const installedById = new Map(installed.map((app) => [app.id, app]));

  return capabilities.map((capability) => {
    const app = catalogById.get(capability.id);
    const runtime = installedById.get(capability.id);
    const enabled = runtime?.enabled ?? app?.isEnabled ?? false;
    const callable = runtime?.callable === true;
    return {
      callable,
      enabled,
      id: capability.id,
      name: app?.name ?? capability.name,
      status: !runtime
        ? "pending"
        : !enabled
          ? "disabled"
          : callable
            ? "ready"
            : "pending",
    };
  });
}
