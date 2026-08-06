import { AppServerClient } from "@/lib/api/appServer";
import { open as openDesktopDialog } from "@/lib/desktop-host/plugin-dialog";
import type {
  AppServerPluginCatalogEnabledSetParams,
  AppServerPluginCatalogEnabledSetResponse,
  AppServerPluginCatalogInstallParams,
  AppServerPluginCatalogInstallResponse,
  AppServerPluginCatalogInstalledParams,
  AppServerPluginCatalogListParams,
  AppServerPluginCatalogListResponse,
  AppServerPluginCatalogReadParams,
  AppServerPluginCatalogReadResponse,
  AppServerPluginCatalogUninstallParams,
  AppServerPluginCatalogUninstallResponse,
  AppServerPluginSearchParams,
  AppServerPluginSearchResponse,
} from "@/lib/api/appServerTypes";

export type PluginCatalogClient = Pick<
  AppServerClient,
  | "listPluginCatalog"
  | "searchPlugins"
  | "setPluginCatalogEnabled"
  | "readPluginCatalog"
  | "installPluginCatalog"
  | "uninstallPluginCatalog"
  | "listInstalledPluginCatalog"
>;

export const PLUGIN_CATALOG_CHANGED_EVENT = "lime:plugin-catalog-changed";

function emitPluginCatalogChanged(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(PLUGIN_CATALOG_CHANGED_EVENT));
  }
}

export interface SelectPluginCatalogSourceOptions {
  title?: string;
}

export async function selectPluginCatalogSource(
  options: SelectPluginCatalogSourceOptions = {},
): Promise<string | null> {
  const selected = await openDesktopDialog({
    title: options.title,
    directory: true,
    multiple: false,
  });
  return typeof selected === "string" ? selected : null;
}

export async function listPluginCatalog(
  params: AppServerPluginCatalogListParams = {},
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginCatalogListResponse> {
  return (await client.listPluginCatalog(params)).result;
}

export async function searchPluginCatalog(
  params: AppServerPluginSearchParams,
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginSearchResponse> {
  return (await client.searchPlugins(params)).result;
}

export async function setPluginCatalogEnabled(
  params: AppServerPluginCatalogEnabledSetParams,
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginCatalogEnabledSetResponse> {
  const result = (await client.setPluginCatalogEnabled(params)).result;
  emitPluginCatalogChanged();
  return result;
}

export async function readPluginCatalog(
  params: AppServerPluginCatalogReadParams,
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginCatalogReadResponse> {
  return (await client.readPluginCatalog(params)).result;
}

export async function installPluginCatalog(
  params: AppServerPluginCatalogInstallParams,
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginCatalogInstallResponse> {
  const result = (await client.installPluginCatalog(params)).result;
  emitPluginCatalogChanged();
  return result;
}

export async function uninstallPluginCatalog(
  params: AppServerPluginCatalogUninstallParams,
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginCatalogUninstallResponse> {
  const result = (await client.uninstallPluginCatalog(params)).result;
  emitPluginCatalogChanged();
  return result;
}

export async function listInstalledPluginCatalog(
  params: AppServerPluginCatalogInstalledParams = {},
  client: PluginCatalogClient = new AppServerClient(),
): Promise<AppServerPluginCatalogListResponse> {
  return (await client.listInstalledPluginCatalog(params)).result;
}
