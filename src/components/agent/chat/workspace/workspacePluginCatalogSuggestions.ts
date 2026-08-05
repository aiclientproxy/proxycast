import type { AppServerPluginCatalogSummary } from "@/lib/api/appServerTypes";
import type { InputbarPluginCapability } from "../components/Inputbar/pluginInputCapability";

const PLUGIN_DISABLED_BLOCKER = "PLUGIN_DISABLED";

export function buildWorkspacePluginCatalogSuggestions(
  plugins: readonly AppServerPluginCatalogSummary[],
): InputbarPluginCapability[] {
  return plugins
    .filter((plugin) => plugin.installed)
    .map((plugin) => ({
      pluginId: plugin.id,
      displayName: plugin.name.trim() || plugin.id,
      description: plugin.description.trim() || plugin.id,
      disabled: !plugin.enabled,
      blockerCodes: plugin.enabled ? [] : [PLUGIN_DISABLED_BLOCKER],
    }));
}
