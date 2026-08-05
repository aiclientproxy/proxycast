import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  AppServerPluginCatalogListResponse,
  AppServerPluginCatalogSummary,
} from "@/lib/api/appServerTypes";
import {
  PLUGIN_CATALOG_CHANGED_EVENT,
  listInstalledPluginCatalog,
} from "@/lib/api/pluginCatalog";
import type { InputbarPluginCapability } from "../components/Inputbar/pluginInputCapability";
import { buildWorkspacePluginCatalogSuggestions } from "./workspacePluginCatalogSuggestions";

const EMPTY_PLUGINS: readonly AppServerPluginCatalogSummary[] = [];

export interface UseWorkspacePluginCatalogSuggestionsOptions {
  enabled?: boolean;
  listInstalled?: () => Promise<AppServerPluginCatalogListResponse>;
}

export interface UseWorkspacePluginCatalogSuggestionsResult {
  suggestions: InputbarPluginCapability[];
  loading: boolean;
  error: Error | null;
  refresh: () => void;
}

export function useWorkspacePluginCatalogSuggestions({
  enabled = false,
  listInstalled = listInstalledPluginCatalog,
}: UseWorkspacePluginCatalogSuggestionsOptions): UseWorkspacePluginCatalogSuggestionsResult {
  const [plugins, setPlugins] =
    useState<readonly AppServerPluginCatalogSummary[]>(EMPTY_PLUGINS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const refresh = useCallback(() => {
    setRefreshKey((value) => value + 1);
  }, []);

  useEffect(() => {
    let disposed = false;
    let loadSequence = 0;

    if (!enabled) {
      setPlugins(EMPTY_PLUGINS);
      setLoading(false);
      setError(null);
      return () => {
        disposed = true;
      };
    }

    const load = async () => {
      const currentSequence = (loadSequence += 1);
      setLoading(true);
      try {
        const result = await listInstalled();
        if (!disposed && currentSequence === loadSequence) {
          setPlugins(result.plugins);
          setError(null);
        }
      } catch (caught) {
        if (!disposed && currentSequence === loadSequence) {
          setPlugins(EMPTY_PLUGINS);
          setError(
            caught instanceof Error ? caught : new Error(String(caught)),
          );
        }
      } finally {
        if (!disposed && currentSequence === loadSequence) {
          setLoading(false);
        }
      }
    };

    void load();
    if (typeof window !== "undefined") {
      window.addEventListener(PLUGIN_CATALOG_CHANGED_EVENT, load);
    }
    return () => {
      disposed = true;
      if (typeof window !== "undefined") {
        window.removeEventListener(PLUGIN_CATALOG_CHANGED_EVENT, load);
      }
    };
  }, [enabled, listInstalled, refreshKey]);

  const suggestions = useMemo(
    () => buildWorkspacePluginCatalogSuggestions(plugins),
    [plugins],
  );

  return { suggestions, loading, error, refresh };
}
