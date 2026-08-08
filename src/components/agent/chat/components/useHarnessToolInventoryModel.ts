import { useMemo, useState } from "react";
import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { deriveRuntimeToolAvailability } from "../utils/runtimeToolAvailability";
import {
  buildRuntimeToolCapabilityGaps,
  buildToolInventorySourceStats,
  matchesCatalogToolInventoryFilter,
  sortRuntimeToolsByVisibility,
  type ToolInventoryFilterValue,
} from "./harnessStatusPanelViewModel";

interface UseHarnessToolInventoryModelParams {
  toolInventory?: AgentRuntimeToolInventory | null;
  toolInventoryError?: string | null;
  toolInventoryLoading: boolean;
}

export function useHarnessToolInventoryModel({
  toolInventory,
  toolInventoryError,
  toolInventoryLoading,
}: UseHarnessToolInventoryModelParams) {
  const [toolInventoryFilter, setToolInventoryFilter] =
    useState<ToolInventoryFilterValue>("all");

  const hasToolInventorySection =
    toolInventoryLoading ||
    Boolean(toolInventoryError) ||
    Boolean(toolInventory);
  const toolInventorySourceStats = useMemo(
    () => buildToolInventorySourceStats(toolInventory?.catalog_tools || []),
    [toolInventory],
  );
  const filteredCatalogTools = useMemo(
    () =>
      (toolInventory?.catalog_tools || []).filter((entry) =>
        matchesCatalogToolInventoryFilter(entry, toolInventoryFilter),
      ),
    [toolInventory, toolInventoryFilter],
  );
  const toolInventoryRuntimeTools = useMemo(
    () => sortRuntimeToolsByVisibility(toolInventory?.runtime_tools || []),
    [toolInventory?.runtime_tools],
  );
  const runtimeToolAvailability = useMemo(
    () => deriveRuntimeToolAvailability(toolInventory),
    [toolInventory],
  );
  const runtimeToolTotal =
    toolInventory?.counts.runtime_total ?? toolInventoryRuntimeTools.length;
  const runtimeToolVisibleTotal =
    toolInventory?.counts.runtime_visible_total ??
    toolInventoryRuntimeTools.filter((entry) => entry.visible_in_context)
      .length;
  const runtimeToolCapabilityGaps = useMemo(
    () =>
      buildRuntimeToolCapabilityGaps(
        Boolean(toolInventory),
        runtimeToolAvailability,
      ),
    [runtimeToolAvailability, toolInventory],
  );

  return {
    filteredCatalogTools,
    hasToolInventorySection,
    runtimeToolAvailability,
    runtimeToolCapabilityGaps,
    runtimeToolTotal,
    runtimeToolVisibleTotal,
    setToolInventoryFilter,
    toolInventoryCatalogTools: toolInventory?.catalog_tools || [],
    toolInventoryExtensionSurfaces: toolInventory?.extension_surfaces || [],
    toolInventoryExtensionTools: toolInventory?.extension_tools || [],
    toolInventoryFilter,
    toolInventoryMcpTools: toolInventory?.mcp_tools || [],
    toolInventoryNativeTools: toolInventory?.native_tools || [],
    toolInventoryRuntimeTools,
    toolInventorySourceStats,
    toolInventoryWarnings: toolInventory?.warnings || [],
  };
}
