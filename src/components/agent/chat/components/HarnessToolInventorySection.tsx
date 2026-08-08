import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import type { RuntimeToolAvailability } from "../utils/runtimeToolAvailability";
import { HarnessCatalogToolInventoryList } from "./HarnessCatalogToolInventoryList";
import { HarnessExtensionToolInventorySections } from "./HarnessExtensionToolInventorySections";
import { HarnessNativeToolInventoryList } from "./HarnessNativeToolInventoryList";
import { HarnessRuntimeToolInventoryList } from "./HarnessRuntimeToolInventoryList";
import {
  HarnessStatusSection as Section,
  type HarnessSectionKey,
} from "./HarnessStatusSectionFrame";
import { HarnessToolInventoryOverview } from "./HarnessToolInventoryOverview";
import type { RuntimeToolCapabilityGap } from "./HarnessToolInventoryTypes";
import { agentText } from "./harnessPanelText";
import type {
  ToolInventoryFilterValue,
  ToolInventorySourceStats,
} from "./harnessStatusPanelViewModel";

interface HarnessToolInventorySectionProps {
  hasToolInventorySection: boolean;
  toolInventory?: AgentRuntimeToolInventory | null;
  toolInventoryLoading: boolean;
  toolInventoryError: string | null;
  runtimeToolVisibleTotal: number;
  runtimeToolTotal: number;
  registerSectionRef: (
    key: HarnessSectionKey,
    node: HTMLElement | null,
  ) => void;
  onRefreshToolInventory?: () => void;
  toolInventorySourceStats: ToolInventorySourceStats;
  toolInventoryWarnings: AgentRuntimeToolInventory["warnings"];
  runtimeToolAvailability: RuntimeToolAvailability;
  runtimeToolCapabilityGaps: RuntimeToolCapabilityGap[];
  toolInventoryRuntimeTools: NonNullable<
    AgentRuntimeToolInventory["runtime_tools"]
  >;
  toolInventoryCatalogTools: AgentRuntimeToolInventory["catalog_tools"];
  filteredCatalogTools: AgentRuntimeToolInventory["catalog_tools"];
  toolInventoryFilter: ToolInventoryFilterValue;
  setToolInventoryFilter: (value: ToolInventoryFilterValue) => void;
  toolInventoryNativeTools: AgentRuntimeToolInventory["native_tools"];
  toolInventoryExtensionSurfaces: AgentRuntimeToolInventory["extension_surfaces"];
  toolInventoryExtensionTools: AgentRuntimeToolInventory["extension_tools"];
  toolInventoryMcpTools: AgentRuntimeToolInventory["mcp_tools"];
}

export function HarnessToolInventorySection({
  hasToolInventorySection,
  toolInventory,
  toolInventoryLoading,
  toolInventoryError,
  runtimeToolVisibleTotal,
  runtimeToolTotal,
  registerSectionRef,
  onRefreshToolInventory,
  toolInventorySourceStats,
  toolInventoryWarnings,
  runtimeToolAvailability,
  runtimeToolCapabilityGaps,
  toolInventoryRuntimeTools,
  toolInventoryCatalogTools,
  filteredCatalogTools,
  toolInventoryFilter,
  setToolInventoryFilter,
  toolInventoryNativeTools,
  toolInventoryExtensionSurfaces,
  toolInventoryExtensionTools,
  toolInventoryMcpTools,
}: HarnessToolInventorySectionProps) {
  if (!hasToolInventorySection) {
    return null;
  }

  return (
    <Section
      sectionKey="inventory"
      title={agentText("agentChat.harness.generated.0ddd6d9a60", "工具与权限")}
      badge={
        toolInventoryLoading
          ? "读取中"
          : toolInventory
            ? `runtime ${runtimeToolVisibleTotal}/${runtimeToolTotal}`
            : toolInventoryError
              ? "读取失败"
              : "待同步"
      }
      registerRef={registerSectionRef}
    >
      <div className="space-y-4">
        <HarnessToolInventoryOverview
          toolInventory={toolInventory}
          toolInventoryLoading={toolInventoryLoading}
          toolInventoryError={toolInventoryError}
          runtimeToolVisibleTotal={runtimeToolVisibleTotal}
          runtimeToolTotal={runtimeToolTotal}
          onRefreshToolInventory={onRefreshToolInventory}
          toolInventorySourceStats={toolInventorySourceStats}
          toolInventoryWarnings={toolInventoryWarnings}
          runtimeToolAvailability={runtimeToolAvailability}
          runtimeToolCapabilityGaps={runtimeToolCapabilityGaps}
        />

        {toolInventory ? (
          <>
            <HarnessRuntimeToolInventoryList
              runtimeToolVisibleTotal={runtimeToolVisibleTotal}
              runtimeToolTotal={runtimeToolTotal}
              toolInventoryRuntimeTools={toolInventoryRuntimeTools}
            />
            <HarnessCatalogToolInventoryList
              toolInventoryCatalogTools={toolInventoryCatalogTools}
              filteredCatalogTools={filteredCatalogTools}
              toolInventoryFilter={toolInventoryFilter}
              setToolInventoryFilter={setToolInventoryFilter}
            />
            <HarnessNativeToolInventoryList
              toolInventoryNativeTools={toolInventoryNativeTools}
            />
            <HarnessExtensionToolInventorySections
              toolInventoryExtensionSurfaces={toolInventoryExtensionSurfaces}
              toolInventoryExtensionTools={toolInventoryExtensionTools}
              toolInventoryMcpTools={toolInventoryMcpTools}
            />
          </>
        ) : !toolInventoryLoading && !toolInventoryError ? (
          <div className="rounded-lg border border-dashed border-border px-3 py-3 text-sm text-muted-foreground">
            {agentText(
              "agentChat.harness.generated.1f864fb681",
              "当前尚未拿到工具库存快照。",
            )}
          </div>
        ) : null}
      </div>
    </Section>
  );
}
