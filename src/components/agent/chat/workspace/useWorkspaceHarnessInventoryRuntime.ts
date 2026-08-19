import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getAgentRuntimeToolInventory } from "@/lib/api/agentRuntime/inventoryClient";
import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { extractArtifactProtocolPathsFromRecord } from "@/lib/artifact-protocol";
import type {
  GeneralWorkbenchRunState as BackendGeneralWorkbenchRunState,
  GeneralWorkbenchRunTerminalItem,
  GeneralWorkbenchRunTodoItem,
} from "@/lib/api/executionRun";

interface UseWorkspaceHarnessInventoryRuntimeParams {
  enabled: boolean;
  chatMode: "agent" | "general" | "workbench";
  harnessPanelVisible: boolean;
  harnessRequestMetadata: Record<string, unknown>;
  isThemeWorkbench: boolean;
  themeWorkbenchRunState: "idle" | "auto_running" | "await_user_decision";
  currentGate: {
    title: string;
    description: string;
  };
  themeWorkbenchBackendRunState: BackendGeneralWorkbenchRunState | null;
  themeWorkbenchActiveQueueItem: GeneralWorkbenchRunTodoItem | null | undefined;
  harnessPendingCount: number;
}

export function useWorkspaceHarnessInventoryRuntime({
  enabled,
  chatMode,
  harnessPanelVisible,
  harnessRequestMetadata,
  isThemeWorkbench,
  themeWorkbenchRunState,
  currentGate,
  themeWorkbenchBackendRunState,
  themeWorkbenchActiveQueueItem,
  harnessPendingCount,
}: UseWorkspaceHarnessInventoryRuntimeParams) {
  const [toolInventory, setToolInventory] =
    useState<AgentRuntimeToolInventory | null>(null);
  const [toolInventoryLoading, setToolInventoryLoading] = useState(false);
  const [toolInventoryError, setToolInventoryError] = useState<string | null>(
    null,
  );
  const toolInventoryRequestIdRef = useRef(0);

  const refreshToolInventory = useCallback(async () => {
    if (!enabled || !harnessPanelVisible) {
      setToolInventory(null);
      setToolInventoryLoading(false);
      setToolInventoryError(null);
      return;
    }

    const requestId = toolInventoryRequestIdRef.current + 1;
    toolInventoryRequestIdRef.current = requestId;
    setToolInventoryLoading(true);
    setToolInventoryError(null);

    try {
      const nextInventory = await getAgentRuntimeToolInventory({
        caller: "assistant",
        workbench: chatMode === "workbench",
        metadata: {
          harness: harnessRequestMetadata,
        },
      });

      if (toolInventoryRequestIdRef.current !== requestId) {
        return;
      }

      setToolInventory(nextInventory);
    } catch (error) {
      if (toolInventoryRequestIdRef.current !== requestId) {
        return;
      }

      setToolInventoryError(
        error instanceof Error ? error.message : "读取工具库存失败",
      );
    } finally {
      if (toolInventoryRequestIdRef.current === requestId) {
        setToolInventoryLoading(false);
      }
    }
  }, [
    chatMode,
    enabled,
    harnessPanelVisible,
    harnessRequestMetadata,
  ]);

  useEffect(() => {
    if (enabled && harnessPanelVisible) {
      return;
    }

    toolInventoryRequestIdRef.current += 1;
    setToolInventory(null);
    setToolInventoryLoading(false);
    setToolInventoryError(null);
  }, [enabled, harnessPanelVisible]);

  useEffect(() => {
    if (!enabled || !harnessPanelVisible) {
      return;
    }

    void refreshToolInventory();
  }, [enabled, harnessPanelVisible, refreshToolInventory]);

  const generalWorkbenchHarnessSummary = useMemo(() => {
    if (!enabled || !isThemeWorkbench) {
      return null;
    }

    const latestTerminal: GeneralWorkbenchRunTerminalItem | null =
      themeWorkbenchBackendRunState?.latest_terminal ?? null;
    const activeRun = themeWorkbenchActiveQueueItem ?? latestTerminal;
    const activeArtifactPaths = extractArtifactProtocolPathsFromRecord(
      themeWorkbenchActiveQueueItem,
    );
    const latestTerminalArtifactPaths =
      extractArtifactProtocolPathsFromRecord(latestTerminal);
    const artifactPaths =
      activeArtifactPaths.length > 0
        ? activeArtifactPaths
        : latestTerminalArtifactPaths;

    return {
      runState: themeWorkbenchRunState,
      stageTitle: currentGate.title,
      stageDescription: currentGate.description,
      runTitle: activeRun?.title || null,
      artifactCount: artifactPaths.length,
      updatedAt:
        themeWorkbenchBackendRunState?.updated_at ||
        latestTerminal?.finished_at ||
        latestTerminal?.started_at ||
        themeWorkbenchActiveQueueItem?.started_at ||
        null,
      pendingCount: harnessPendingCount,
    };
  }, [
    currentGate.description,
    currentGate.title,
    enabled,
    harnessPendingCount,
    isThemeWorkbench,
    themeWorkbenchActiveQueueItem,
    themeWorkbenchBackendRunState?.latest_terminal,
    themeWorkbenchBackendRunState?.updated_at,
    themeWorkbenchRunState,
  ]);

  return {
    toolInventory,
    toolInventoryLoading,
    toolInventoryError,
    refreshToolInventory,
    generalWorkbenchHarnessSummary,
  };
}
