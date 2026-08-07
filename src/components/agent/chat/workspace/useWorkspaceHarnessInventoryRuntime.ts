import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getAgentRuntimeToolInventory } from "@/lib/api/agentRuntime/inventoryClient";
import {
  type AgentRuntimeToolInventory,
  type AgentRuntimeToolInventoryPluginMcpTarget,
} from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  mcpApi,
  type McpCallProofRequest,
  type McpPrepareRequest,
  type McpPrepareResult,
  type McpToolDefinition,
} from "@/lib/api/mcp";
import { extractArtifactProtocolPathsFromRecord } from "@/lib/artifact-protocol";
import type {
  GeneralWorkbenchRunState as BackendGeneralWorkbenchRunState,
  GeneralWorkbenchRunTerminalItem,
  GeneralWorkbenchRunTodoItem,
} from "@/lib/api/executionRun";

interface UseWorkspaceHarnessInventoryRuntimeParams {
  enabled: boolean;
  chatMode: "agent" | "general" | "workbench";
  mappedTheme: string;
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
  threadId?: string | null;
}

interface PluginMcpPrepareTarget {
  expectedToolName: string | null;
  callProofRequests: McpCallProofRequest[];
  prepareRequests: McpPrepareRequest[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getPluginMcpTargets(
  inventory: AgentRuntimeToolInventory | null,
): AgentRuntimeToolInventoryPluginMcpTarget[] {
  return inventory?.plugin_mcp_targets ?? [];
}

function isCandidateMcpCallProofRequest(
  request: unknown,
): request is McpCallProofRequest {
  return (
    isRecord(request) &&
    request.status === "candidate" &&
    typeof request.method === "string"
  );
}

function getCandidateMcpPrepareTargets(
  inventory: AgentRuntimeToolInventory | null,
): PluginMcpPrepareTarget[] {
  return getPluginMcpTargets(inventory).flatMap((target) => {
    const prepareRequests = target.prepareRequests.filter(
      (request): request is McpPrepareRequest =>
        isRecord(request) && request.status === "candidate",
    );
    const callProofRequests = isCandidateMcpCallProofRequest(
      target.callProofRequest,
    )
      ? [target.callProofRequest]
      : [];
    if (
      prepareRequests.length === 0 &&
      callProofRequests.length === 0 &&
      target.toolListRequest
    ) {
      prepareRequests.push({
        method: "mcpTool/listForContext",
        params: target.toolListRequest,
        reason: "tool_listing_default_proof",
        status: "candidate",
      });
    }
    if (prepareRequests.length === 0 && callProofRequests.length === 0) {
      return [];
    }

    const expectedToolName =
      typeof target.expectedToolName === "string" &&
      target.expectedToolName.trim().length > 0
        ? target.expectedToolName.trim()
        : null;
    return [
      {
        expectedToolName,
        callProofRequests,
        prepareRequests,
      },
    ];
  });
}

function mcpToolMatchesExpectedName(
  tool: McpToolDefinition,
  expectedToolName: string,
): boolean {
  return tool.name.trim().toLowerCase() === expectedToolName.toLowerCase();
}

function assertMcpPrepareResultsExposeExpectedTools(
  targets: PluginMcpPrepareTarget[],
  requests: McpPrepareRequest[],
  results: McpPrepareResult[],
): void {
  const missingTool = targets.find((target) => {
    const expectedToolName = target.expectedToolName;
    return (
      expectedToolName &&
      !target.prepareRequests.some((request) => {
        if (request.method !== "mcpTool/listForContext") {
          return false;
        }
        const requestIndex = requests.indexOf(request);
        if (requestIndex < 0) {
          return false;
        }
        const result = results[requestIndex];
        return (result?.tools ?? []).some((tool) =>
          mcpToolMatchesExpectedName(tool, expectedToolName),
        );
      })
    );
  });
  if (missingTool) {
    throw new Error("准备 MCP 工具失败");
  }
}

function buildAutoMcpListProofSignature(
  targets: PluginMcpPrepareTarget[],
): string | null {
  if (targets.length === 0) {
    return null;
  }
  const entries = targets.map((target) => {
    if (
      !target.expectedToolName ||
      target.callProofRequests.length > 0 ||
      target.prepareRequests.length === 0 ||
      !target.prepareRequests.every(
        (request) => request.method === "mcpTool/listForContext",
      )
    ) {
      return null;
    }
    return {
      expectedToolName: target.expectedToolName,
      requests: target.prepareRequests.map((request) => ({
        method: request.method,
        params: request.params ?? {},
      })),
    };
  });
  return entries.every((entry): entry is NonNullable<typeof entry> =>
    Boolean(entry),
  )
    ? JSON.stringify(entries)
    : null;
}

export function useWorkspaceHarnessInventoryRuntime({
  enabled,
  chatMode,
  mappedTheme,
  harnessPanelVisible,
  harnessRequestMetadata,
  isThemeWorkbench,
  themeWorkbenchRunState,
  currentGate,
  themeWorkbenchBackendRunState,
  themeWorkbenchActiveQueueItem,
  harnessPendingCount,
  threadId,
}: UseWorkspaceHarnessInventoryRuntimeParams) {
  const [toolInventory, setToolInventory] =
    useState<AgentRuntimeToolInventory | null>(null);
  const [toolInventoryLoading, setToolInventoryLoading] = useState(false);
  const [toolInventoryError, setToolInventoryError] = useState<string | null>(
    null,
  );
  const toolInventoryRequestIdRef = useRef(0);
  const [mcpPrepareLoading, setMcpPrepareLoading] = useState(false);
  const [mcpPrepareError, setMcpPrepareError] = useState<string | null>(null);
  const mcpPrepareRequestIdRef = useRef(0);
  const mcpAutoPrepareSignatureRef = useRef<string | null>(null);

  const refreshToolInventory = useCallback(async () => {
    if (!enabled || !harnessPanelVisible) {
      setToolInventory(null);
      setToolInventoryLoading(false);
      setToolInventoryError(null);
      setMcpPrepareLoading(false);
      setMcpPrepareError(null);
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
        browserAssist: mappedTheme === "general",
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
    mappedTheme,
  ]);

  useEffect(() => {
    if (enabled && harnessPanelVisible) {
      return;
    }

    toolInventoryRequestIdRef.current += 1;
    mcpPrepareRequestIdRef.current += 1;
    mcpAutoPrepareSignatureRef.current = null;
    setToolInventory(null);
    setToolInventoryLoading(false);
    setToolInventoryError(null);
    setMcpPrepareLoading(false);
    setMcpPrepareError(null);
  }, [enabled, harnessPanelVisible]);

  useEffect(() => {
    if (!enabled || !harnessPanelVisible) {
      return;
    }

    void refreshToolInventory();
  }, [enabled, harnessPanelVisible, refreshToolInventory]);

  const mcpPrepareTargets = useMemo(
    () => getCandidateMcpPrepareTargets(toolInventory),
    [toolInventory],
  );
  const mcpPrepareCandidateRequests = useMemo(
    () => mcpPrepareTargets.flatMap((target) => target.prepareRequests),
    [mcpPrepareTargets],
  );
  const mcpCallProofCandidateRequests = useMemo(
    () => mcpPrepareTargets.flatMap((target) => target.callProofRequests),
    [mcpPrepareTargets],
  );
  const mcpAutoPrepareSignature = useMemo(
    () => buildAutoMcpListProofSignature(mcpPrepareTargets),
    [mcpPrepareTargets],
  );

  const prepareMcpTargets = useCallback(async () => {
    if (!enabled || !harnessPanelVisible) {
      return;
    }

    if (
      mcpPrepareCandidateRequests.length === 0 &&
      mcpCallProofCandidateRequests.length === 0
    ) {
      setMcpPrepareError(null);
      return;
    }

    const requestId = mcpPrepareRequestIdRef.current + 1;
    mcpPrepareRequestIdRef.current = requestId;
    setMcpPrepareLoading(true);
    setMcpPrepareError(null);

    try {
      if (mcpPrepareCandidateRequests.length > 0) {
        const results = await mcpApi.executePrepareRequests(
          mcpPrepareCandidateRequests,
        );
        assertMcpPrepareResultsExposeExpectedTools(
          mcpPrepareTargets,
          mcpPrepareCandidateRequests,
          results,
        );
      }
      if (mcpCallProofCandidateRequests.length > 0) {
        const ownerThreadId = threadId?.trim();
        if (!ownerThreadId) {
          throw new Error("MCP call proof requires the active Thread");
        }
        await mcpApi.executeCallProofRequests(
          mcpCallProofCandidateRequests,
          ownerThreadId,
        );
      }
      if (mcpPrepareRequestIdRef.current !== requestId) {
        return;
      }
      await refreshToolInventory();
    } catch (error) {
      if (mcpPrepareRequestIdRef.current !== requestId) {
        return;
      }
      setMcpPrepareError(
        error instanceof Error ? error.message : "准备 MCP 工具失败",
      );
    } finally {
      if (mcpPrepareRequestIdRef.current === requestId) {
        setMcpPrepareLoading(false);
      }
    }
  }, [
    enabled,
    harnessPanelVisible,
    mcpCallProofCandidateRequests,
    mcpPrepareCandidateRequests,
    mcpPrepareTargets,
    refreshToolInventory,
    threadId,
  ]);

  useEffect(() => {
    if (!mcpAutoPrepareSignature) {
      mcpAutoPrepareSignatureRef.current = null;
      return;
    }
    if (
      !enabled ||
      !harnessPanelVisible ||
      toolInventoryLoading ||
      mcpPrepareLoading
    ) {
      return;
    }
    if (mcpAutoPrepareSignatureRef.current === mcpAutoPrepareSignature) {
      return;
    }
    mcpAutoPrepareSignatureRef.current = mcpAutoPrepareSignature;
    void prepareMcpTargets();
  }, [
    enabled,
    harnessPanelVisible,
    mcpAutoPrepareSignature,
    mcpPrepareLoading,
    prepareMcpTargets,
    toolInventoryLoading,
  ]);

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
    mcpPrepareCandidateCount:
      mcpPrepareCandidateRequests.length + mcpCallProofCandidateRequests.length,
    mcpPrepareLoading,
    mcpPrepareError,
    prepareMcpTargets,
    generalWorkbenchHarnessSummary,
  };
}
