import type { MutableRefObject } from "react";
import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem } from "../types";
import { normalizeExecutionStrategy } from "./agentChatCoreUtils";
import type { AgentAccessMode } from "./agentChatStorage";
import { createSessionAccessModeFromExecutionRuntime } from "../utils/sessionExecutionRuntime";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type { AgentSessionDetailMergeMode } from "./agentSessionState";
import { mergeRuntimeSyncThreadItems } from "./agentSessionTimelineMergePolicy";

export interface AgentSessionDetailRefreshRequest {
  source?: string | null;
  detailMergeMode?: AgentSessionDetailMergeMode | null;
}

export interface AgentSessionReadModelSnapshot {
  threadRead: AgentRuntimeThreadReadModel | null;
}

export function createAgentSessionReadModelSnapshot(
  threadRead?: AgentRuntimeThreadReadModel | null,
): AgentSessionReadModelSnapshot {
  return {
    threadRead: threadRead ?? null,
  };
}

export function mergeAgentSessionReadModelThreadItems(
  currentItems: AgentThreadItem[],
  snapshot: AgentSessionReadModelSnapshot,
): AgentThreadItem[] {
  const incomingItems = snapshot.threadRead?.thread_items;
  if (!Array.isArray(incomingItems) || incomingItems.length === 0) {
    return currentItems;
  }
  return mergeRuntimeSyncThreadItems(currentItems, incomingItems);
}

export async function hydrateFreshAgentSessionReadModel(
  runtime: Pick<AgentRuntimeAdapter, "getSessionReadModel">,
  sessionId: string,
): Promise<AgentRuntimeThreadReadModel> {
  const normalizedSessionId = sessionId.trim();
  if (!normalizedSessionId) {
    throw new Error("sessionId is required to hydrate a fresh session");
  }

  const threadRead = await runtime.getSessionReadModel(normalizedSessionId);
  const threadId = threadRead?.thread_id?.trim();
  if (!threadId) {
    throw new Error(
      "fresh session read model did not include a canonical threadId",
    );
  }
  return { ...threadRead, thread_id: threadId };
}

export function resolveDefaultAgentSessionDetailMergeMode(): AgentSessionDetailMergeMode {
  return "history_hydrate";
}

interface RefreshAgentSessionDetailOptions {
  runtime: Pick<AgentRuntimeAdapter, "getSession">;
  sessionIdRef: MutableRefObject<string | null>;
  targetSessionId?: string;
  applySessionDetail: (
    sessionId: string,
    detail: Awaited<ReturnType<AgentRuntimeAdapter["getSession"]>>,
    options: {
      preserveExecutionStrategyOnMissingDetail: boolean;
      detailMergeMode?: AgentSessionDetailMergeMode;
    },
  ) => void;
  markSessionExecutionStrategySynced: (
    sessionId: string,
    executionStrategy: AgentExecutionStrategy,
  ) => void;
  persistSessionAccessMode?: (
    sessionId: string,
    accessMode: AgentAccessMode,
  ) => void;
  setAccessModeState?: (accessMode: AgentAccessMode) => void;
  onWarn?: (error: unknown) => void;
  source?: string | null;
  detailMergeMode?: AgentSessionDetailMergeMode | null;
}

export async function refreshAgentSessionDetailState(
  options: RefreshAgentSessionDetailOptions,
) {
  const {
    runtime,
    sessionIdRef,
    targetSessionId,
    applySessionDetail,
    markSessionExecutionStrategySynced,
    onWarn,
  } = options;
  const resolvedSessionId = targetSessionId || sessionIdRef.current;
  if (!resolvedSessionId?.trim()) {
    return false;
  }

  try {
    const detail = await runtime.getSession(resolvedSessionId, {
      historyLimit: 40,
      ...(options.source?.trim() ? { source: options.source.trim() } : {}),
    });
    if (sessionIdRef.current !== resolvedSessionId) {
      return false;
    }
    applySessionDetail(resolvedSessionId, detail, {
      preserveExecutionStrategyOnMissingDetail: true,
      detailMergeMode:
        options.detailMergeMode ?? resolveDefaultAgentSessionDetailMergeMode(),
    });
    const runtimeAccessMode = createSessionAccessModeFromExecutionRuntime(
      detail.execution_runtime,
    );
    if (runtimeAccessMode) {
      options.persistSessionAccessMode?.(resolvedSessionId, runtimeAccessMode);
      options.setAccessModeState?.(runtimeAccessMode);
    }
    if (detail.execution_strategy) {
      markSessionExecutionStrategySynced(
        resolvedSessionId,
        normalizeExecutionStrategy(detail.execution_strategy),
      );
    }
    return true;
  } catch (error) {
    onWarn?.(error);
    return false;
  }
}

interface RefreshAgentSessionReadModelOptions {
  runtime: Pick<AgentRuntimeAdapter, "getSessionReadModel">;
  sessionIdRef: MutableRefObject<string | null>;
  targetSessionId?: string;
  applyReadModelSnapshot: (snapshot: AgentSessionReadModelSnapshot) => void;
  onWarn?: (error: unknown) => void;
}

export async function refreshAgentSessionReadModelState(
  options: RefreshAgentSessionReadModelOptions,
) {
  const {
    runtime,
    sessionIdRef,
    targetSessionId,
    applyReadModelSnapshot,
    onWarn,
  } = options;
  const resolvedSessionId = targetSessionId || sessionIdRef.current;
  if (!resolvedSessionId?.trim()) {
    return false;
  }

  try {
    const threadRead = await runtime.getSessionReadModel(resolvedSessionId);
    if (sessionIdRef.current !== resolvedSessionId) {
      return false;
    }
    applyReadModelSnapshot(createAgentSessionReadModelSnapshot(threadRead));
    return true;
  } catch (error) {
    onWarn?.(error);
    return false;
  }
}
