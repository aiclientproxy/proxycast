import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import { shouldAutoRefreshWorkspaceRightSurfacePending } from "./agentChatWorkspaceHelpers";
import type { WorkspaceRightSurfaceKind } from "./right-surface";
import type { WorkspaceRightSurfacePendingActions } from "./useWorkspaceRightSurfaceArtifactOpenRuntime";
import {
  useWorkspaceRightSurfacePendingRuntime,
  type WorkspaceRightSurfacePendingRuntime,
} from "./useWorkspaceRightSurfacePendingRuntime";

interface UseWorkspaceRightSurfacePendingBridgeRuntimeParams {
  bindRightSurfacePendingActions: (
    actions: WorkspaceRightSurfacePendingActions,
  ) => void;
  canvasWorkbenchRootPath: string | null;
  manualRightSurface: WorkspaceRightSurfaceKind | null;
  runtimeWorkspaceId: string | null;
  sceneIsPreparingSend: boolean;
  sceneIsSending: boolean;
  sceneLayoutMode: LayoutMode;
  sceneSessionId?: string | null;
  sessionId?: string | null;
  taskCenterHomeHotpathActive: boolean;
}

export function useWorkspaceRightSurfacePendingBridgeRuntime({
  bindRightSurfacePendingActions,
  canvasWorkbenchRootPath,
  manualRightSurface,
  runtimeWorkspaceId,
  sceneIsPreparingSend,
  sceneIsSending,
  sceneLayoutMode,
  sceneSessionId,
  sessionId,
  taskCenterHomeHotpathActive,
}: UseWorkspaceRightSurfacePendingBridgeRuntimeParams): WorkspaceRightSurfacePendingRuntime {
  const rightSurfacePendingSessionId = sessionId || sceneSessionId;
  const shouldAutoRefreshRightSurfacePending =
    shouldAutoRefreshWorkspaceRightSurfacePending({
      sessionId: rightSurfacePendingSessionId,
      workspaceId: runtimeWorkspaceId,
      workspaceRoot: canvasWorkbenchRootPath,
      sceneIsSending,
      sceneIsPreparingSend,
      sceneLayoutMode,
      taskCenterHomeHotpathActive,
      manualRightSurfaceActive: manualRightSurface !== null,
      pluginActivationActive: false,
    });
  const rightSurfaceAppServerPendingRuntime =
    useWorkspaceRightSurfacePendingRuntime({
      enabled: true,
      autoRefreshEnabled: shouldAutoRefreshRightSurfacePending,
      workspaceId: runtimeWorkspaceId,
      workspaceRoot: canvasWorkbenchRootPath,
      sessionId: rightSurfacePendingSessionId,
    });

  bindRightSurfacePendingActions({
    consumePendingRequestsForSurface:
      rightSurfaceAppServerPendingRuntime.consumePendingRequestsForSurface,
    refreshRightSurfacePendingRequests:
      rightSurfaceAppServerPendingRuntime.refreshPendingRequests,
  });

  return rightSurfaceAppServerPendingRuntime;
}
