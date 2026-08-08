import {
  useEffect,
  useMemo,
  useRef,
  type Dispatch,
  type SetStateAction,
} from "react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { BrowserAssistSessionState } from "../types";
import type { WorkspaceFilesSurfaceTarget } from "./WorkspaceFilesSurface";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";
import type { BrowserSessionRef } from "./workspaceBrowserSessionRef";
import { useWorkspaceRightSurfaceActionRuntime } from "./useWorkspaceRightSurfaceActionRuntime";
import type { WorkspaceRightSurfacePendingActions } from "./useWorkspaceRightSurfaceArtifactOpenRuntime";
import { useWorkspaceRightSurfaceDerivedRuntime } from "./useWorkspaceRightSurfaceDerivedRuntime";
import type { WorkspaceRightSurfaceLocalStateRuntime } from "./useWorkspaceRightSurfaceLocalStateRuntime";
import { useWorkspaceRightSurfacePendingBridgeRuntime } from "./useWorkspaceRightSurfacePendingBridgeRuntime";
import { useWorkspaceRightSurfaceProjectionRuntime } from "./useWorkspaceRightSurfaceProjectionRuntime";
import {
  buildWorkspacePluginSurfacesFromThreadItems,
  mergeWorkspacePluginSurfaceDescriptors,
  resolveWorkspacePluginSurfaceSessionEpoch,
  resolveWorkspacePluginSurfaceThreadId,
  type WorkspacePluginSurfaceSessionEpoch,
} from "./workspacePluginSurfaceModel";

interface UseWorkspaceRightSurfaceCoordinatorRuntimeParams {
  articleEditorRightSurface: WorkspaceArticleWorkspace | null;
  articleEditorRightSurfaceAvailable: boolean;
  bindRightSurfacePendingActions: (
    actions: WorkspaceRightSurfacePendingActions,
  ) => void;
  browserAssistLaunching: boolean;
  browserAssistSessionRef: BrowserSessionRef | null;
  browserAssistSessionState: BrowserAssistSessionState | null;
  canvasWorkbenchRootPath: string | null;
  clawTraceEnabled: boolean;
  currentBrowserAssistScopeKey: string | null;
  expertInfoPanelCollapsed: boolean;
  expertInfoPanelVisible: boolean;
  handleToggleCanvas: () => void;
  harnessPendingCount: number;
  hasExpertInfoPanel: boolean;
  localState: WorkspaceRightSurfaceLocalStateRuntime;
  preferredServiceSkillResultFileTarget: WorkspaceFilesSurfaceTarget | null;
  runtimeWorkspaceId: string | null;
  sceneIsPreparingSend: boolean;
  sceneIsSending: boolean;
  sceneLayoutMode: LayoutMode;
  sceneSessionId?: string | null;
  sceneThreadId?: string | null;
  sessionId?: string | null;
  shellRightSurfaceAvailable: boolean;
  showHarnessToggle: boolean;
  suppressHomeNavbarUtilityActions: boolean;
  taskCenterHomeHotpathActive: boolean;
  threadItems: readonly AgentThreadItem[];
  setExpertInfoPanelCollapsed: (value: boolean) => void;
  setHarnessPanelVisible: (value: boolean) => void;
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
}

export function useWorkspaceRightSurfaceCoordinatorRuntime({
  articleEditorRightSurface,
  articleEditorRightSurfaceAvailable,
  bindRightSurfacePendingActions,
  browserAssistLaunching,
  browserAssistSessionRef,
  browserAssistSessionState,
  canvasWorkbenchRootPath,
  clawTraceEnabled,
  currentBrowserAssistScopeKey,
  expertInfoPanelCollapsed,
  expertInfoPanelVisible,
  handleToggleCanvas,
  harnessPendingCount,
  hasExpertInfoPanel,
  localState,
  preferredServiceSkillResultFileTarget,
  runtimeWorkspaceId,
  sceneIsPreparingSend,
  sceneIsSending,
  sceneLayoutMode,
  sceneSessionId,
  sceneThreadId,
  sessionId,
  shellRightSurfaceAvailable,
  showHarnessToggle,
  suppressHomeNavbarUtilityActions,
  taskCenterHomeHotpathActive,
  threadItems,
  setExpertInfoPanelCollapsed,
  setHarnessPanelVisible,
  setLayoutMode,
}: UseWorkspaceRightSurfaceCoordinatorRuntimeParams) {
  const effectiveSessionId = sessionId || sceneSessionId || null;
  const effectiveSceneThreadId = resolveWorkspacePluginSurfaceThreadId(
    sceneThreadId,
    threadItems,
  );
  const pluginSurfaceSessionEpochRef =
    useRef<WorkspacePluginSurfaceSessionEpoch | null>(null);
  const pluginSurfaceSessionEpoch = resolveWorkspacePluginSurfaceSessionEpoch({
    currentSessionId: effectiveSessionId,
    currentThreadId: effectiveSceneThreadId,
    previousEpoch: pluginSurfaceSessionEpochRef.current,
  });
  useEffect(() => {
    if (pluginSurfaceSessionEpoch.ready) {
      pluginSurfaceSessionEpochRef.current = pluginSurfaceSessionEpoch.epoch;
    }
  }, [pluginSurfaceSessionEpoch.epoch, pluginSurfaceSessionEpoch.ready]);
  const canonicalPluginSurfaces = useMemo(
    () =>
      pluginSurfaceSessionEpoch.ready
        ? buildWorkspacePluginSurfacesFromThreadItems(
            threadItems,
            localState.dismissedPluginSurfaceContainerIds,
            pluginSurfaceSessionEpoch.epoch?.threadId,
          )
        : [],
    [
      localState.dismissedPluginSurfaceContainerIds,
      pluginSurfaceSessionEpoch.epoch?.threadId,
      pluginSurfaceSessionEpoch.ready,
      threadItems,
    ],
  );
  const pendingRuntime = useWorkspaceRightSurfacePendingBridgeRuntime({
    bindRightSurfacePendingActions,
    canvasWorkbenchRootPath,
    manualRightSurface: localState.manualRightSurface,
    runtimeWorkspaceId,
    sceneIsPreparingSend,
    sceneIsSending,
    sceneLayoutMode,
    sceneSessionId,
    sessionId,
    taskCenterHomeHotpathActive,
  });
  const mergedIncomingPluginSurfaces = useMemo(
    () =>
      mergeWorkspacePluginSurfaceDescriptors(
        pendingRuntime.pendingPluginSurfaces,
        canonicalPluginSurfaces,
      ),
    [canonicalPluginSurfaces, pendingRuntime.pendingPluginSurfaces],
  );
  const {
    setActivePluginSurfaceContainerId,
    setActivePluginSurfaces,
    setDismissedPluginSurfaceContainerIds,
    setManualRightSurface,
  } = localState;
  useEffect(() => {
    setActivePluginSurfaces([]);
    setActivePluginSurfaceContainerId(null);
    setDismissedPluginSurfaceContainerIds([]);
    setManualRightSurface((current) =>
      current === "appSurface" ? null : current,
    );
  }, [
    effectiveSessionId,
    setActivePluginSurfaceContainerId,
    setActivePluginSurfaces,
    setDismissedPluginSurfaceContainerIds,
    setManualRightSurface,
  ]);
  const derivedRuntime = useWorkspaceRightSurfaceDerivedRuntime({
    activeBrowserRightSurfaceIntent: localState.activeBrowserRightSurfaceIntent,
    activeFilesRightSurfaceTarget: localState.activeFilesRightSurfaceTarget,
    activeObjectCanvasRightSurfaceCandidate:
      localState.activeObjectCanvasRightSurfaceCandidate,
    activePluginSurfaceContainerId: localState.activePluginSurfaceContainerId,
    activePluginSurfaces: localState.activePluginSurfaces,
    browserAssistLaunching,
    browserAssistSessionRef,
    browserAssistSessionState,
    currentBrowserAssistScopeKey,
    pendingBrowserRightSurfaceIntent: pendingRuntime.pendingBrowserIntent,
    pendingFileTarget: pendingRuntime.pendingFileTarget,
    pendingObjectCanvasCandidate: pendingRuntime.pendingObjectCanvasCandidate,
    pendingPluginSurfaces: mergedIncomingPluginSurfaces,
    preferredServiceSkillResultFileTarget,
  });
  const rightSurfaceHarnessEnabled =
    !suppressHomeNavbarUtilityActions && showHarnessToggle;
  const rightSurfaceTraceAvailable = !suppressHomeNavbarUtilityActions;
  const rightSurfaceTraceEnabled =
    rightSurfaceTraceAvailable && clawTraceEnabled;
  const { rightSurfaceLaunchers, rightSurfaceState } =
    useWorkspaceRightSurfaceProjectionRuntime({
      appServerPendingIntents: pendingRuntime.pendingIntents,
      appSurfaceAvailable: derivedRuntime.pluginSurfaceRightSurfaceAvailable,
      articleWorkspaceAvailable: articleEditorRightSurfaceAvailable,
      expertInfoVisible: expertInfoPanelVisible,
      filesAvailable: derivedRuntime.filesRightSurfaceAvailable,
      harnessPendingCount,
      hasExpertInfoPanel,
      manualRightSurface: localState.manualRightSurface,
      objectCanvasAvailable: derivedRuntime.objectCanvasRightSurfaceAvailable,
      objectCanvasCandidateId:
        derivedRuntime.browserAssistObjectCanvasCandidateId,
      preferredServiceSkillResultFileTargetRelativePath:
        preferredServiceSkillResultFileTarget?.relativePath,
      sceneLayoutMode,
      shellAvailable: shellRightSurfaceAvailable,
      showHarnessToggle,
      suppressHomeNavbarUtilityActions,
      traceAvailable: rightSurfaceTraceAvailable,
    });
  const actionRuntime = useWorkspaceRightSurfaceActionRuntime({
    activeBrowserRightSurfaceIntent: localState.activeBrowserRightSurfaceIntent,
    activePluginSurfaceContainerId: localState.activePluginSurfaceContainerId,
    activePluginSurfaces: localState.activePluginSurfaces,
    articleEditorRightSurface,
    articleEditorRightSurfaceAvailable,
    browserRightSurfaceAvailable: derivedRuntime.browserRightSurfaceAvailable,
    consumePendingRequestsForSurface:
      pendingRuntime.consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface:
      pendingRuntime.dismissPendingRequestsForSurface,
    expertInfoPanelCollapsed,
    filesRightSurfaceAvailable: derivedRuntime.filesRightSurfaceAvailable,
    filesRightSurfaceTarget: derivedRuntime.filesRightSurfaceTarget,
    handleToggleCanvas,
    manualRightSurface: localState.manualRightSurface,
    objectCanvasRightSurfaceAvailable:
      derivedRuntime.objectCanvasRightSurfaceAvailable,
    objectCanvasRightSurfaceCandidate:
      derivedRuntime.objectCanvasRightSurfaceCandidate,
    pendingBrowserRightSurfaceIntent: pendingRuntime.pendingBrowserIntent,
    pendingPluginSurfaces: mergedIncomingPluginSurfaces,
    pluginSurfaceRightSurface: derivedRuntime.pluginSurfaceRightSurface,
    pluginSurfaceRightSurfaceAvailable:
      derivedRuntime.pluginSurfaceRightSurfaceAvailable,
    pluginSurfaceRightSurfaces: derivedRuntime.pluginSurfaceRightSurfaces,
    refreshRightSurfacePendingRequests: pendingRuntime.refreshPendingRequests,
    rightSurfaceActiveSurface: rightSurfaceState.activeSurface,
    rightSurfaceHarnessEnabled,
    rightSurfaceTraceAvailable,
    sceneLayoutMode,
    setActiveArticleWorkspace: localState.setActiveArticleWorkspace,
    setActiveBrowserRightSurfaceIntent:
      localState.setActiveBrowserRightSurfaceIntent,
    setActiveFilesRightSurfaceTarget:
      localState.setActiveFilesRightSurfaceTarget,
    setActiveObjectCanvasRightSurfaceCandidate:
      localState.setActiveObjectCanvasRightSurfaceCandidate,
    setActivePluginSurfaceContainerId:
      localState.setActivePluginSurfaceContainerId,
    setActivePluginSurfaces: localState.setActivePluginSurfaces,
    setDismissedPluginSurfaceContainerIds:
      localState.setDismissedPluginSurfaceContainerIds,
    setExpertInfoPanelCollapsed,
    setHarnessPanelVisible,
    setLayoutMode,
    setManualRightSurface: localState.setManualRightSurface,
    setRightSurfaceBrowserTitle: localState.setRightSurfaceBrowserTitle,
  });

  return {
    ...derivedRuntime,
    ...actionRuntime,
    activePluginSurfaceContainerId: localState.activePluginSurfaceContainerId,
    rightSurfaceBrowserTitle: localState.rightSurfaceBrowserTitle,
    rightSurfaceHarnessEnabled,
    rightSurfaceLaunchers,
    rightSurfaceState,
    rightSurfaceTraceAvailable,
    rightSurfaceTraceEnabled,
  };
}

export type WorkspaceRightSurfaceCoordinatorRuntime = ReturnType<
  typeof useWorkspaceRightSurfaceCoordinatorRuntime
>;
