import { useEffect, type MutableRefObject } from "react";
import { useWorkspaceWriteFileAction } from "./useWorkspaceWriteFileAction";
import {
  useWorkspaceArtifactOpenRuntime,
  type UseWorkspaceArtifactOpenRuntimeParams,
} from "./useWorkspaceArtifactOpenRuntime";
import type { WorkspaceArtifactWriteFile } from "./useWorkspaceArtifactDocumentSaveRuntime";

type WorkspaceWriteFileParams = Parameters<
  typeof useWorkspaceWriteFileAction
>[0];

export type UseWorkspaceArtifactActionRuntimeParams = WorkspaceWriteFileParams &
  Omit<UseWorkspaceArtifactOpenRuntimeParams, "handleWriteFile"> & {
    handleWriteFileRef: MutableRefObject<
      WorkspaceArtifactWriteFile | undefined
    >;
  };

export function useWorkspaceArtifactActionRuntime({
  activeTheme,
  artifacts,
  contentId,
  currentCanvasArtifact,
  currentGateKey,
  currentTurnId,
  effectiveThreadItems,
  generalCanvasState,
  handleWriteFileRef,
  initialProjectFileOpenTarget,
  isInitialContentLoading,
  isThemeWorkbench,
  layoutMode,
  mappedTheme,
  messages,
  onNavigate,
  openArticleWorkspaceRightSurface,
  projectId,
  projectRootPath,
  readSessionFile,
  saveSessionFile,
  sessionFiles,
  sessionId,
  threadId,
  setArtifactViewMode,
  setCanvasState,
  setCanvasWorkbenchLayoutMode,
  setDocumentVersionStatusMap,
  setExpertInfoPanelCollapsed,
  setGeneralCanvasState,
  setHarnessPanelVisible,
  setLayoutMode,
  setSelectedArtifactId,
  setSelectedFileId,
  setTaskFiles,
  socialStageLogRef,
  suppressCanvasAutoOpen,
  syncGeneralArtifactToResource,
  taskFiles,
  taskFilesRef,
  themeWorkbenchActiveQueueItem,
  updateCurrentImageWorkbenchState,
  upsertGeneralArtifact,
  workbenchRequests,
}: UseWorkspaceArtifactActionRuntimeParams) {
  const handleWriteFile = useWorkspaceWriteFileAction({
    activeTheme,
    artifacts,
    contentId,
    currentGateKey,
    isThemeWorkbench,
    mappedTheme,
    projectId,
    sessionId,
    themeWorkbenchActiveQueueItem,
    taskFilesRef,
    socialStageLogRef,
    setDocumentVersionStatusMap,
    saveSessionFile,
    syncGeneralArtifactToResource,
    upsertGeneralArtifact,
    setSelectedArtifactId,
    setArtifactViewMode,
    setLayoutMode,
    suppressCanvasAutoOpen,
    setTaskFiles,
    setSelectedFileId,
    setCanvasState,
  });

  useEffect(() => {
    handleWriteFileRef.current = handleWriteFile;
  }, [handleWriteFile, handleWriteFileRef]);

  const artifactOpenRuntime = useWorkspaceArtifactOpenRuntime({
    activeTheme,
    artifacts,
    contentId,
    currentCanvasArtifact,
    currentTurnId,
    effectiveThreadItems,
    generalCanvasState,
    handleWriteFile,
    initialProjectFileOpenTarget,
    isInitialContentLoading,
    isThemeWorkbench,
    layoutMode,
    mappedTheme,
    messages,
    onNavigate,
    openArticleWorkspaceRightSurface,
    projectId,
    projectRootPath,
    readSessionFile,
    sessionFiles,
    threadId,
    setArtifactViewMode,
    setCanvasState,
    setCanvasWorkbenchLayoutMode,
    setExpertInfoPanelCollapsed,
    setGeneralCanvasState,
    setHarnessPanelVisible,
    setLayoutMode,
    setSelectedArtifactId,
    setSelectedFileId,
    setTaskFiles,
    syncGeneralArtifactToResource,
    taskFiles,
    updateCurrentImageWorkbenchState,
    upsertGeneralArtifact,
    workbenchRequests,
  });

  return {
    handleWriteFile,
    ...artifactOpenRuntime,
  };
}
