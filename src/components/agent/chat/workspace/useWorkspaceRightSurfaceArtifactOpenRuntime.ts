import { useCallback, useRef } from "react";
import type { Artifact } from "@/lib/artifact/types";
import type { WorkspaceRightSurfaceKind } from "./right-surface";
import { buildArticleWorkspaceForArtifactOpen } from "./workspaceArticleWorkspaceArtifactOpen";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";

type SetBoolean = (value: boolean) => void;

type ConsumePendingRequestsForSurface = (
  surface: WorkspaceRightSurfaceKind,
) => Promise<unknown>;

type RefreshRightSurfacePendingRequests = () => Promise<unknown>;

export interface WorkspaceRightSurfacePendingActions {
  consumePendingRequestsForSurface?: ConsumePendingRequestsForSurface;
  refreshRightSurfacePendingRequests?: RefreshRightSurfacePendingRequests;
}

interface UseWorkspaceRightSurfaceArtifactOpenRuntimeParams {
  clearFocusedArtifactBlock: () => void;
  fallbackOpenArtifact: (artifact: Artifact) => void;
  openArticleWorkspaceRightSurface: (
    articleWorkspace: WorkspaceArticleWorkspace,
  ) => void;
  setExpertInfoPanelCollapsed: SetBoolean;
  setHarnessPanelVisible: SetBoolean;
}

interface WorkspaceRightSurfaceArtifactOpenRuntime {
  bindArticleEditorRightSurface: (
    articleEditorRightSurface: WorkspaceArticleWorkspace | null,
  ) => void;
  bindRightSurfacePendingActions: (
    actions: WorkspaceRightSurfacePendingActions,
  ) => void;
  handleWorkspaceArtifactClick: (artifact: Artifact) => void;
}

export function useWorkspaceRightSurfaceArtifactOpenRuntime({
  clearFocusedArtifactBlock,
  fallbackOpenArtifact,
  openArticleWorkspaceRightSurface,
  setExpertInfoPanelCollapsed,
  setHarnessPanelVisible,
}: UseWorkspaceRightSurfaceArtifactOpenRuntimeParams): WorkspaceRightSurfaceArtifactOpenRuntime {
  const articleEditorRightSurfaceRef =
    useRef<WorkspaceArticleWorkspace | null>(null);
  const rightSurfacePendingActionsRef =
    useRef<WorkspaceRightSurfacePendingActions>({});

  const bindArticleEditorRightSurface = useCallback(
    (articleEditorRightSurface: WorkspaceArticleWorkspace | null) => {
      articleEditorRightSurfaceRef.current = articleEditorRightSurface;
    },
    [],
  );

  const bindRightSurfacePendingActions = useCallback(
    (actions: WorkspaceRightSurfacePendingActions) => {
      rightSurfacePendingActionsRef.current = actions;
    },
    [],
  );

  const handleWorkspaceArtifactClick = useCallback(
    (artifact: Artifact) => {
      const articleWorkspaceFromArtifact = buildArticleWorkspaceForArtifactOpen(
        artifact,
        articleEditorRightSurfaceRef.current,
      );
      if (articleWorkspaceFromArtifact) {
        clearFocusedArtifactBlock();
        setHarnessPanelVisible(false);
        setExpertInfoPanelCollapsed(true);
        openArticleWorkspaceRightSurface(articleWorkspaceFromArtifact);
        void rightSurfacePendingActionsRef.current.refreshRightSurfacePendingRequests?.();
        void rightSurfacePendingActionsRef.current.consumePendingRequestsForSurface?.(
          "articleWorkspace",
        );
        return;
      }

      clearFocusedArtifactBlock();
      fallbackOpenArtifact(artifact);
    },
    [
      clearFocusedArtifactBlock,
      fallbackOpenArtifact,
      openArticleWorkspaceRightSurface,
      setExpertInfoPanelCollapsed,
      setHarnessPanelVisible,
    ],
  );

  return {
    bindArticleEditorRightSurface,
    bindRightSurfacePendingActions,
    handleWorkspaceArtifactClick,
  };
}
