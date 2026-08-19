import { useCallback, type Dispatch, type SetStateAction } from "react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import type { WorkspaceWorkbenchRequestsController } from "../hooks/useWorkspaceWorkbenchRequests";
import { useWorkspaceSceneAppExecutionSurfaceRuntime } from "./useWorkspaceSceneAppExecutionSurfaceRuntime";

export interface UseWorkspaceArtifactSurfaceRuntimeParams {
  sceneAppExecution: Parameters<
    typeof useWorkspaceSceneAppExecutionSurfaceRuntime
  >[0];
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  workbenchRequests: WorkspaceWorkbenchRequestsController;
}

export function useWorkspaceArtifactSurfaceRuntime({
  sceneAppExecution,
  setLayoutMode,
  workbenchRequests,
}: UseWorkspaceArtifactSurfaceRuntimeParams) {
  const sceneAppExecutionSurfaceRuntime =
    useWorkspaceSceneAppExecutionSurfaceRuntime(sceneAppExecution);
  const handleJumpToTimelineItem = useCallback(
    (itemId: string) => {
      if (!workbenchRequests.jumpToTimelineItem(itemId)) {
        return;
      }

      setLayoutMode((current) =>
        current === "canvas" ? "chat-canvas" : current,
      );
    },
    [setLayoutMode, workbenchRequests],
  );

  return {
    defaultCuratedTaskReferenceEntries:
      sceneAppExecutionSurfaceRuntime.defaultCuratedTaskReferenceEntries,
    defaultCuratedTaskReferenceMemoryIds:
      sceneAppExecutionSurfaceRuntime.defaultCuratedTaskReferenceMemoryIds,
    handleJumpToTimelineItem,
    sceneAppExecutionSummaryCard: sceneAppExecutionSurfaceRuntime.summaryCard,
    sceneAppReviewDecisionDialogNode:
      sceneAppExecutionSurfaceRuntime.reviewDecisionDialogNode,
    serviceSkillExecutionCard: null,
  };
}
