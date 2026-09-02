import { useCallback, type Dispatch, type SetStateAction } from "react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import type { WorkspaceWorkbenchRequestsController } from "../hooks/useWorkspaceWorkbenchRequests";

export interface UseWorkspaceArtifactSurfaceRuntimeParams {
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  workbenchRequests: WorkspaceWorkbenchRequestsController;
}

export function useWorkspaceArtifactSurfaceRuntime({
  setLayoutMode,
  workbenchRequests,
}: UseWorkspaceArtifactSurfaceRuntimeParams) {
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
    handleJumpToTimelineItem,
    serviceSkillExecutionCard: null,
  };
}
