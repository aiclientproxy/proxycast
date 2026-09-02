import { useWorkspaceArtifactActionRuntime } from "./useWorkspaceArtifactActionRuntime";
import {
  useWorkspaceArtifactSurfaceRuntime,
  type UseWorkspaceArtifactSurfaceRuntimeParams,
} from "./useWorkspaceArtifactSurfaceRuntime";

type ArtifactActionParams = Parameters<
  typeof useWorkspaceArtifactActionRuntime
>[0];
type ArtifactSurfaceParams = UseWorkspaceArtifactSurfaceRuntimeParams;

export type UseAgentChatWorkspaceArtifactInteractionRuntimeParams = {
  action: ArtifactActionParams;
  surface: {
    setLayoutMode: ArtifactSurfaceParams["setLayoutMode"];
    workbenchRequests: ArtifactSurfaceParams["workbenchRequests"];
  };
};

/** 统一 Artifact action 与 surface 的接线，避免父级重建同一组打开/预览 handler。 */
export function useAgentChatWorkspaceArtifactInteractionRuntime({
  action,
  surface,
}: UseAgentChatWorkspaceArtifactInteractionRuntimeParams) {
  const artifactActionRuntime = useWorkspaceArtifactActionRuntime(action);
  const artifactSurfaceRuntime = useWorkspaceArtifactSurfaceRuntime({
    setLayoutMode: surface.setLayoutMode,
    workbenchRequests: surface.workbenchRequests,
  });

  return {
    ...artifactActionRuntime,
    ...artifactSurfaceRuntime,
  };
}
