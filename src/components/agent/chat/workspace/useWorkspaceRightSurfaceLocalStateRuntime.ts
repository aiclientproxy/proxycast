import { useCallback, useState } from "react";
import type { WorkspaceFilesSurfaceTarget } from "./WorkspaceFilesSurface";
import type { WorkspaceRightSurfaceKind } from "./right-surface";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";
import type { WorkspacePluginSurfaceDescriptor } from "./workspacePluginSurfaceModel";
import type { WorkspaceRightSurfaceBrowserIntent } from "./workspaceRightSurfaceBrowserIntent";

export function useWorkspaceRightSurfaceLocalStateRuntime() {
  const [manualRightSurface, setManualRightSurface] =
    useState<WorkspaceRightSurfaceKind | null>(null);
  const [activeFilesRightSurfaceTarget, setActiveFilesRightSurfaceTarget] =
    useState<WorkspaceFilesSurfaceTarget | null>(null);
  const [activeArticleWorkspace, setActiveArticleWorkspace] =
    useState<WorkspaceArticleWorkspace | null>(null);
  const [rightSurfaceBrowserTitle, setRightSurfaceBrowserTitle] = useState<
    string | null
  >(null);
  const [activeBrowserRightSurfaceIntent, setActiveBrowserRightSurfaceIntent] =
    useState<WorkspaceRightSurfaceBrowserIntent | null>(null);
  const [activePluginSurfaces, setActivePluginSurfaces] = useState<
    WorkspacePluginSurfaceDescriptor[]
  >([]);
  const [activePluginSurfaceContainerId, setActivePluginSurfaceContainerId] =
    useState<string | null>(null);
  const [
    dismissedPluginSurfaceContainerIds,
    setDismissedPluginSurfaceContainerIds,
  ] = useState<string[]>([]);

  const clearActiveRightSurfaceTargets = useCallback(() => {
    setManualRightSurface(null);
    setActiveFilesRightSurfaceTarget(null);
    setActiveArticleWorkspace(null);
  }, []);

  const openArticleWorkspaceRightSurface = useCallback(
    (articleWorkspace: WorkspaceArticleWorkspace) => {
      setActiveFilesRightSurfaceTarget(null);
      setActiveArticleWorkspace(articleWorkspace);
      setManualRightSurface("articleWorkspace");
    },
    [],
  );

  return {
    activeArticleWorkspace,
    activeBrowserRightSurfaceIntent,
    activeFilesRightSurfaceTarget,
    activePluginSurfaceContainerId,
    activePluginSurfaces,
    clearActiveRightSurfaceTargets,
    dismissedPluginSurfaceContainerIds,
    manualRightSurface,
    openArticleWorkspaceRightSurface,
    rightSurfaceBrowserTitle,
    setActiveArticleWorkspace,
    setActiveBrowserRightSurfaceIntent,
    setActiveFilesRightSurfaceTarget,
    setActivePluginSurfaceContainerId,
    setActivePluginSurfaces,
    setDismissedPluginSurfaceContainerIds,
    setManualRightSurface,
    setRightSurfaceBrowserTitle,
  };
}

export type WorkspaceRightSurfaceLocalStateRuntime = ReturnType<
  typeof useWorkspaceRightSurfaceLocalStateRuntime
>;
