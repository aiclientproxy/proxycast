import type { WorkspaceFilesSurfaceTarget } from "./WorkspaceFilesSurface";
import type { WorkspacePluginSurfaceDescriptor } from "./workspacePluginSurfaceModel";
import { selectWorkspacePluginSurfaceDescriptor } from "./workspacePluginSurfaceModel";
import type { WorkspaceRightSurfaceBrowserIntent } from "./workspaceRightSurfaceBrowserIntent";

interface UseWorkspaceRightSurfaceDerivedRuntimeParams {
  activeBrowserRightSurfaceIntent: WorkspaceRightSurfaceBrowserIntent | null;
  activeFilesRightSurfaceTarget: WorkspaceFilesSurfaceTarget | null;
  activePluginSurfaceContainerId: string | null;
  activePluginSurfaces: WorkspacePluginSurfaceDescriptor[];
  pendingBrowserRightSurfaceIntent: WorkspaceRightSurfaceBrowserIntent | null;
  pendingFileTarget: WorkspaceFilesSurfaceTarget | null;
  pendingPluginSurfaces: WorkspacePluginSurfaceDescriptor[];
  preferredServiceSkillResultFileTarget: WorkspaceFilesSurfaceTarget | null;
}

interface UseWorkspaceRightSurfaceDerivedRuntimeResult {
  browserRightSurfaceAvailable: boolean;
  browserRightSurfaceIntent: WorkspaceRightSurfaceBrowserIntent | null;
  filesRightSurfaceAvailable: boolean;
  filesRightSurfaceTarget: WorkspaceFilesSurfaceTarget | null;
  pluginSurfaceRightSurface: WorkspacePluginSurfaceDescriptor | null;
  pluginSurfaceRightSurfaceAvailable: boolean;
  pluginSurfaceRightSurfaces: WorkspacePluginSurfaceDescriptor[];
}

export function useWorkspaceRightSurfaceDerivedRuntime({
  activeBrowserRightSurfaceIntent,
  activeFilesRightSurfaceTarget,
  activePluginSurfaceContainerId,
  activePluginSurfaces,
  pendingBrowserRightSurfaceIntent,
  pendingFileTarget,
  pendingPluginSurfaces,
  preferredServiceSkillResultFileTarget,
}: UseWorkspaceRightSurfaceDerivedRuntimeParams): UseWorkspaceRightSurfaceDerivedRuntimeResult {
  const browserRightSurfaceAvailable = true;
  const browserRightSurfaceIntent =
    activeBrowserRightSurfaceIntent ?? pendingBrowserRightSurfaceIntent;
  const liveFilesRightSurfaceTarget: WorkspaceFilesSurfaceTarget | null =
    preferredServiceSkillResultFileTarget ?? pendingFileTarget;
  const pluginSurfaceRightSurfaces =
    activePluginSurfaces.length > 0
      ? activePluginSurfaces
      : pendingPluginSurfaces;
  const pluginSurfaceRightSurface = selectWorkspacePluginSurfaceDescriptor(
    pluginSurfaceRightSurfaces,
    activePluginSurfaceContainerId,
  );
  const pluginSurfaceRightSurfaceAvailable =
    pluginSurfaceRightSurfaces.length > 0;
  const filesRightSurfaceTarget: WorkspaceFilesSurfaceTarget | null =
    activeFilesRightSurfaceTarget ?? liveFilesRightSurfaceTarget;
  const filesRightSurfaceAvailable = Boolean(
    filesRightSurfaceTarget?.relativePath,
  );
  return {
    browserRightSurfaceAvailable,
    browserRightSurfaceIntent,
    filesRightSurfaceAvailable,
    filesRightSurfaceTarget,
    pluginSurfaceRightSurface,
    pluginSurfaceRightSurfaceAvailable,
    pluginSurfaceRightSurfaces,
  };
}
