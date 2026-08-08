import {
  buildWorkspaceRightSurfaceFilePreviewIntents,
  buildWorkspaceRightSurfaceHarnessPendingIntents,
  buildWorkspaceRightSurfaceLauncherProjections,
  buildWorkspaceRightSurfaceObjectCanvasCandidateIntents,
  type WorkspaceRightSurfaceIntent,
  type WorkspaceRightSurfaceKind,
  type WorkspaceRightSurfaceLauncherProjection,
  type WorkspaceRightSurfaceState,
} from "./right-surface";
export interface BuildWorkspaceRightSurfaceRuntimePendingIntentsParams {
  createdAt: number;
  harnessPendingCount: number;
  objectCanvasCandidateId?: string | null;
  preferredServiceSkillResultFileTargetRelativePath?: string | null;
  showHarnessToggle: boolean;
  suppressHomeNavbarUtilityActions: boolean;
}

export interface BuildWorkspaceRightSurfaceRuntimeLaunchersParams {
  filesAvailable: boolean;
  appSurfaceAvailable?: boolean;
  hasExpertInfoPanel: boolean;
  objectCanvasAvailable: boolean;
  articleWorkspaceAvailable?: boolean;
  pendingIntents: WorkspaceRightSurfaceIntent[];
  shellAvailable: boolean;
  showHarnessToggle: boolean;
  traceAvailable?: boolean;
  suppressHomeNavbarUtilityActions: boolean;
  surfaceState: WorkspaceRightSurfaceState;
}

export function hasWorkspaceRightSurfaceRuntimePendingSignals({
  harnessPendingCount,
  objectCanvasCandidateId,
  preferredServiceSkillResultFileTargetRelativePath,
  showHarnessToggle,
  suppressHomeNavbarUtilityActions,
}: Omit<BuildWorkspaceRightSurfaceRuntimePendingIntentsParams, "createdAt">): boolean {
  if (
    !suppressHomeNavbarUtilityActions &&
    showHarnessToggle &&
    harnessPendingCount > 0
  ) {
    return true;
  }
  if (preferredServiceSkillResultFileTargetRelativePath?.trim()) {
    return true;
  }
  if (objectCanvasCandidateId?.trim()) {
    return true;
  }
  return false;
}

export function buildWorkspaceRightSurfaceRuntimePendingIntents({
  createdAt,
  harnessPendingCount,
  objectCanvasCandidateId,
  preferredServiceSkillResultFileTargetRelativePath,
  showHarnessToggle,
  suppressHomeNavbarUtilityActions,
}: BuildWorkspaceRightSurfaceRuntimePendingIntentsParams): WorkspaceRightSurfaceIntent[] {
  return [
    ...buildWorkspaceRightSurfaceHarnessPendingIntents({
      enabled: !suppressHomeNavbarUtilityActions && showHarnessToggle,
      pendingCount: harnessPendingCount,
      createdAt,
    }),
    ...buildWorkspaceRightSurfaceFilePreviewIntents({
      enabled: Boolean(preferredServiceSkillResultFileTargetRelativePath),
      relativePath: preferredServiceSkillResultFileTargetRelativePath,
      createdAt,
    }),
    ...buildWorkspaceRightSurfaceObjectCanvasCandidateIntents({
      enabled: Boolean(objectCanvasCandidateId),
      candidateId: objectCanvasCandidateId,
      createdAt,
    }),
  ];
}

export function buildWorkspaceRightSurfaceRuntimeAvailableSurfaces({
  filesAvailable,
  appSurfaceAvailable = false,
  hasExpertInfoPanel,
  objectCanvasAvailable,
  articleWorkspaceAvailable = false,
  shellAvailable,
  showHarnessToggle,
  traceAvailable = false,
  suppressHomeNavbarUtilityActions,
}: Pick<
  BuildWorkspaceRightSurfaceRuntimeLaunchersParams,
  | "hasExpertInfoPanel"
  | "appSurfaceAvailable"
  | "filesAvailable"
  | "objectCanvasAvailable"
  | "articleWorkspaceAvailable"
  | "shellAvailable"
  | "showHarnessToggle"
  | "traceAvailable"
  | "suppressHomeNavbarUtilityActions"
>): ReadonlySet<WorkspaceRightSurfaceKind> {
  const surfaces: WorkspaceRightSurfaceKind[] = ["workbench"];
  if (appSurfaceAvailable) {
    surfaces.push("appSurface");
  }
  if (articleWorkspaceAvailable) {
    surfaces.push("articleWorkspace");
  }
  if (hasExpertInfoPanel) {
    surfaces.push("expertInfo");
  }
  if (objectCanvasAvailable) {
    surfaces.push("objectCanvas");
  }
  surfaces.push("browser");
  if (filesAvailable) {
    surfaces.push("files");
  }
  if (!suppressHomeNavbarUtilityActions && shellAvailable) {
    surfaces.push("shell");
  }
  if (!suppressHomeNavbarUtilityActions && showHarnessToggle) {
    surfaces.push("harness");
  }
  if (!suppressHomeNavbarUtilityActions && traceAvailable) {
    surfaces.push("trace");
  }
  return new Set(surfaces);
}

export function buildWorkspaceRightSurfaceRuntimeLaunchers({
  filesAvailable,
  appSurfaceAvailable,
  hasExpertInfoPanel,
  objectCanvasAvailable,
  articleWorkspaceAvailable,
  pendingIntents,
  shellAvailable,
  showHarnessToggle,
  traceAvailable,
  suppressHomeNavbarUtilityActions,
  surfaceState,
}: BuildWorkspaceRightSurfaceRuntimeLaunchersParams): WorkspaceRightSurfaceLauncherProjection[] {
  return buildWorkspaceRightSurfaceLauncherProjections({
    surfaceState,
    pendingIntents,
    availableSurfaces: buildWorkspaceRightSurfaceRuntimeAvailableSurfaces({
      filesAvailable,
      appSurfaceAvailable,
      hasExpertInfoPanel,
      objectCanvasAvailable,
      articleWorkspaceAvailable,
      shellAvailable,
      showHarnessToggle,
      traceAvailable,
      suppressHomeNavbarUtilityActions,
    }),
  });
}
