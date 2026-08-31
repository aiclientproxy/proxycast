import {
  buildWorkspaceRightSurfaceFilePreviewIntents,
  buildWorkspaceRightSurfaceHarnessPendingIntents,
  buildWorkspaceRightSurfaceLauncherProjections,
  type WorkspaceRightSurfaceIntent,
  type WorkspaceRightSurfaceKind,
  type WorkspaceRightSurfaceLauncherProjection,
  type WorkspaceRightSurfaceState,
} from "./right-surface";
export interface BuildWorkspaceRightSurfaceRuntimePendingIntentsParams {
  createdAt: number;
  harnessPendingCount: number;
  preferredServiceSkillResultFileTargetRelativePath?: string | null;
  showHarnessToggle: boolean;
  suppressHomeNavbarUtilityActions: boolean;
}

export interface BuildWorkspaceRightSurfaceRuntimeLaunchersParams {
  filesAvailable: boolean;
  appSurfaceAvailable?: boolean;
  hasExpertInfoPanel: boolean;
  articleWorkspaceAvailable?: boolean;
  pendingIntents: WorkspaceRightSurfaceIntent[];
  shellAvailable: boolean;
  showHarnessToggle: boolean;
  traceAvailable?: boolean;
  activityAvailable?: boolean;
  suppressHomeNavbarUtilityActions: boolean;
  surfaceState: WorkspaceRightSurfaceState;
}

export function hasWorkspaceRightSurfaceRuntimePendingSignals({
  harnessPendingCount,
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
  return false;
}

export function buildWorkspaceRightSurfaceRuntimePendingIntents({
  createdAt,
  harnessPendingCount,
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
  ];
}

export function buildWorkspaceRightSurfaceRuntimeAvailableSurfaces({
  filesAvailable,
  appSurfaceAvailable = false,
  hasExpertInfoPanel,
  articleWorkspaceAvailable = false,
  shellAvailable,
  showHarnessToggle,
  traceAvailable = false,
  activityAvailable = false,
  suppressHomeNavbarUtilityActions,
}: Pick<
  BuildWorkspaceRightSurfaceRuntimeLaunchersParams,
  | "hasExpertInfoPanel"
  | "appSurfaceAvailable"
  | "filesAvailable"
  | "articleWorkspaceAvailable"
  | "shellAvailable"
  | "showHarnessToggle"
  | "traceAvailable"
  | "activityAvailable"
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
  if (!suppressHomeNavbarUtilityActions && activityAvailable) {
    surfaces.push("activity");
  }
  return new Set(surfaces);
}

export function buildWorkspaceRightSurfaceRuntimeLaunchers({
  filesAvailable,
  appSurfaceAvailable,
  hasExpertInfoPanel,
  articleWorkspaceAvailable,
  pendingIntents,
  shellAvailable,
  showHarnessToggle,
  traceAvailable,
  activityAvailable,
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
      articleWorkspaceAvailable,
      shellAvailable,
      showHarnessToggle,
      traceAvailable,
      activityAvailable,
      suppressHomeNavbarUtilityActions,
    }),
  });
}
