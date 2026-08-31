import { useMemo } from "react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import {
  resolveWorkspaceRightSurfaceState,
  type WorkspaceRightSurfaceIntent,
  type WorkspaceRightSurfaceKind,
  type WorkspaceRightSurfaceLauncherProjection,
  type WorkspaceRightSurfaceState,
} from "./right-surface";
import {
  buildWorkspaceRightSurfaceRuntimeLaunchers,
  buildWorkspaceRightSurfaceRuntimePendingIntents,
  hasWorkspaceRightSurfaceRuntimePendingSignals,
} from "./workspaceRightSurfaceRuntimeProjection";

interface UseWorkspaceRightSurfaceProjectionRuntimeParams {
  appServerPendingIntents: readonly WorkspaceRightSurfaceIntent[];
  appSurfaceAvailable: boolean;
  articleWorkspaceAvailable: boolean;
  expertInfoVisible: boolean;
  filesAvailable: boolean;
  harnessPendingCount: number;
  hasExpertInfoPanel: boolean;
  manualRightSurface: WorkspaceRightSurfaceKind | null;
  preferredServiceSkillResultFileTargetRelativePath?: string | null;
  sceneLayoutMode: LayoutMode;
  shellAvailable: boolean;
  showHarnessToggle: boolean;
  suppressHomeNavbarUtilityActions: boolean;
  traceAvailable: boolean;
  activityAvailable?: boolean;
}

interface UseWorkspaceRightSurfaceProjectionRuntimeResult {
  rightSurfaceLaunchers: WorkspaceRightSurfaceLauncherProjection[];
  rightSurfacePendingIntents: WorkspaceRightSurfaceIntent[];
  rightSurfaceState: WorkspaceRightSurfaceState;
}

function resolveWorkspaceRightSurfaceOpenSurfaces({
  appSurfaceAvailable,
  filesAvailable,
  hasExpertInfoPanel,
  manualRightSurface,
  sceneLayoutMode,
  shellAvailable,
  harnessAvailable,
  traceAvailable,
  activityAvailable,
}: {
  appSurfaceAvailable: boolean;
  filesAvailable: boolean;
  hasExpertInfoPanel: boolean;
  manualRightSurface: WorkspaceRightSurfaceKind | null;
  sceneLayoutMode: LayoutMode;
  shellAvailable: boolean;
  harnessAvailable: boolean;
  traceAvailable: boolean;
  activityAvailable: boolean;
}): WorkspaceRightSurfaceKind[] {
  const next: WorkspaceRightSurfaceKind[] = [];
  const add = (kind: WorkspaceRightSurfaceKind, enabled: boolean) => {
    if (enabled && !next.includes(kind)) {
      next.push(kind);
    }
  };

  add("workbench", sceneLayoutMode !== "chat");
  add("appSurface", appSurfaceAvailable);
  add("expertInfo", hasExpertInfoPanel);
  add("files", filesAvailable);
  add("shell", shellAvailable);
  add("harness", harnessAvailable);
  add("trace", traceAvailable);
  add("activity", activityAvailable);
  add("articleWorkspace", manualRightSurface === "articleWorkspace");
  add("files", manualRightSurface === "files");
  add("shell", manualRightSurface === "shell");
  add("harness", manualRightSurface === "harness");
  add("trace", manualRightSurface === "trace");
  add("activity", manualRightSurface === "activity");
  add("activity", manualRightSurface === "activity");
  add("appSurface", manualRightSurface === "appSurface");
  add("expertInfo", manualRightSurface === "expertInfo");
  add("browser", manualRightSurface === "browser");
  return next;
}

export function useWorkspaceRightSurfaceProjectionRuntime({
  appServerPendingIntents,
  appSurfaceAvailable,
  articleWorkspaceAvailable,
  expertInfoVisible,
  filesAvailable,
  harnessPendingCount,
  hasExpertInfoPanel,
  manualRightSurface,
  preferredServiceSkillResultFileTargetRelativePath,
  sceneLayoutMode,
  shellAvailable,
  showHarnessToggle,
  suppressHomeNavbarUtilityActions,
  traceAvailable,
  activityAvailable,
}: UseWorkspaceRightSurfaceProjectionRuntimeParams): UseWorkspaceRightSurfaceProjectionRuntimeResult {
  const effectiveActivityAvailable = activityAvailable === true;
  const harnessAvailable =
    !suppressHomeNavbarUtilityActions && showHarnessToggle;
  const effectiveTraceAvailable =
    !suppressHomeNavbarUtilityActions && traceAvailable;
  const openSurfaces = useMemo(
    () =>
      resolveWorkspaceRightSurfaceOpenSurfaces({
        appSurfaceAvailable,
        filesAvailable,
        hasExpertInfoPanel,
        manualRightSurface,
        sceneLayoutMode,
        shellAvailable,
        harnessAvailable,
        traceAvailable: effectiveTraceAvailable,
        activityAvailable: effectiveActivityAvailable,
      }),
    [
      appSurfaceAvailable,
      effectiveTraceAvailable,
      filesAvailable,
      harnessAvailable,
      hasExpertInfoPanel,
      manualRightSurface,
      sceneLayoutMode,
      shellAvailable,
      effectiveActivityAvailable,
    ],
  );
  const rightSurfaceState = useMemo(
    () =>
      resolveWorkspaceRightSurfaceState({
        layoutMode: sceneLayoutMode,
        hasExpertInfo: hasExpertInfoPanel,
        expertInfoVisible,
        openSurfaces,
        requestedSurface: manualRightSurface ?? undefined,
        source: manualRightSurface ? "user" : undefined,
      }),
    [
      expertInfoVisible,
      hasExpertInfoPanel,
      manualRightSurface,
      openSurfaces,
      sceneLayoutMode,
    ],
  );
  const runtimePendingIntents = useMemo(() => {
    const params = {
      createdAt: Date.now(),
      harnessPendingCount,
      preferredServiceSkillResultFileTargetRelativePath,
      showHarnessToggle,
      suppressHomeNavbarUtilityActions,
    };
    return hasWorkspaceRightSurfaceRuntimePendingSignals(params)
      ? buildWorkspaceRightSurfaceRuntimePendingIntents(params)
      : [];
  }, [
    harnessPendingCount,
    preferredServiceSkillResultFileTargetRelativePath,
    showHarnessToggle,
    suppressHomeNavbarUtilityActions,
  ]);
  const rightSurfacePendingIntents = useMemo(
    () => [...runtimePendingIntents, ...appServerPendingIntents],
    [appServerPendingIntents, runtimePendingIntents],
  );
  const rightSurfaceLaunchers = useMemo(
    () =>
      buildWorkspaceRightSurfaceRuntimeLaunchers({
        surfaceState: rightSurfaceState,
        pendingIntents: rightSurfacePendingIntents,
        filesAvailable,
        appSurfaceAvailable,
        hasExpertInfoPanel,
        articleWorkspaceAvailable,
        shellAvailable,
        showHarnessToggle,
        traceAvailable,
        activityAvailable: effectiveActivityAvailable,
        suppressHomeNavbarUtilityActions,
      }),
    [
      appSurfaceAvailable,
      articleWorkspaceAvailable,
      filesAvailable,
      hasExpertInfoPanel,
      rightSurfacePendingIntents,
      rightSurfaceState,
      shellAvailable,
      showHarnessToggle,
      suppressHomeNavbarUtilityActions,
      traceAvailable,
      effectiveActivityAvailable,
    ],
  );

  return {
    rightSurfaceLaunchers,
    rightSurfacePendingIntents,
    rightSurfaceState,
  };
}
