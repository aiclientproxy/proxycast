import {
  useCallback,
  useEffect,
  type Dispatch,
  type SetStateAction,
} from "react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import type { WorkspaceFilesSurfaceTarget } from "./WorkspaceFilesSurface";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";
import {
  closeWorkspacePluginSurfaceDescriptor,
  mergeWorkspacePluginSurfaceDescriptors,
  resolveWorkspacePluginSurfaceActiveContainerId,
  type WorkspacePluginSurfaceDescriptor,
} from "./workspacePluginSurfaceModel";
import type { WorkspaceRightSurfaceKind } from "./right-surface";
import type { WorkspaceRightSurfaceBrowserIntent } from "./workspaceRightSurfaceBrowserIntent";

type ConsumePendingRequestsForSurface = (
  surface: WorkspaceRightSurfaceKind,
) => Promise<unknown>;

type DismissPendingRequestsForSurface = (
  surface: WorkspaceRightSurfaceKind,
  reason?: string,
) => Promise<unknown>;

type RefreshRightSurfacePendingRequests = () => Promise<unknown>;
type SetBoolean = (value: boolean) => void;

interface UseWorkspaceRightSurfaceActionRuntimeParams {
  activeBrowserRightSurfaceIntent: WorkspaceRightSurfaceBrowserIntent | null;
  activePluginSurfaceContainerId: string | null;
  activePluginSurfaces: WorkspacePluginSurfaceDescriptor[];
  articleEditorRightSurface: WorkspaceArticleWorkspace | null;
  articleEditorRightSurfaceAvailable: boolean;
  browserRightSurfaceAvailable: boolean;
  consumePendingRequestsForSurface: ConsumePendingRequestsForSurface;
  dismissPendingRequestsForSurface: DismissPendingRequestsForSurface;
  expertInfoPanelCollapsed: boolean;
  filesRightSurfaceAvailable: boolean;
  filesRightSurfaceTarget: WorkspaceFilesSurfaceTarget | null;
  handleToggleCanvas: () => void;
  manualRightSurface: WorkspaceRightSurfaceKind | null;
  pendingBrowserRightSurfaceIntent: WorkspaceRightSurfaceBrowserIntent | null;
  pendingPluginSurfaces: WorkspacePluginSurfaceDescriptor[];
  pluginSurfaceRightSurface: WorkspacePluginSurfaceDescriptor | null;
  pluginSurfaceRightSurfaceAvailable: boolean;
  pluginSurfaceRightSurfaces: WorkspacePluginSurfaceDescriptor[];
  refreshRightSurfacePendingRequests: RefreshRightSurfacePendingRequests;
  rightSurfaceActiveSurface: WorkspaceRightSurfaceKind | null;
  rightSurfaceHarnessEnabled: boolean;
  rightSurfaceTraceAvailable: boolean;
  sceneLayoutMode: LayoutMode;
  setActiveArticleWorkspace: Dispatch<
    SetStateAction<WorkspaceArticleWorkspace | null>
  >;
  setActiveBrowserRightSurfaceIntent: Dispatch<
    SetStateAction<WorkspaceRightSurfaceBrowserIntent | null>
  >;
  setActiveFilesRightSurfaceTarget: Dispatch<
    SetStateAction<WorkspaceFilesSurfaceTarget | null>
  >;
  setActivePluginSurfaceContainerId: Dispatch<SetStateAction<string | null>>;
  setActivePluginSurfaces: Dispatch<
    SetStateAction<WorkspacePluginSurfaceDescriptor[]>
  >;
  setDismissedPluginSurfaceContainerIds: Dispatch<SetStateAction<string[]>>;
  setExpertInfoPanelCollapsed: SetBoolean;
  setHarnessPanelVisible: SetBoolean;
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  setManualRightSurface: Dispatch<
    SetStateAction<WorkspaceRightSurfaceKind | null>
  >;
  setRightSurfaceBrowserTitle: Dispatch<SetStateAction<string | null>>;
}

interface WorkspaceRightSurfaceActionRuntime {
  handleClosePluginSurface: (surface: WorkspacePluginSurfaceDescriptor) => void;
  handleCloseRightSurfaceShell: () => void;
  handleRightSurfaceBrowserNavigate: (
    url: string,
    title?: string | null,
  ) => void;
  handleSelectPluginSurface: (
    surface: WorkspacePluginSurfaceDescriptor,
  ) => void;
  handleSelectRightSurfaceTab: (kind: WorkspaceRightSurfaceKind) => void;
  handleToggleExpertInfoPanel: () => void;
  handleToggleCanvasFromRightSurface: () => void;
  handleToggleRightSurfaceBrowser: () => void;
  handleToggleRightSurfaceFiles: () => void;
  handleToggleRightSurfaceHarness: () => void;
  handleToggleRightSurfaceShell: () => void;
  handleToggleRightSurfaceTrace: () => void;
}

export function useWorkspaceRightSurfaceActionRuntime({
  activeBrowserRightSurfaceIntent,
  activePluginSurfaceContainerId,
  activePluginSurfaces,
  articleEditorRightSurface,
  articleEditorRightSurfaceAvailable,
  browserRightSurfaceAvailable,
  consumePendingRequestsForSurface,
  dismissPendingRequestsForSurface,
  expertInfoPanelCollapsed,
  filesRightSurfaceAvailable,
  filesRightSurfaceTarget,
  handleToggleCanvas,
  manualRightSurface,
  pendingBrowserRightSurfaceIntent,
  pendingPluginSurfaces,
  pluginSurfaceRightSurface,
  pluginSurfaceRightSurfaceAvailable,
  pluginSurfaceRightSurfaces,
  refreshRightSurfacePendingRequests,
  rightSurfaceActiveSurface,
  rightSurfaceHarnessEnabled,
  rightSurfaceTraceAvailable,
  sceneLayoutMode,
  setActiveArticleWorkspace,
  setActiveBrowserRightSurfaceIntent,
  setActiveFilesRightSurfaceTarget,
  setActivePluginSurfaceContainerId,
  setActivePluginSurfaces,
  setDismissedPluginSurfaceContainerIds,
  setExpertInfoPanelCollapsed,
  setHarnessPanelVisible,
  setLayoutMode,
  setManualRightSurface,
  setRightSurfaceBrowserTitle,
}: UseWorkspaceRightSurfaceActionRuntimeParams): WorkspaceRightSurfaceActionRuntime {
  const closeCompetingRightSurfaces = useCallback(() => {
    setHarnessPanelVisible(false);
    setExpertInfoPanelCollapsed(true);
    setActiveFilesRightSurfaceTarget(null);
    setActiveArticleWorkspace(null);
  }, [
    setActiveArticleWorkspace,
    setActiveFilesRightSurfaceTarget,
    setExpertInfoPanelCollapsed,
    setHarnessPanelVisible,
  ]);

  useEffect(() => {
    if (pendingPluginSurfaces.length === 0) {
      return;
    }
    const hasNewPluginSurface = pendingPluginSurfaces.some(
      (incoming) =>
        !activePluginSurfaces.some(
          (current) => current.containerId === incoming.containerId,
        ),
    );
    if (!hasNewPluginSurface) {
      return;
    }

    setActivePluginSurfaces((current) =>
      mergeWorkspacePluginSurfaceDescriptors(current, pendingPluginSurfaces),
    );
    setActivePluginSurfaceContainerId((current) =>
      resolveWorkspacePluginSurfaceActiveContainerId({
        activeContainerId: current,
        preferredContainerId:
          pendingPluginSurfaces[pendingPluginSurfaces.length - 1]?.containerId,
        surfaces: mergeWorkspacePluginSurfaceDescriptors(
          activePluginSurfaces,
          pendingPluginSurfaces,
        ),
      }),
    );
    setHarnessPanelVisible(false);
    setExpertInfoPanelCollapsed(true);
    setManualRightSurface(
      (current) =>
        current ?? (sceneLayoutMode === "chat" ? "appSurface" : current),
    );
    void refreshRightSurfacePendingRequests();
    void consumePendingRequestsForSurface("appSurface");
  }, [
    activePluginSurfaces,
    consumePendingRequestsForSurface,
    pendingPluginSurfaces,
    refreshRightSurfacePendingRequests,
    sceneLayoutMode,
    setActivePluginSurfaceContainerId,
    setActivePluginSurfaces,
    setExpertInfoPanelCollapsed,
    setHarnessPanelVisible,
    setManualRightSurface,
  ]);

  useEffect(() => {
    if (
      !pendingBrowserRightSurfaceIntent ||
      pendingBrowserRightSurfaceIntent.priority !== "foreground"
    ) {
      return;
    }
    if (
      manualRightSurface === "browser" &&
      activeBrowserRightSurfaceIntent?.sourceRequestId ===
        pendingBrowserRightSurfaceIntent.sourceRequestId
    ) {
      return;
    }

    closeCompetingRightSurfaces();
    setActiveBrowserRightSurfaceIntent(pendingBrowserRightSurfaceIntent);
    setRightSurfaceBrowserTitle(
      pendingBrowserRightSurfaceIntent.title?.trim() || null,
    );
    setManualRightSurface("browser");
    void refreshRightSurfacePendingRequests();
    void consumePendingRequestsForSurface("browser");
  }, [
    activeBrowserRightSurfaceIntent?.sourceRequestId,
    closeCompetingRightSurfaces,
    consumePendingRequestsForSurface,
    manualRightSurface,
    pendingBrowserRightSurfaceIntent,
    refreshRightSurfacePendingRequests,
    setActiveBrowserRightSurfaceIntent,
    setManualRightSurface,
    setRightSurfaceBrowserTitle,
  ]);

  const handleToggleRightSurfaceFiles = useCallback(() => {
    if (!filesRightSurfaceAvailable) {
      return;
    }
    const shouldOpenFiles = manualRightSurface !== "files";
    closeCompetingRightSurfaces();
    setActiveFilesRightSurfaceTarget(
      shouldOpenFiles ? filesRightSurfaceTarget : null,
    );
    setManualRightSurface(shouldOpenFiles ? "files" : null);
    if (shouldOpenFiles) {
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface("files");
    } else {
      void dismissPendingRequestsForSurface("files", "user_closed_surface");
    }
  }, [
    closeCompetingRightSurfaces,
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
    filesRightSurfaceAvailable,
    filesRightSurfaceTarget,
    manualRightSurface,
    refreshRightSurfacePendingRequests,
    setActiveFilesRightSurfaceTarget,
    setManualRightSurface,
  ]);

  const handleToggleRightSurfaceShell = useCallback(() => {
    const shouldOpenShell = manualRightSurface !== "shell";
    closeCompetingRightSurfaces();
    setManualRightSurface(shouldOpenShell ? "shell" : null);
    if (shouldOpenShell) {
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface("shell");
    } else {
      void dismissPendingRequestsForSurface("shell", "user_closed_surface");
    }
  }, [
    closeCompetingRightSurfaces,
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
    manualRightSurface,
    refreshRightSurfacePendingRequests,
    setManualRightSurface,
  ]);

  const handleCloseRightSurfaceShell = useCallback(() => {
    setManualRightSurface((current) => (current === "shell" ? null : current));
    void dismissPendingRequestsForSurface("shell", "user_closed_surface");
  }, [dismissPendingRequestsForSurface, setManualRightSurface]);

  const handleToggleRightSurfaceBrowser = useCallback(() => {
    if (!browserRightSurfaceAvailable) {
      return;
    }
    const shouldOpenBrowser = manualRightSurface !== "browser";
    closeCompetingRightSurfaces();
    setManualRightSurface(shouldOpenBrowser ? "browser" : null);
    if (shouldOpenBrowser) {
      if (pendingBrowserRightSurfaceIntent) {
        setActiveBrowserRightSurfaceIntent(pendingBrowserRightSurfaceIntent);
        setRightSurfaceBrowserTitle(
          pendingBrowserRightSurfaceIntent.title?.trim() || null,
        );
      }
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface("browser");
    } else {
      void dismissPendingRequestsForSurface("browser", "user_closed_surface");
    }
  }, [
    browserRightSurfaceAvailable,
    closeCompetingRightSurfaces,
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
    manualRightSurface,
    pendingBrowserRightSurfaceIntent,
    refreshRightSurfacePendingRequests,
    setActiveBrowserRightSurfaceIntent,
    setManualRightSurface,
    setRightSurfaceBrowserTitle,
  ]);

  const handleRightSurfaceBrowserNavigate = useCallback(
    (_url: string, title?: string | null) => {
      setRightSurfaceBrowserTitle(title?.trim() || null);
    },
    [setRightSurfaceBrowserTitle],
  );

  const handleToggleRightSurfaceHarness = useCallback(() => {
    if (!rightSurfaceHarnessEnabled) {
      return;
    }
    const shouldOpenHarness = manualRightSurface !== "harness";
    closeCompetingRightSurfaces();
    setManualRightSurface(shouldOpenHarness ? "harness" : null);
    if (shouldOpenHarness) {
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface("harness");
    } else {
      void dismissPendingRequestsForSurface("harness", "user_closed_surface");
    }
  }, [
    closeCompetingRightSurfaces,
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
    manualRightSurface,
    refreshRightSurfacePendingRequests,
    rightSurfaceHarnessEnabled,
    setManualRightSurface,
  ]);

  const handleToggleRightSurfaceTrace = useCallback(() => {
    if (!rightSurfaceTraceAvailable) {
      return;
    }
    const shouldOpenTrace = manualRightSurface !== "trace";
    closeCompetingRightSurfaces();
    setManualRightSurface(shouldOpenTrace ? "trace" : null);
    if (shouldOpenTrace) {
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface("trace");
    } else {
      void dismissPendingRequestsForSurface("trace", "user_closed_surface");
    }
  }, [
    closeCompetingRightSurfaces,
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
    manualRightSurface,
    refreshRightSurfacePendingRequests,
    rightSurfaceTraceAvailable,
    setManualRightSurface,
  ]);

  const handleToggleExpertInfoPanel = useCallback(() => {
    setHarnessPanelVisible(false);
    setManualRightSurface(null);
    setActiveFilesRightSurfaceTarget(null);
    setActiveArticleWorkspace(null);
    const shouldOpenExpertInfo =
      expertInfoPanelCollapsed || sceneLayoutMode !== "chat";
    if (shouldOpenExpertInfo) {
      setLayoutMode("chat");
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface("expertInfo");
    } else {
      void dismissPendingRequestsForSurface(
        "expertInfo",
        "user_closed_surface",
      );
    }
    setExpertInfoPanelCollapsed(!shouldOpenExpertInfo);
  }, [
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
    expertInfoPanelCollapsed,
    refreshRightSurfacePendingRequests,
    sceneLayoutMode,
    setActiveArticleWorkspace,
    setActiveFilesRightSurfaceTarget,
    setExpertInfoPanelCollapsed,
    setHarnessPanelVisible,
    setLayoutMode,
    setManualRightSurface,
  ]);

  useEffect(() => {
    if (manualRightSurface === "harness" && !rightSurfaceHarnessEnabled) {
      setManualRightSurface(null);
    }
    if (manualRightSurface === "trace" && !rightSurfaceTraceAvailable) {
      setManualRightSurface(null);
    }
    if (manualRightSurface === "files" && !filesRightSurfaceAvailable) {
      setManualRightSurface(null);
      setActiveFilesRightSurfaceTarget(null);
    }
    if (
      manualRightSurface === "appSurface" &&
      !pluginSurfaceRightSurfaceAvailable
    ) {
      setManualRightSurface(null);
      setActivePluginSurfaces([]);
      setActivePluginSurfaceContainerId(null);
    }
    if (
      manualRightSurface === "articleWorkspace" &&
      !articleEditorRightSurfaceAvailable
    ) {
      setManualRightSurface(null);
      setActiveArticleWorkspace(null);
    }
    if (manualRightSurface === "browser" && !browserRightSurfaceAvailable) {
      setManualRightSurface(null);
    }
  }, [
    articleEditorRightSurfaceAvailable,
    browserRightSurfaceAvailable,
    filesRightSurfaceAvailable,
    manualRightSurface,
    pluginSurfaceRightSurfaceAvailable,
    rightSurfaceHarnessEnabled,
    rightSurfaceTraceAvailable,
    setActiveArticleWorkspace,
    setActiveFilesRightSurfaceTarget,
    setActivePluginSurfaceContainerId,
    setActivePluginSurfaces,
    setManualRightSurface,
  ]);

  const handleToggleCanvasFromRightSurface = useCallback(() => {
    if (manualRightSurface && sceneLayoutMode !== "chat") {
      void dismissPendingRequestsForSurface(
        manualRightSurface,
        "user_switched_surface",
      );
      setManualRightSurface(null);
      setActiveFilesRightSurfaceTarget(null);
      setActiveArticleWorkspace(null);
      return;
    }

    setHarnessPanelVisible(false);
    if (manualRightSurface) {
      void dismissPendingRequestsForSurface(
        manualRightSurface,
        "user_switched_surface",
      );
    }
    setActiveFilesRightSurfaceTarget(null);
    setActiveArticleWorkspace(null);
    setManualRightSurface(null);
    handleToggleCanvas();
  }, [
    dismissPendingRequestsForSurface,
    handleToggleCanvas,
    manualRightSurface,
    sceneLayoutMode,
    setActiveArticleWorkspace,
    setActiveFilesRightSurfaceTarget,
    setHarnessPanelVisible,
    setManualRightSurface,
  ]);

  const handleSelectRightSurfaceTab = useCallback(
    (kind: WorkspaceRightSurfaceKind) => {
      if (kind === rightSurfaceActiveSurface) {
        return;
      }

      setHarnessPanelVisible(false);
      setExpertInfoPanelCollapsed(kind !== "expertInfo");
      setActiveFilesRightSurfaceTarget(
        kind === "files" ? filesRightSurfaceTarget : null,
      );
      if (kind === "appSurface" && pluginSurfaceRightSurface) {
        setActivePluginSurfaces((current) =>
          mergeWorkspacePluginSurfaceDescriptors(current, [
            pluginSurfaceRightSurface,
          ]),
        );
        setActivePluginSurfaceContainerId(
          pluginSurfaceRightSurface.containerId,
        );
      }
      setActiveArticleWorkspace(
        kind === "articleWorkspace" && articleEditorRightSurface
          ? articleEditorRightSurface
          : null,
      );
      if (kind === "browser" && pendingBrowserRightSurfaceIntent) {
        setActiveBrowserRightSurfaceIntent(pendingBrowserRightSurfaceIntent);
        setRightSurfaceBrowserTitle(
          pendingBrowserRightSurfaceIntent.title?.trim() || null,
        );
      }
      setManualRightSurface(kind === "workbench" ? null : kind);
      void refreshRightSurfacePendingRequests();
      void consumePendingRequestsForSurface(kind);
    },
    [
      articleEditorRightSurface,
      consumePendingRequestsForSurface,
      filesRightSurfaceTarget,
      pendingBrowserRightSurfaceIntent,
      pluginSurfaceRightSurface,
      refreshRightSurfacePendingRequests,
      rightSurfaceActiveSurface,
      setActiveArticleWorkspace,
      setActiveBrowserRightSurfaceIntent,
      setActiveFilesRightSurfaceTarget,
      setActivePluginSurfaceContainerId,
      setActivePluginSurfaces,
      setExpertInfoPanelCollapsed,
      setHarnessPanelVisible,
      setManualRightSurface,
      setRightSurfaceBrowserTitle,
    ],
  );

  const handleSelectPluginSurface = useCallback(
    (surface: WorkspacePluginSurfaceDescriptor) => {
      setHarnessPanelVisible(false);
      setExpertInfoPanelCollapsed(true);
      setActivePluginSurfaceContainerId(surface.containerId);
      setManualRightSurface("appSurface");
    },
    [
      setActivePluginSurfaceContainerId,
      setExpertInfoPanelCollapsed,
      setHarnessPanelVisible,
      setManualRightSurface,
    ],
  );

  const handleClosePluginSurface = useCallback(
    (surface: WorkspacePluginSurfaceDescriptor) => {
      const result = closeWorkspacePluginSurfaceDescriptor({
        activeContainerId: activePluginSurfaceContainerId,
        containerId: surface.containerId,
        surfaces: pluginSurfaceRightSurfaces,
      });
      setActivePluginSurfaces(result.surfaces);
      setDismissedPluginSurfaceContainerIds((current) =>
        current.includes(surface.containerId)
          ? current
          : [...current, surface.containerId],
      );
      setActivePluginSurfaceContainerId(result.activeContainerId);
      if (result.surfaces.length === 0 && manualRightSurface === "appSurface") {
        setManualRightSurface(null);
        void dismissPendingRequestsForSurface(
          "appSurface",
          "user_closed_surface",
        );
      }
    },
    [
      activePluginSurfaceContainerId,
      dismissPendingRequestsForSurface,
      manualRightSurface,
      pluginSurfaceRightSurfaces,
      setActivePluginSurfaceContainerId,
      setActivePluginSurfaces,
      setDismissedPluginSurfaceContainerIds,
      setManualRightSurface,
    ],
  );

  return {
    handleClosePluginSurface,
    handleCloseRightSurfaceShell,
    handleRightSurfaceBrowserNavigate,
    handleSelectPluginSurface,
    handleSelectRightSurfaceTab,
    handleToggleCanvasFromRightSurface,
    handleToggleExpertInfoPanel,
    handleToggleRightSurfaceBrowser,
    handleToggleRightSurfaceFiles,
    handleToggleRightSurfaceHarness,
    handleToggleRightSurfaceShell,
    handleToggleRightSurfaceTrace,
  };
}
