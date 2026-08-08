/**
 * 应用主入口组件
 *
 * 管理页面路由和全局状态
 * 支持静态页面和动态插件页面路由
 * 包含启动画面和全局图标侧边栏
 *
 * _需求: 2.2, 3.2, 5.2_
 */

import React, { Suspense, lazy, useState, useCallback, useEffect } from "react";
import styled from "styled-components";
import { useTranslation } from "react-i18next";
import { withI18nPatch } from "./i18n/withI18nPatch";
import { AppPageContent } from "./components/AppPageContent";
import type { AgentBackgroundSessionRuntimeSnapshot } from "./components/agent/chat";
import { AppServerConfigWarningToastBridge } from "./components/AppServerConfigWarningToastBridge";
import { SplashScreen } from "./components/SplashScreen";
import { AppSidebar } from "./components/AppSidebar";
import { startupTracker } from "./lib/diagnostics/startupPerformance";
import { preloadDefaultProject } from "./lib/api/project";
import {
  ProjectType,
  createProject,
  isUserProjectType,
} from "./lib/api/project";
import { useDeepLink } from "./hooks/useDeepLink";
import { useSkillCatalogBootstrap } from "./hooks/useSkillCatalogBootstrap";
import { useServiceSkillCatalogBootstrap } from "./hooks/useServiceSkillCatalogBootstrap";
import { useSiteAdapterCatalogBootstrap } from "./hooks/useSiteAdapterCatalogBootstrap";
import { useAppNavigation } from "./hooks/useAppNavigation";
import { useAppShellLayout } from "./hooks/useAppShellLayout";
import { useAppStartupEffects } from "./hooks/useAppStartupEffects";
import { useGlobalTrayModelSync } from "./hooks/useGlobalTrayModelSync";
import { useClawTraceRegressionAlertMonitor } from "./hooks/useClawTraceRegressionAlertMonitor";
import { useOemLimeHubProviderSync } from "./hooks/useOemLimeHubProviderSync";
import { useSkillPackageOpenRequests } from "./hooks/useSkillPackageOpenRequests";
import { ComponentDebugProvider } from "./contexts/ComponentDebugContext";
import { ComponentDebugOverlay } from "./components/dev";
import {
  useResourceManagerNavigationIntents,
  type ResourceManagerNavigationDestination,
  type ResourceManagerNavigationIntent,
} from "./features/resource-manager";
import type { OpenDeepLinkPayload } from "./hooks/useDeepLink";
import { buildClawAgentParams } from "./lib/workspace/navigation";
import {
  resolveWebsiteInstalledSkillNavigation,
  resolveWebsiteOpenNavigation,
  resolveWebsiteSkillTitle,
} from "./lib/deepLink/websiteLaunch";
import { installOfficialMarketplaceSkill } from "./lib/api/officialSkillMarketplace";
import { toast } from "sonner";
import { SettingsTabs } from "./types/settings";
import { hasDesktopHostInvokeCapability } from "./lib/desktop-runtime";
import {
  hasNativeStartupScreen,
  hideNativeStartupOverlayWhenReady,
} from "./lib/nativeStartupScreen";
import { shouldReserveMacWindowControls } from "./lib/windowControls";
import { startWindowDragFromMouseEvent } from "./lib/windowDrag";
import {
  listenOpenVoiceModelSettingsRequest,
  persistVoiceModelSettingsFocusRequest,
} from "./lib/voiceModelSettingsNavigation";
import type { AgentPageParams } from "./types/page";

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  width: 100vw;
  background: var(--lime-app-bg, hsl(var(--background)));
  overflow: hidden;
`;

const MainContent = styled.main<{ $withSidebarGap?: boolean }>`
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding-left: ${(props) => (props.$withSidebarGap ? "10px" : "0")};
  background: var(--lime-app-bg, hsl(var(--background)));
`;

const WINDOW_DRAG_TOP_HEIGHT = 30;
const WINDOW_DRAG_TOP_HANDLE_WIDTH = 360;
const WINDOW_DRAG_EDGE_WIDTH = 8;
const WINDOW_DRAG_DEFAULT_SAFE_LEFT = 160;
const WINDOW_DRAG_MAC_SAFE_LEFT = 92;

const WindowDragLayer = styled.div`
  position: fixed;
  inset: 0;
  z-index: 1000;
  pointer-events: none;
`;

const WindowTopDragRegion = styled.div<{ $reserveMacWindowControls?: boolean }>`
  position: absolute;
  top: 0;
  left: ${({ $reserveMacWindowControls }) =>
    $reserveMacWindowControls
      ? `${WINDOW_DRAG_MAC_SAFE_LEFT}px`
      : `${WINDOW_DRAG_DEFAULT_SAFE_LEFT}px`};
  width: ${WINDOW_DRAG_TOP_HANDLE_WIDTH}px;
  height: ${WINDOW_DRAG_TOP_HEIGHT}px;
  pointer-events: auto;
  user-select: none;
  app-region: drag;
  -webkit-app-region: drag;
`;

const WindowSideDragRegion = styled.div<{ $side: "left" | "right" }>`
  position: absolute;
  top: ${WINDOW_DRAG_TOP_HEIGHT}px;
  bottom: 0;
  ${({ $side }) => $side}: 0;
  width: ${WINDOW_DRAG_EDGE_WIDTH}px;
  pointer-events: auto;
  user-select: none;
  app-region: drag;
  -webkit-app-region: drag;
`;

const RecentImageInsertFloating = lazy(() =>
  import("./components/image-gen/RecentImageInsertFloating").then((module) => ({
    default: module.RecentImageInsertFloating,
  })),
);
const CreateProjectDialog = lazy(() =>
  import("./components/projects/CreateProjectDialog").then((module) => ({
    default: module.CreateProjectDialog,
  })),
);
const ConnectConfirmDialog = lazy(() =>
  import("./components/connect").then((module) => ({
    default: module.ConnectConfirmDialog,
  })),
);

function AppContent() {
  startupTracker.mark("AppContent: render start");

  const { t } = useTranslation("common");
  const hasDesktopHostRuntime = hasDesktopHostInvokeCapability();
  const nativeStartupScreenAvailable = hasNativeStartupScreen();
  const reserveMacWindowControls = shouldReserveMacWindowControls();
  const [showSplash, setShowSplash] = useState(!nativeStartupScreenAvailable);
  const {
    currentPage,
    pageParams,
    requestedPage,
    requestedPageParams,
    handleNavigate,
  } = useAppNavigation();
  const [agentHasMessages, setAgentHasMessages] = useState(false);
  const [activeAgentSessionId, setActiveAgentSessionId] = useState<
    string | null
  >(null);
  const [activeAgentStreaming, setActiveAgentStreaming] = useState(false);
  const [backgroundAgentSessionRuntime, setBackgroundAgentSessionRuntime] =
    useState<AgentBackgroundSessionRuntimeSnapshot | null>(null);
  const activeAgentPage = requestedPage ?? currentPage;
  const activeAgentPageParams = requestedPageParams ?? pageParams;
  const activeAgentRouteSessionId =
    activeAgentPage === "agent"
      ? ((
          activeAgentPageParams as AgentPageParams | undefined
        )?.initialSessionId?.trim() ?? null)
      : null;

  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const [pendingRecommendation, setPendingRecommendation] = useState<{
    shortLabel: string;
    fullPrompt: string;
    projectType: ProjectType;
    projectName: string;
  } | null>(null);

  useSkillCatalogBootstrap();
  useServiceSkillCatalogBootstrap();
  useSiteAdapterCatalogBootstrap();
  useOemLimeHubProviderSync();
  useClawTraceRegressionAlertMonitor();
  const handleResourceManagerNavigationHandled = useCallback(
    ({
      destination,
    }: {
      intent: ResourceManagerNavigationIntent;
      destination: ResourceManagerNavigationDestination;
    }) => {
      toast.success(destination.noticeTitle, {
        description: destination.noticeDescription,
      });
    },
    [],
  );
  const handleResourceManagerNavigationUnsupported = useCallback(() => {
    toast.info(t("common.app.resourceNavigation.unsupported.title"), {
      description: t("common.app.resourceNavigation.unsupported.description"),
    });
  }, [t]);
  useResourceManagerNavigationIntents({
    onNavigate: handleNavigate,
    onHandled: handleResourceManagerNavigationHandled,
    onUnsupported: handleResourceManagerNavigationUnsupported,
  });
  useGlobalTrayModelSync({
    currentPage,
    pageParams,
  });
  useSkillPackageOpenRequests({
    onNavigate: handleNavigate,
  });
  useEffect(
    () =>
      listenOpenVoiceModelSettingsRequest((detail) => {
        persistVoiceModelSettingsFocusRequest(detail);
        handleNavigate("settings", { tab: SettingsTabs.MediaServices });
      }),
    [handleNavigate],
  );

  const _handleRequestRecommendation = useCallback(
    (shortLabel: string, fullPrompt: string, currentTheme: string) => {
      const themeLabels: Record<string, string> = {
        general: t("common.app.project.type.general"),
      };

      const prefix =
        themeLabels[currentTheme] || t("common.app.project.type.fallback");
      const projectName = t("common.app.project.recommendedName", {
        prefix,
        shortLabel,
      });

      setPendingRecommendation({
        shortLabel,
        fullPrompt,
        projectType: currentTheme as ProjectType,
        projectName,
      });
      setProjectDialogOpen(true);
    },
    [t],
  );

  const handleCreateProjectFromRecommendation = async (
    name: string,
    type: ProjectType,
    rootPath: string,
  ) => {
    const project = await createProject({
      name,
      rootPath,
      workspaceType: type,
    });

    if (pendingRecommendation) {
      handleNavigate(
        "agent",
        buildClawAgentParams({
          projectId: project.id,
          initialUserPrompt: pendingRecommendation.fullPrompt,
        }),
      );

      setPendingRecommendation(null);
    } else if (isUserProjectType(type)) {
      handleNavigate(
        "agent",
        buildClawAgentParams({
          projectId: project.id,
        }),
      );
    } else {
      handleNavigate("agent", {
        projectId: project.id,
      });
    }

    toast.success(t("common.app.project.created"));
  };

  const handleOpenBrowserConnectorSettings = useCallback(
    ({ enable }: { enable: boolean }) => {
      handleNavigate("settings", {
        tab: SettingsTabs.ChromeRelay,
      });

      if (enable) {
        toast.info(t("common.app.browserConnector.opened.title"), {
          description: t("common.app.browserConnector.opened.description"),
        });
      }
    },
    [handleNavigate, t],
  );

  const handleOpenWebsiteDeepLink = useCallback(
    async (payload: OpenDeepLinkPayload) => {
      if (payload.kind === "skill" && payload.action === "install") {
        const installNavigation =
          resolveWebsiteInstalledSkillNavigation(payload);
        if (!installNavigation) {
          toast.error(t("common.app.websiteDeepLink.unsupported.title"), {
            description: t(
              "common.app.websiteDeepLink.unsupported.description",
            ),
          });
          return;
        }

        const title = resolveWebsiteSkillTitle(payload.slug) ?? payload.slug;
        try {
          await installOfficialMarketplaceSkill(payload.slug, "lime");
          toast.success(
            t("common.app.websiteDeepLink.install.success.title", { title }),
            {
              description: t(
                "common.app.websiteDeepLink.install.success.description",
              ),
            },
          );
        } catch (error) {
          const message =
            error instanceof Error && error.message.trim()
              ? error.message.trim()
              : String(error);
          if (!/already exists|已存在/i.test(message)) {
            toast.error(t("common.app.websiteDeepLink.install.failed.title"), {
              description: t(
                "common.app.websiteDeepLink.install.failed.description",
                { message },
              ),
            });
            return;
          }

          toast.info(
            t("common.app.websiteDeepLink.install.alreadyInstalled.title", {
              title,
            }),
            {
              description: t(
                "common.app.websiteDeepLink.install.alreadyInstalled.description",
              ),
            },
          );
        }

        handleNavigate(installNavigation.page, installNavigation.params);
        return;
      }

      const resolved = resolveWebsiteOpenNavigation(payload);

      if (!resolved) {
        toast.error(t("common.app.websiteDeepLink.unsupported.title"), {
          description: t("common.app.websiteDeepLink.unsupported.description"),
        });
        return;
      }

      handleNavigate(resolved.page, resolved.params);
    },
    [handleNavigate, t],
  );

  const {
    connectPayload,
    relayInfo,
    isVerified,
    isDialogOpen,
    isSaving,
    error,
    handleConfirm,
    handleCancel,
  } = useDeepLink({
    onOpenBrowserConnectorSettings: handleOpenBrowserConnectorSettings,
    onOpenWebsiteDeepLink: handleOpenWebsiteDeepLink,
  });

  useAppStartupEffects({
    currentPage,
  });
  const { shouldShowAppSidebar, shouldAddMainContentGap } = useAppShellLayout({
    currentPage,
    pageParams,
    agentHasMessages,
  });
  const pageLoadingFallback = (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "hsl(var(--muted-foreground))",
        fontSize: "14px",
      }}
    >
      {t("common.app.loadingPage")}
    </div>
  );

  const handleSplashComplete = useCallback(() => {
    startupTracker.mark("SplashScreen: complete");
    setShowSplash(false);

    // Splash 完成后立即预加载默认项目
    preloadDefaultProject();
  }, []);

  useEffect(() => {
    if (showSplash || !nativeStartupScreenAvailable) {
      return;
    }

    startupTracker.mark("Native startup screen: renderer ready");
    preloadDefaultProject();
    hideNativeStartupOverlayWhenReady();
  }, [nativeStartupScreenAvailable, showSplash]);

  const handleWindowDragStart = useCallback(
    (event: React.MouseEvent<HTMLElement>) => {
      void startWindowDragFromMouseEvent(event, { source: "app_shell" });
    },
    [],
  );
  useEffect(() => {
    if (activeAgentPage !== "agent") {
      setActiveAgentSessionId(null);
      setActiveAgentStreaming(false);
      return;
    }

    setActiveAgentSessionId(activeAgentRouteSessionId);
  }, [activeAgentPage, activeAgentRouteSessionId]);

  const handleBackgroundSessionRuntimeChange = useCallback(
    (snapshot: AgentBackgroundSessionRuntimeSnapshot | null) => {
      const sessionId = snapshot?.sessionId.trim();
      if (!sessionId || !snapshot) {
        setBackgroundAgentSessionRuntime(null);
        return;
      }
      setBackgroundAgentSessionRuntime({ sessionId, status: snapshot.status });
    },
    [],
  );

  if (showSplash) {
    startupTracker.mark("AppContent: showing splash");
    return <SplashScreen onComplete={handleSplashComplete} />;
  }

  startupTracker.mark("AppContent: rendering main app");

  return (
    <ComponentDebugProvider>
      <AppContainer>
        {hasDesktopHostRuntime ? (
          <WindowDragLayer aria-hidden="true">
            <WindowTopDragRegion
              $reserveMacWindowControls={reserveMacWindowControls}
              data-lime-window-drag-region
              onMouseDown={handleWindowDragStart}
            />
            <WindowSideDragRegion
              $side="left"
              data-lime-window-drag-region
              onMouseDown={handleWindowDragStart}
            />
            <WindowSideDragRegion
              $side="right"
              data-lime-window-drag-region
              onMouseDown={handleWindowDragStart}
            />
          </WindowDragLayer>
        ) : null}
        {shouldShowAppSidebar && (
          <AppSidebar
            currentPage={currentPage}
            currentPageParams={pageParams}
            activeAgentSessionId={activeAgentSessionId}
            activeAgentStreaming={activeAgentStreaming}
            backgroundAgentSessionRuntime={backgroundAgentSessionRuntime}
            requestedPage={requestedPage}
            requestedPageParams={requestedPageParams}
            onNavigate={handleNavigate}
            onStartWindowDrag={handleWindowDragStart}
          />
        )}
        <MainContent
          $withSidebarGap={shouldAddMainContentGap}
          data-lime-window-drag-region
          onMouseDown={(event) => {
            void startWindowDragFromMouseEvent(event, {
              allowDescendantTargets: false,
              source: "main_content",
            });
          }}
        >
          <Suspense fallback={pageLoadingFallback}>
            <AppPageContent
              currentPage={currentPage}
              pageParams={pageParams}
              requestedPage={requestedPage}
              requestedPageParams={requestedPageParams}
              onNavigate={handleNavigate}
              onAgentHasMessagesChange={setAgentHasMessages}
              onAgentSessionChange={setActiveAgentSessionId}
              onAgentStreamingChange={setActiveAgentStreaming}
              onBackgroundSessionRuntimeChange={
                handleBackgroundSessionRuntimeChange
              }
            />
          </Suspense>
        </MainContent>
        <Suspense fallback={null}>
          <RecentImageInsertFloating onNavigate={handleNavigate} />
        </Suspense>

        <Suspense fallback={null}>
          <ConnectConfirmDialog
            open={isDialogOpen}
            relay={relayInfo}
            relayId={connectPayload?.relay ?? ""}
            apiKey={connectPayload?.key ?? ""}
            keyName={connectPayload?.name}
            isVerified={isVerified}
            isSaving={isSaving}
            error={error}
            onConfirm={handleConfirm}
            onCancel={handleCancel}
          />
        </Suspense>

        <Suspense fallback={null}>
          <CreateProjectDialog
            open={projectDialogOpen}
            onOpenChange={(open) => {
              setProjectDialogOpen(open);
              if (!open) {
                setPendingRecommendation(null);
              }
            }}
            onSubmit={handleCreateProjectFromRecommendation}
            defaultType={pendingRecommendation?.projectType}
            defaultName={pendingRecommendation?.projectName}
          />
        </Suspense>

        <ComponentDebugOverlay />
        <AppServerConfigWarningToastBridge />
      </AppContainer>
    </ComponentDebugProvider>
  );
}

const App = withI18nPatch(AppContent);
export default App;
